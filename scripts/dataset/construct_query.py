import os
import sys
sys.path.append(os.getcwd())
import json
from typing import Dict, List, Optional
from scripts.utils.call_llm import CallLLM
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def structure_template(area: str, task: str, abstract: str, scheme: str) -> Dict:
    """
    build structured query based on existing prompt template
    
    Args:
        area: research domain
        task: specific task
        abstract: paper abstract
        scheme: dataset format scheme
        
    Returns:
        Dict: structured query JSON
    """
    # build system prompt and user prompt
    system_prompt = """I will provide you with an abstract of a paper. Based on this abstract, the associated research domain and task and the dataset scheme, please extract the evaluation and testing characteristics from this research. Focus on identifying the test set properties, evaluation scenarios, and dataset requirements rather than the proposed methods. Using these extracted characteristics, construct a JSON query to search for datasets that match the testing requirements of this research. Response Format: 
    {
        "area": "Research Domain – Specifies the particular area of study, such as Natural Language Processing, Computer Vision, Machine Learning, and other major fields within AI research.",
        "task": "Detailed Task Description—First, explain the motivation and necessity for performing this task—why the task is important, what problem it aims to solve, and its value in a specific context. Further elaborate on the objectives and key functionalities of the task. Additionally, provide a detailed explanation of the task requirements, including a clear definition and description of the input as well as a detailed description of the expected output.",
        "scheme": "Instruction Fine-tuning Strategy - covering the organizational structure and format of datasets, such as the instruction-input-output triplet format, prompt-response dialogue format, and multi-turn dialogue structures.",
        "question":"Question Content Type - Describes the primary knowledge domains and content types covered by the questions in the test dataset, such as open-domain knowledge of film and entertainment, scientific common sense, history and geography, literature and arts, sports news, and professional technical fields.",
        "input":"Input Context Content - This specifies what the context for the input in the test dataset consists of, such as: the top 30 web search results for a given query, background paragraphs for a math problem, relevant document snippets, or dialogue history.",
        "output":"Output Answer Format - Describes the required format for answers in the test dataset, such as multiple-choice answers (A/B/C/D), True/False judgments, short-text answers, full-sentence answers, or numerical answers.",
        "language": "en/cn",
        "scale":"Dataset Scale - Describing the numble of samples in the dataset, such as 1K-10K, 10K-100K, 100K+, or specific quantity requirements.",
        "other_demands": "Other Special Requirements - Including task-specific demands such as long-text processing, real-time performance, domain-specific knowledge, multi-turn dialogue, tabular data, mathematical reasoning, and interpretability."
    }"""
    
    user_prompt = f"""The research belongs to the field of {area} and focuses on the task of {task}. Its abstract is as follows: {abstract}. And the dataset scheme is: {scheme}. Please extract the relevant keywords from the abstract and create the dataset search query in JSON response format."""
    
    # build structured query
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

def nl_query_transfer(structured_query: Dict) -> str:
    """
    convert structured query to natural language query
    """
    system_prompt = """You are an expert dataset search query generator with deep understanding of task specifications. I will provide you with task information in JSON format. Based on this information, especially the details within the "task" field, please generate a precise and natural language dataset search query intended for retrieval by a large language model. This query must focus on identifying or generating a dataset that aligns closely with the task's requirements and is suitable for supervised fine-tuning the model. Your query should comprehensively reflect the goals, coverage, examples, and expected output described in the "task" field. DO NOT include explanations or additional information in your response—only the query itself."""
    user_prompt = f"The task information is {structured_query}. Please carefully analyze the 'task' field and generate a highly relevant dataset search query based on its details. The query should specifically aim to identify or generate a dataset that meets the task's specified input, output, and functional requirements to enable efficient fine-tuning (SFT) of the model."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages

class QueryGenerationPipeline:
    """
    Query generation pipeline that processes benchmarks and generates natural language queries
    """
    
    def __init__(self, llm_caller: CallLLM):
        """
        Initialize the pipeline
        
        Args:
            llm_caller: Instance of CallLLM for making API calls
        """
        self.llm_caller = llm_caller
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
    
    def process_single_benchmark(self, benchmark: Dict) -> Dict:
        """
        Process a single benchmark to generate query
        
        Args:
            benchmark: Single benchmark data
            
        Returns:
            Dict: Processed benchmark with generated query
        """
        try:
            # Extract required fields
            area = benchmark.get("area_name", "")
            task = benchmark.get("task_name", "")
            abstract = benchmark.get("abstract", "")
            dataset = self._find_dataset(benchmark.get("bench_dataset", ""), benchmark.get("dataset_id", ""))
            
            # Generate scheme based on task characteristics
            scheme = self._generate_scheme(dataset)
            
            # Step 1: Generate structured query using structure_template
            logger.info(f"Generating structured query for {benchmark.get('bench_id', 'unknown')}")
            struct_messages = structure_template(area, task, abstract, scheme)
            
            # Call LLM to get structured query
            structured_response, prompt_tokens_1, completion_tokens_1 = self.llm_caller.post_request(struct_messages)
            
            # Update token counts
            self.total_prompt_tokens += prompt_tokens_1
            self.total_completion_tokens += completion_tokens_1
            self.total_tokens += (prompt_tokens_1 + completion_tokens_1)
            
            # Parse the structured response (assuming it returns JSON)
            try:
                if isinstance(structured_response, str):
                    # Try to extract JSON from the response
                    import re
                    json_match = re.search(r'\{.*\}', structured_response, re.DOTALL)
                    if json_match:
                        structured_query = json.loads(json_match.group())
                    else:
                        structured_query = json.loads(structured_response)
                else:
                    structured_query = structured_response
            except json.JSONDecodeError:
                logger.error(f"Failed to parse structured response for {benchmark.get('bench_id')}")
                structured_query = {
                    "area": area,
                    "task": task,
                    "scheme": scheme,
                    "question": "General domain questions",
                    "input": "Task-specific input",
                    "output": "Task-specific output",
                    "language": "en",
                    "scale": "10K-100K",
                    "other_demands": "Standard requirements"
                }
            
            # Step 2: Generate natural language query using nl_query_transfer
            logger.info(f"Generating natural language query for {benchmark.get('bench_id', 'unknown')}")
            nl_messages = nl_query_transfer(structured_query)
            
            # Call LLM to get natural language query
            nl_query, prompt_tokens_2, completion_tokens_2 = self.llm_caller.post_request(nl_messages)
            
            # Update token counts
            self.total_prompt_tokens += prompt_tokens_2
            self.total_completion_tokens += completion_tokens_2
            self.total_tokens += (prompt_tokens_2 + completion_tokens_2)
            
            # Create result dictionary
            result = benchmark.copy()
            result["dataset"] = dataset
            result["scheme"] = scheme
            result["structured_query"] = structured_query
            result["natural_language_query"] = nl_query
            result["processing_status"] = "success"
            result["tokens_used"] = {
                "struct_prompt_tokens": prompt_tokens_1,
                "struct_completion_tokens": completion_tokens_1,
                "nl_prompt_tokens": prompt_tokens_2,
                "nl_completion_tokens": completion_tokens_2,
                "total_tokens": prompt_tokens_1 + completion_tokens_1 + prompt_tokens_2 + completion_tokens_2
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing benchmark {benchmark.get('bench_id', 'unknown')}: {str(e)}")
            result = benchmark.copy()
            result["structured_query"] = None
            result["natural_language_query"] = None
            result["processing_status"] = "error"
            result["error_message"] = str(e)
            result["tokens_used"] = {
                "struct_prompt_tokens": 0,
                "struct_completion_tokens": 0,
                "nl_prompt_tokens": 0,
                "nl_completion_tokens": 0,
                "total_tokens": 0
            }
            return result
    
    def _find_dataset(self, bench_dataset: str, dataset_id) -> str:
        """
        Find dataset based on bench_dataset and dataset_id
        """
        dataset = None
        if dataset_id is not None:
            dataset = dataset_id.replace("/", "_")
        else:
            dataset = bench_dataset
        return dataset


    def _generate_scheme(self, dataset: str) -> str:
        """
        Generate appropriate scheme based on task and abstract
        
        Args:
            task: Task name
            abstract: Paper abstract
            
        Returns:
            str: Generated scheme
        """
        # Simple heuristic to determine scheme based on task type
        path = f"./batch_downloaded_datasets/{dataset}/process.json"
        if os.path.exists(path):
            with open(path, "r") as f:
                process_info = json.load(f)
            # extract a sample from it as scheme
            sample = process_info[0]
            return sample
        else:
            return "instruction-input-output format"
    
    def load_existing_results(self, output_file: str) -> List[Dict]:
        """
        Load existing results from output file if it exists
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            List[Dict]: Existing results or empty list
        """
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load existing results from {output_file}: {str(e)}")
                return []
        return []
    
    def save_single_result(self, result: Dict, output_file: str) -> None:
        """
        Save a single result to output file by appending to existing results
        
        Args:
            result: Single processed benchmark result
            output_file: Path to output JSON file
        """
        try:
            # Load existing results
            existing_results = self.load_existing_results(output_file)
            
            # Add new result
            existing_results.append(result)
            
            # Save updated results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Failed to save result to {output_file}: {str(e)}")
    
    def get_processed_bench_ids(self, output_file: str) -> set:
        """
        Get set of already processed benchmark IDs
        
        Args:
            output_file: Path to output JSON file
            
        Returns:
            set: Set of processed benchmark IDs
        """
        existing_results = self.load_existing_results(output_file)
        return {result.get('bench_id') for result in existing_results if result.get('bench_id')}
    
    def print_token_usage(self) -> None:
        """
        Print current token usage statistics
        """
        logger.info(f"=== Token Usage Statistics ===")
        logger.info(f"Total Prompt Tokens: {self.total_prompt_tokens:,}")
        logger.info(f"Total Completion Tokens: {self.total_completion_tokens:,}")
        logger.info(f"Total Tokens: {self.total_tokens:,}")
        logger.info(f"==============================")
    
    def process_benchmarks(self, input_file: str, output_file: str, skip_existing: bool = True) -> None:
        """
        Process all benchmarks from input file and save results to output file
        
        Args:
            input_file: Path to input JSON file containing benchmarks
            output_file: Path to output JSON file to save results
            skip_existing: Whether to skip already processed benchmarks
        """
        logger.info(f"Loading benchmarks from {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                benchmarks = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load benchmarks from {input_file}: {str(e)}")
            return
        
        if not isinstance(benchmarks, list):
            logger.error("Input file should contain a list of benchmarks")
            return
        
        # Get already processed benchmark IDs if skip_existing is True
        processed_ids = set()
        if skip_existing:
            processed_ids = self.get_processed_bench_ids(output_file)
            logger.info(f"Found {len(processed_ids)} already processed benchmarks")
        
        # Filter benchmarks to process
        benchmarks_to_process = []
        if skip_existing:
            benchmarks_to_process = [b for b in benchmarks if b.get('bench_id') not in processed_ids]
        else:
            benchmarks_to_process = benchmarks
            
        logger.info(f"Processing {len(benchmarks_to_process)} benchmarks (Total: {len(benchmarks)})")
        
        for i, benchmark in enumerate(benchmarks_to_process):
            bench_id = benchmark.get('bench_id', 'unknown')
            logger.info(f"Processing benchmark {i+1}/{len(benchmarks_to_process)}: {bench_id}")
            
            # Process single benchmark
            result = self.process_single_benchmark(benchmark)
            
            # Save result immediately
            self.save_single_result(result, output_file)
            
            # Log progress and token usage
            if result.get("processing_status") == "success":
                tokens_info = result.get("tokens_used", {})
                logger.info(f"✓ Successfully processed {bench_id} (Used {tokens_info.get('total_tokens', 0)} tokens)")
            else:
                logger.error(f"✗ Failed to process {bench_id}: {result.get('error_message', 'Unknown error')}")
            
            # Print token usage every 10 benchmarks
            if (i + 1) % 10 == 0:
                self.print_token_usage()
        
        # Final token usage statistics
        self.print_token_usage()
        logger.info(f"Processing completed. Results saved to {output_file}")
        
        # Load and display final statistics
        try:
            final_results = self.load_existing_results(output_file)
            stats = self.get_processing_statistics(final_results)
            logger.info(f"Final Processing Statistics: {stats}")
        except Exception as e:
            logger.error(f"Failed to load results for final statistics: {str(e)}")
    
    def get_processing_statistics(self, results: List[Dict]) -> Dict:
        """
        Get processing statistics from results
        
        Args:
            results: List of processed benchmark results
            
        Returns:
            Dict: Statistics about processing
        """
        total = len(results)
        successful = sum(1 for r in results if r.get("processing_status") == "success")
        failed = total - successful
        
        # Calculate total tokens used
        total_tokens_used = sum(r.get("tokens_used", {}).get("total_tokens", 0) for r in results)
        
        return {
            "total_benchmarks": total,
            "successful_processing": successful,
            "failed_processing": failed,
            "success_rate": successful / total if total > 0 else 0,
            "total_tokens_consumed": total_tokens_used
        }

def main():
    """
    Main function to run the query generation pipeline
    """
    # Initialize LLM caller (you may need to adjust this based on your CallLLM implementation)
    try:
        llm_caller = CallLLM(
            model="o3-mini",
            api_base="",
            api_key=''
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM caller: {str(e)}")
        return
    
    # Initialize pipeline
    pipeline = QueryGenerationPipeline(llm_caller)
    
    # Process benchmarks
    input_file = "./datasets/benchs.json"  # Input file path
    output_file = "./datasets/generated_queries.json"  # Output file path
    
    pipeline.process_benchmarks(input_file, output_file)
    
    # Load and display statistics
    try:
        final_results = pipeline.load_existing_results(output_file)
        stats = pipeline.get_processing_statistics(final_results)
        logger.info(f"Final Processing Statistics: {stats}")
        
        # Print final token usage
        pipeline.print_token_usage()
    except Exception as e:
        logger.error(f"Failed to load results for statistics: {str(e)}")

if __name__ == "__main__":
    main()