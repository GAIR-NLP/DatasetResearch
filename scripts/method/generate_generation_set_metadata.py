#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate search_metadata for generation data, based on original_metadata and 5 examples
refactored to class structure, support configuration file and CLI parameters
"""
import os
import sys
sys.path.append(os.getcwd())
import json
import argparse
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any
import yaml
from scripts.utils.call_llm import CallLLM
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MetadataGenerationConfig:
    """metadata generation configuration"""
    # input data configuration
    input_data_template: str = "LLaMA-Factory/data/deep_research_dataset/{model}/{dataset_id}.json"
    samples_count: int = 5
    models: List[str] = field(default_factory=lambda: ["o3-w", "o3-wo", "gemini", "grok", "openai"])
    
    # output path configuration
    output_paths: Dict[str, str] = field(default_factory=lambda: {
        "generation_metadata_template": "datasets/results/generation_metadata_{model}.json"
    })
    
    # generation parameters configuration
    generation_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_retries": 10,
        "retry_interval": 1,
        "samples_per_task": 5,
        "enable_incremental": True
    })
    
    # LLM configuration
    llm_config: Dict[str, Any] = field(default_factory=lambda: {
        "model_name": "gpt-4o-search-preview",
        "api_model": "o3", 
        "api_base": "",
        "api_key": "",
        "max_retries": 10,
        "retry_interval": 1
    })
    
    # prompt configuration
    prompts: Dict[str, str] = field(default_factory=lambda: {
        "generation_system_prompt": "You are an expert in dataset analysis and language model fine-tuning. You are given 5 representative input/output examples of a task instance. Please analyze and output the following metadata in JSON format (fields: introduction, task_type, input, output, source, example, samples_count).",
        "generation_user_template": """You are an expert in dataset analysis and language model fine-tuning.
Below is 5 representative input/output examples.
Please analyze and output the following metadata in JSON format (fields: introduction, task_type, input, output, source, example, samples_count).

Input/Output Examples:
{examples}

Please output in the following JSON format:
{{
    'introduction': 'task and area description of this task instance',
    'task': 'task type, you can only choose from the following types: text-generation, summarization, translation, question-answering, multiple-choice text-classification.',
    'question': 'Question Content Type - Describes the primary knowledge domains and content types covered by the questions in the test dataset, such as open-domain knowledge of film and entertainment, scientific common sense, history and geography, literature and arts, sports news, and professional technical fields.',
    'input': 'Structured retrieval results and contextual information - Input consists of formatted search results containing metadata fields such as descriptions, display URLs, titles, actual URLs, and ranking information, along with potential tabular data, document snippets, and conversational dialogue history for multi-turn scenarios.',
    'output': 'Direct factual answer format - Outputs are concise, definitive answers that directly address the question based on the provided context, formatted as complete statements such as \\'The answer is [specific fact]\\' for factual queries, numerical values for arithmetic problems, and explicit acknowledgment when questions cannot be answered.',
    'source': 'synthetic data',
    'example': 'give an input/output example',
    'samples_count': number of samples in the dataset
}}"""
    })
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'MetadataGenerationConfig':
        """load configuration from YAML file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # extract metadata generation related configuration
        metadata_gen_config = config_data.get('metadata_generation', {})
        llm_config = config_data.get('llm_config', {})
        prompts_config = config_data.get('prompts', {})
        
        return cls(
            input_data_template=metadata_gen_config.get('input_data_template', cls().input_data_template),
            samples_count=metadata_gen_config.get('samples_count', 5),
            models=metadata_gen_config.get('models', cls().models),
            output_paths=metadata_gen_config.get('output_paths', cls().output_paths),
            generation_params=metadata_gen_config.get('generation_params', cls().generation_params),
            llm_config=llm_config,
            prompts=prompts_config
        )
    
    def override_from_env(self):
        """override configuration from environment variables - disabled, completely rely on configuration file"""
        # no longer override configuration from environment variables, completely rely on configuration file
        pass


class LLMClient:
    """LLM client wrapper"""
    
    def __init__(self, config: MetadataGenerationConfig):
        self.config = config
        self.llm_caller = CallLLM(
            model=config.llm_config['api_model'],
            api_base=config.llm_config['api_base'], 
            api_key=config.llm_config['api_key']
        )
        logger.info(f"initialize LLM client: {config.llm_config['api_model']}")
    
    def generate_metadata(self, examples: List[Dict]) -> Dict:
        """generate metadata"""
        # construct prompt
        prompt = self.config.prompts['generation_user_template'].format(
            examples=json.dumps(examples, ensure_ascii=False, indent=2)
        )
        
        messages = [
            {"role": "system", "content": self.config.prompts['generation_system_prompt']},
            {"role": "user", "content": prompt}
        ]
        
        # retry logic
        max_retries = self.config.generation_params['max_retries']
        retry_interval = self.config.generation_params['retry_interval']
        
        for attempt in range(max_retries):
            try:
                # call LLM
                response, *_ = self.llm_caller.post_request(messages)
                
                # extract JSON
                json_content = self._extract_json_from_response(response)
                
                # try to parse JSON
                metadata = json.loads(json_content)
                logger.info(f"JSON parsed successfully, attempt {attempt + 1}")
                return metadata
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed, attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"waiting {retry_interval} seconds before retrying...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"JSON parsing failed, tried {max_retries} times")
                    logger.error(f"last raw response: {response[:300]}")
                    return {}
            except Exception as e:
                logger.error(f"call failed, tried {attempt + 1} times: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"waiting {retry_interval} seconds before retrying...")
                    time.sleep(retry_interval)
                else:
                    logger.error(f"call failed, tried {max_retries} times")
                    return {}
    
    def _extract_json_from_response(self, response: str) -> str:
        """extract JSON content from response"""
        if '```json' in response:
            start = response.find('```json') + 7
            end = response.find('```', start)
            return response[start:end].strip()
        elif '{' in response and '}' in response:
            start = response.find('{')
            end = response.rfind('}') + 1
            return response[start:end]
        else:
            return response


class DataProcessor:
    """data processing class"""
    
    @staticmethod
    def load_data(input_file: str) -> List[Dict[str, Any]]:
        """load input data"""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"loaded {len(data)} input data")
        return data
    
    @staticmethod
    def load_examples(generation_path: str, samples_count: int = 5) -> List[Dict[str, Any]]:
        """load example data"""
        with open(generation_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)
        return examples[:samples_count]
    
    @staticmethod
    def load_existing_results(output_file: str) -> List[Dict[str, Any]]:
        """load existing results"""
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    @staticmethod
    def save_results(output_file: str, results: List[Dict[str, Any]]):
        """save results to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"saved {len(results)} results to {output_file}")


class MetadataGenerator:
    """metadata generator main class"""
    
    def __init__(self, config: MetadataGenerationConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.data_processor = DataProcessor()
    
    def generate_metadata_for_model(self, input_file: str, output_file: str, model: str, dry_run: bool = False):
        """generate metadata for specified model"""
        logger.info(f"start generating metadata for model {model}")
        
        # load input data
        data = self.data_processor.load_data(input_file)
        
        # load existing results (for incremental processing)
        existing_results = self.data_processor.load_existing_results(output_file) if self.config.generation_params['enable_incremental'] else []
        existing_task_ids = {result['task_id'] for result in existing_results}
        
        # process each task
        for item in tqdm(data, desc=f"generating {model} metadata"):
            task_id = item.get('task_id')
            
            # check if it already exists (incremental processing)
            if self.config.generation_params['enable_incremental'] and task_id in existing_task_ids:
                # check if the existing result is valid
                existing_result = next(r for r in existing_results if r['task_id'] == task_id)
                if existing_result.get('search_metadata') not in ['Generation Failed', {}]:
                    logger.info(f"task_id {task_id} already exists, skip")
                    continue
                else:
                    # remove invalid results
                    existing_results = [r for r in existing_results if r['task_id'] != task_id]
                    existing_task_ids.remove(task_id)
            
            # generate metadata
            metadata_result = self._process_single_task(item, model)
            
            if dry_run:
                print(json.dumps(metadata_result, ensure_ascii=False, indent=2))
            else:
                existing_results.append(metadata_result)
                self.data_processor.save_results(output_file, existing_results)
    
    def _process_single_task(self, item: Dict[str, Any], model: str) -> Dict[str, Any]:
        """process single task"""
        task_id = item.get('task_id')
        original_metadata = item.get('original_metadata')
        dataset_id = item.get('search_dataset_id')
        
        try:
            # build generation data path
            if '/' in dataset_id:
                generation_path = self.config.input_data_template.format(
                    model=model, 
                    dataset_id=dataset_id.replace('/', '_')
                )
            else:
                generation_path = self.config.input_data_template.format(
                    model=model,
                    dataset_id=dataset_id.replace('/', '_')
                )
            
            # load example data
            examples = self.data_processor.load_examples(
                generation_path, 
                self.config.generation_params['samples_per_task']
            )
            
            if not original_metadata or not examples:
                logger.warning(f"missing original_metadata or examples, skip: {task_id}")
                return {
                    "task_id": task_id,
                    "original_metadata": original_metadata,
                    "examples": 'Generation Failed',
                    "search_metadata": 'Generation Failed'
                }
            
            # generate metadata
            search_metadata = self.llm_client.generate_metadata(examples)
            
            return {
                "task_id": task_id,
                "original_metadata": original_metadata,
                "search_metadata": search_metadata
            }
            
        except Exception as e:
            logger.error(f"error occurred when processing task {task_id}: {e}")
            return {
                "task_id": task_id,
                "original_metadata": original_metadata,
                "examples": 'Generation Failed',
                "search_metadata": 'Generation Failed'
            }


def main():
    parser = argparse.ArgumentParser(description='generate search_metadata for generation data, based on original_metadata and 5 examples')
    parser.add_argument('input_file', help='input JSON file, each element contains original_metadata and examples')
    parser.add_argument('output_file', help='output JSON file')
    parser.add_argument('model', help='model')
    parser.add_argument('--config', default='configs/evaluate_metadata_config.yaml', help='configuration file path')
    parser.add_argument('--dry-run', action='store_true', help='only print, do not write file')
    
    # LLM configuration override parameters
    parser.add_argument('--api-key', help='override API key')
    parser.add_argument('--api-base', help='override API base URL')
    parser.add_argument('--api-model', help='override API model')
    
    args = parser.parse_args()
    
    try:
        # load configuration
        config = MetadataGenerationConfig.from_yaml(args.config)
        config.override_from_env()
        
        # command line parameter override
        if args.api_key:
            config.llm_config['api_key'] = args.api_key
        if args.api_base:
            config.llm_config['api_base'] = args.api_base
        if args.api_model:
            config.llm_config['api_model'] = args.api_model
        
        # create generator and run
        generator = MetadataGenerator(config)
        generator.generate_metadata_for_model(
            input_file=args.input_file,
            output_file=args.output_file,
            model=args.model,
            dry_run=args.dry_run
        )
        
        logger.info("metadata generation completed")
        
    except Exception as e:
        logger.error(f"execution failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()