#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metadata evaluation tool
support configuration file and command line parameter calling
"""
import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

# add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.utils.config_loader import ConfigLoader
from scripts.utils.call_llm import CallLLM

@dataclass
class EvaluationConfig:
    """evaluation configuration class"""
    # input file configuration
    json_file_path: str
    
    # output path configuration
    output_dir: str
    output_filename_template: str
    final_results_filename: str
    
    # evaluation parameters configuration
    dimensions: List[str]
    score_range: Dict[str, int]
    batch_size: int
    save_intermediate: bool
    verbose: bool
    
    # LLM API configuration
    model_name: str
    api_model: str
    api_base: str
    api_key: str
    max_retries: int
    retry_interval: int
    
    # prompt configuration
    system_prompt: str
    comparison_template: str
    
    # statistics configuration
    detailed_stats: bool
    include_raw_responses: bool
    output_format: str
    
    @classmethod
    def from_config_file(cls, config_path: str, **override_params) -> 'EvaluationConfig':
        """create configuration object from configuration file"""
        config = ConfigLoader.load_config(config_path)
        
        # check environment variable override API key
        api_key = os.getenv('LLM_API_KEY', config['llm_config']['api_key'])
        
        params = {
            # input file configuration
            'json_file_path': config['input_files']['json_file_path'],
            
            # output path configuration
            'output_dir': config['output_paths']['output_dir'],
            'output_filename_template': config['output_paths']['output_filename_template'],
            'final_results_filename': config['output_paths']['final_results_filename'],
            
            # evaluation parameters configuration
            'dimensions': config['evaluation_params']['dimensions'],
            'score_range': config['evaluation_params']['score_range'],
            'batch_size': config['evaluation_params']['batch_size'],
            'save_intermediate': config['evaluation_params']['save_intermediate'],
            'verbose': config['evaluation_params']['verbose'],
            
            # LLM API configuration
            'model_name': config['llm_config']['model_name'],
            'api_model': config['llm_config']['api_model'],
            'api_base': config['llm_config']['api_base'],
            'api_key': api_key,
            'max_retries': config['llm_config']['max_retries'],
            'retry_interval': config['llm_config']['retry_interval'],
            
            # prompt configuration
            'system_prompt': config['prompts']['system_prompt'],
            'comparison_template': config['prompts']['comparison_template'],
            
            # statistics configuration
            'detailed_stats': config['statistics']['detailed_stats'],
            'include_raw_responses': config['statistics']['include_raw_responses'],
            'output_format': config['statistics']['output_format'],
        }
        
        # apply command line parameter overrides
        params.update(override_params)
        
        return cls(**params)
    
    def create_output_directories(self):
        """create output directories"""
        os.makedirs(self.output_dir, exist_ok=True)


class LLMClient:
    """LLM client wrapper"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = CallLLM(
            model=config.api_model,
            api_base=config.api_base,
            api_key=config.api_key
        )
        self.logger = logging.getLogger(__name__)
    
    def call_api(self, prompt: str) -> Tuple[Optional[str], int, int]:
        """call LLM API"""
        message = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response, prompt_tokens, completion_tokens = self.client.post_request(message)
            
            if self.config.verbose and response:
                self.logger.info(
                    f"Token usage - input: {prompt_tokens}, "
                    f"output: {completion_tokens}, total: {prompt_tokens + completion_tokens}"
                )
            
            return response, prompt_tokens, completion_tokens
            
        except Exception as e:
            self.logger.error(f"LLM API call failed: {e}")
            return None, 0, 0


class DataProcessor:
    """data processor class"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> Optional[List[Dict]]:
        """load JSON data"""
        try:
            with open(self.config.json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.logger.info(f"Successfully loaded data, total {len(data)} records")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            return None
    
    def parse_metadata(self, metadata_str: str) -> Optional[Dict]:
        """parse metadata string to dictionary"""
        try:
            metadata = json.loads(metadata_str)
            return metadata
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to parse metadata: {e}")
            return None
    
    def parse_evaluation_result(self, result_str: str) -> Optional[Dict]:
        """parse evaluation result"""
        try:
            # try to parse JSON directly
            result = json.loads(result_str)
            return result
        except json.JSONDecodeError:
            # if failed, try to extract JSON part
            try:
                start_idx = result_str.find('{')
                end_idx = result_str.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    json_str = result_str[start_idx:end_idx]
                    result = json.loads(json_str)
                    return result
            except:
                pass
            
            self.logger.error(f"Failed to parse evaluation result: {result_str}")
            return None
    
    def save_results(self, results: List[Dict], output_file: str) -> bool:
        """save results to file"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            if self.config.verbose:
                self.logger.info(f"✓ results saved to {output_file}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
            return False


class StatisticsCalculator:
    """statistics calculator"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_statistics(self, all_results: List[Dict]) -> Dict[str, Any]:
        """calculate statistics"""
        dimensions = self.config.dimensions
        
        # collect all dimension scores
        dimension_scores = {dim: [] for dim in dimensions}
        sample_averages = []
        
        for result in all_results:
            if 'evaluation' not in result or result['evaluation'] is None:
                continue
                
            eval_result = result['evaluation']
            sample_scores = []
            
            for dim in dimensions:
                if dim in eval_result and eval_result[dim] is not None:
                    score = eval_result[dim]
                    if isinstance(score, (int, float)) and self.config.score_range['min'] <= score <= self.config.score_range['max']:
                        dimension_scores[dim].append(score)
                        sample_scores.append(score)
            
            # calculate the average score of the sample
            if sample_scores:
                sample_avg = np.mean(sample_scores)
                sample_averages.append(sample_avg)
        
        # calculate the average score of each dimension
        dimension_averages = {}
        for dim in dimensions:
            if dimension_scores[dim]:
                dimension_averages[dim] = float(np.mean(dimension_scores[dim]))
            else:
                dimension_averages[dim] = None
        
        # calculate the overall average score
        overall_average = float(np.mean(sample_averages)) if sample_averages else None
        
        statistics = {
            'dimension_averages': dimension_averages,
            'overall_average': overall_average,
            'total_samples': len(all_results),
            'valid_samples': len(sample_averages),
            'dimension_counts': {dim: len(scores) for dim, scores in dimension_scores.items()}
        }
        
        if self.config.detailed_stats:
            # add detailed statistics
            statistics.update({
                'dimension_std': {
                    dim: float(np.std(scores)) if scores else None
                    for dim, scores in dimension_scores.items()
                },
                'dimension_min': {
                    dim: float(np.min(scores)) if scores else None
                    for dim, scores in dimension_scores.items()
                },
                'dimension_max': {
                    dim: float(np.max(scores)) if scores else None
                    for dim, scores in dimension_scores.items()
                }
            })
        
        return statistics
    
    def print_statistics(self, statistics: Dict[str, Any]):
        """print statistics"""
        self.logger.info(f"\n📊 statistics:")
        self.logger.info(f"total samples: {statistics['total_samples']}")
        self.logger.info(f"valid samples: {statistics['valid_samples']}")
        
        self.logger.info(f"\navg score of each dimension:")
        for dim, avg in statistics['dimension_averages'].items():
            count = statistics['dimension_counts'][dim]
            if avg is not None:
                self.logger.info(f"  {dim}: {avg:.2f} (valid samples: {count})")
            else:
                self.logger.info(f"  {dim}: None (valid samples: {count})")
        
        if statistics['overall_average'] is not None:
            self.logger.info(f"\ntotal avg score: {statistics['overall_average']:.2f}")
        else:
            self.logger.info(f"\ntotal avg score: None")


class MetadataEvaluator:
    """metadata evaluator main class"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.data_processor = DataProcessor(config)
        self.statistics_calculator = StatisticsCalculator(config)
        self.logger = logging.getLogger(__name__)
    
    def generate_comparison_prompt(self, original_metadata: str, search_metadata: str) -> str:
        """generate comparison prompt"""
        return self.config.comparison_template.format(
            original_metadata=original_metadata,
            search_metadata=search_metadata
        )
    
    def evaluate_single_pair(self, idx: int, item: Dict) -> Dict[str, Any]:
        """evaluate a single data pair"""
        task_id = item['task_id']
        original_metadata = item['original_metadata']
        search_metadata = item['search_metadata']
        
        if self.config.verbose:
            self.logger.info(f"\nprocessing the {idx+1}th data pair:")
            self.logger.info(f"original data: {task_id}")
        
        # generate comparison prompt
        prompt = self.generate_comparison_prompt(original_metadata, search_metadata)
        
        # call LLM for evaluation
        response, prompt_tokens, completion_tokens = self.llm_client.call_api(prompt)
        
        evaluation = None
        if response:
            # parse evaluation result
            evaluation = self.data_processor.parse_evaluation_result(response)
            
            if evaluation:
                if self.config.verbose:
                    self.logger.info(f"✓ successfully get evaluation result")
            else:
                self.logger.warning(f"✗ failed to parse evaluation result")
        else:
            self.logger.error(f"✗ API call failed")
        
        # construct result
        result = {
            'task_id': task_id,
            'evaluation': evaluation,
            'token_usage': {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens
            }
        }
        
        # whether to include raw response
        if self.config.include_raw_responses:
            result['raw_response'] = response
        
        return result
    
    def run(self) -> Dict[str, Any]:
        """run evaluation task"""
        # create output directory
        self.config.create_output_directories()
        
        # load data
        data = self.data_processor.load_data()
        if not data:
            raise ValueError("failed to load data")
        
        # determine output file path
        output_filename = self.config.output_filename_template.format(
            model_name=self.config.model_name
        )
        output_file = os.path.join(self.config.output_dir, output_filename)
        
        self.logger.info("start evaluating metadata matching...")
        
        # store all results
        all_results = []
        
        # process each data pair
        progress_bar = tqdm(enumerate(data), total=len(data), desc="evaluation progress", 
                           disable=not self.config.verbose)
        
        for idx, item in progress_bar:
            try:
                result = self.evaluate_single_pair(idx, item)
                all_results.append(result)
                
                # if intermediate save is enabled, save results immediately
                if self.config.save_intermediate:
                    self.data_processor.save_results(all_results, output_file)
                
                if self.config.verbose:
                    self.logger.info(f"✓ successfully processed and saved the {idx+1}th record")
                    
            except Exception as e:
                self.logger.error(f"error occurred when processing the {idx+1}th record: {e}")
                # continue processing the next record
                continue
        
        # calculate statistics
        self.logger.info(f"\n🎉 evaluation completed! calculating statistics...")
        statistics = self.statistics_calculator.calculate_statistics(all_results)
        
        # print statistics
        self.statistics_calculator.print_statistics(statistics)
        
        # save final results
        final_results = {
            'evaluation_results': all_results,
            'statistics': statistics,
            'config': {
                'model_name': self.config.model_name,
                'total_samples': len(data),
                'dimensions': self.config.dimensions,
                'score_range': self.config.score_range
            }
        }
        
        # save to specified file
        final_output_file = os.path.join(self.config.output_dir, self.config.final_results_filename)
        if self.data_processor.save_results(final_results, final_output_file):
            self.logger.info(f"✓ final results saved to {final_output_file}")
        
        return final_results


def setup_logging(verbose: bool = False) -> logging.Logger:
    """setup logging system"""
    log_level = logging.INFO if verbose else logging.WARNING
    
    # setup logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('metadata_evaluation.log', encoding='utf-8')
        ],
        force=True
    )
    
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="metadata evaluation tool, support configuration file and command line parameter calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
usage examples:
  # use configuration file
  python evaluate_metadata.py --config configs/evaluate_metadata_config.yaml
  
  # use configuration file and override some parameters
  python evaluate_metadata.py --config configs/evaluate_metadata_config.yaml --model-name gpt-4o --verbose
  
  # use command line parameters only
  python evaluate_metadata.py --json-file-path data/metadata.json --model-name gpt-4o-search-preview
        """
    )
    
    # configuration file parameters
    parser.add_argument(
        '--config', 
        type=str,
        help='configuration file path (YAML format)'
    )
    
    # input file parameters
    parser.add_argument(
        '--json-file-path',
        type=str,
        help='JSON file path of metadata to be evaluated'
    )
    
    # output path parameters
    parser.add_argument(
        '--output-dir',
        type=str,
        help='output directory'
    )
    
    parser.add_argument(
        '--output-filename-template',
        type=str,
        help='output filename template'
    )
    
    # LLM configuration parameters
    parser.add_argument(
        '--model-name',
        type=str,
        help='model name'
    )
    
    parser.add_argument(
        '--api-model',
        type=str,
        help='API model name'
    )
    
    parser.add_argument(
        '--api-base',
        type=str,
        help='API base URL'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='API key'
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        help='maximum number of retries'
    )
    
    # 评估参数
    parser.add_argument(
        '--batch-size',
        type=int,
        help='batch size'
    )
    
    parser.add_argument(
        '--save-intermediate',
        action='store_true',
        help='save intermediate results'
    )
    
    parser.add_argument(
        '--no-save-intermediate',
        action='store_true',
        help='do not save intermediate results'
    )
    
    # other parameters
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='show detailed output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='silent mode, only show error messages'
    )
    
    return parser.parse_args()


def main():
    """main function"""
    args = parse_arguments()
    
    # set logging
    verbose = args.verbose and not args.quiet
    logger = setup_logging(verbose)
    
    try:
        # prepare configuration parameters
        config_overrides = {}
        
        # extract non-None values from command line parameters
        for key, value in vars(args).items():
            if value is not None and key not in ['config', 'verbose', 'quiet']:
                # convert parameter name from hyphen format to underscore format
                config_key = key.replace('-', '_')
                config_overrides[config_key] = value
        
        # handle special logic for save_intermediate
        if args.no_save_intermediate:
            config_overrides['save_intermediate'] = False
        elif args.save_intermediate:
            config_overrides['save_intermediate'] = True
        
        # create configuration object
        if args.config:
            logger.info(f"use configuration file: {args.config}")
            config = EvaluationConfig.from_config_file(args.config, **config_overrides)
        else:
            # check required parameters
            required_params = ['json_file_path']
            missing_params = [p for p in required_params if p not in config_overrides]
            if missing_params:
                logger.error(f"missing required parameters: {missing_params}")
                logger.error("please provide configuration file or complete command line parameters")
                sys.exit(1)
            
            logger.info("use command line parameters to create configuration")
            # set default values for missing parameters
            defaults = {
                'output_dir': './evaluation/results/metadata_evaluation/',
                'output_filename_template': 'metadata_evaluation_{model_name}.json',
                'final_results_filename': 'final_evaluation_results.json',
                'dimensions': ['introduction', 'task', 'question', 'input', 'output', 'source', 'example', 'samples_count'],
                'score_range': {'min': 0, 'max': 10},
                'batch_size': 1,
                'save_intermediate': True,
                'verbose': verbose,
                'model_name': 'gpt-4o-search-preview',
                'api_model': 'o3',
                'api_base': 'https://gpt.yunstorm.com/',
                'api_key': os.getenv('LLM_API_KEY', ''),
                'max_retries': 10,
                'retry_interval': 1,
                'system_prompt': 'You are a professional dataset metadata comparison assistant that can evaluate the similarity between two dataset metadata descriptions across multiple dimensions.',
                'comparison_template': """I need you to compare two dataset metadata and score their matching degree across the following dimensions.

Dimension descriptions:
- introduction: Dataset introduction and overview
- task: Task type (e.g., text-classification, question-answering, summarization, text-generation, translation, etc.)
- question: Question Content Type - Describes the primary knowledge domains and content types covered by the questions in the test dataset, such as open-domain knowledge of film and entertainment, scientific common sense, history and geography, literature and arts, sports news, and professional technical fields.
- input: Description of input content
- output: Description of output content
- source: Data source (e.g., human-generated, machine-generated, etc.)
- example: Sample data
- samples_count: Number of samples

Original dataset metadata:
{original_metadata}

Search dataset metadata:
{search_metadata}

Please score each dimension on a scale of 0-10 for matching degree, where:
- 10 points: Complete match or highly similar
- 0 points: Complete mismatch or opposite
- Output an integer score. If a dimension is missing or meaningless in one or both metadata, mark it as null

Please output the result strictly in the following JSON format:

{{
    "introduction": score or null,
    "task": score or null,
    "question": score or null,
    "input": score or null,
    "output": score or null,
    "source": score or null,
    "example": score or null,
    "samples_count": score or null,
    "average": average score (excluding null values) or null
}}

Note:
1. Only output JSON format, do not include any other text
2. Scores must be numbers between 0-10 or null
3. average is the mean of all non-null scores""",
                'detailed_stats': True,
                'include_raw_responses': True,
                'output_format': 'json'
            }
            
            # apply default values
            for key, default_value in defaults.items():
                if key not in config_overrides:
                    config_overrides[key] = default_value
            
            config = EvaluationConfig(**config_overrides)
        
        # validate configuration
        if not config.api_key:
            logger.error("API key is not set. Please set it in the configuration file or provide it through the environment variable LLM_API_KEY")
            sys.exit(1)
        
        if not os.path.exists(config.json_file_path):
            logger.error(f"input file does not exist: {config.json_file_path}")
            sys.exit(1)
        
        # create and run evaluator
        evaluator = MetadataEvaluator(config)
        logger.info("🎯 start metadata evaluation task")
        
        results = evaluator.run()
        
        logger.info("🎉 metadata evaluation task completed!")
        
        # return exit code
        if results['statistics']['valid_samples'] > 0:
            sys.exit(0)  # success
        else:
            sys.exit(1)  # no valid samples
        
    except KeyboardInterrupt:
        logger.info("\nuser interrupted execution")
        sys.exit(130)
    except Exception as e:
        logger.error(f"error occurred during execution: {e}")
        if verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()