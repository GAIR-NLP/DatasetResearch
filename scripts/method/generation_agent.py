#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synthetic data generation
support configuration file and command line parameter invocation
"""
import json
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from openai import AzureOpenAI
from tqdm import tqdm
import concurrent.futures
import threading

# add project root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.utils.config_loader import ConfigLoader

@dataclass
class GenerationConfig:
    """generation configuration class"""
    # input file configuration
    agent_query_file: str
    metadata_file: str
    
    # output path configuration
    w_example_output_dir: str
    wo_example_output_dir: str
    results_filename: str
    results_output_dir: str
    
    # generation parameter configuration
    num_data: int
    max_workers: int
    max_json_retries: int
    max_llm_retries: int
    
    # Azure OpenAI configuration
    api_endpoint: str
    api_key: str
    api_version: str
    model: str
    
    # data processing configuration
    min_file_size: int
    skip_existing: bool
    verbose: bool
    
    # prompt configuration
    system_prompt: str
    with_example_template: str
    without_example_template: str
    
    @classmethod
    def from_config_file(cls, config_path: str, **override_params) -> 'GenerationConfig':
        """create configuration object from configuration file"""
        config = ConfigLoader.load_config(config_path)
        
        # directly use API key from configuration file
        api_key = config['azure_openai']['api_key']
        
        params = {
            # input file configuration
            'agent_query_file': config['input_files']['agent_query_file'],
            'metadata_file': config['input_files']['metadata_file'],
            
            # output path configuration
            'w_example_output_dir': config['output_paths']['w_example_output_dir'],
            'wo_example_output_dir': config['output_paths']['wo_example_output_dir'],
            'results_filename': config['output_paths']['results_filename'],
            'results_output_dir': config['output_paths'].get('results_output_dir', os.path.join(os.path.dirname(config['input_files']['metadata_file']), 'results')),
            
            # generation parameter configuration
            'num_data': config['generation_params']['num_data'],
            'max_workers': config['generation_params']['max_workers'],
            'max_json_retries': config['generation_params']['max_json_retries'],
            'max_llm_retries': config['generation_params']['max_llm_retries'],
            
            # Azure OpenAI configuration
            'api_endpoint': config['azure_openai']['api_endpoint'],
            'api_key': api_key,
            'api_version': config['azure_openai']['api_version'],
            'model': config['azure_openai']['model'],
            
            # data processing configuration
            'min_file_size': config['data_processing']['min_file_size'],
            'skip_existing': config['data_processing']['skip_existing'],
            'verbose': config['data_processing']['verbose'],
            
            # prompt configuration
            'system_prompt': config['prompts']['system_prompt'],
            'with_example_template': config['prompts']['with_example_template'],
            'without_example_template': config['prompts']['without_example_template'],
        }
        
        # apply command line parameter override
        params.update(override_params)
        
        return cls(**params)
    
    def create_output_directories(self, model_name: str):
        """create output directory, including model sub-directory"""
        # build model sub-directory path
        self.w_example_output_dir = os.path.join(self.w_example_output_dir, model_name)
        self.wo_example_output_dir = os.path.join(self.wo_example_output_dir, model_name)
        
        os.makedirs(self.w_example_output_dir, exist_ok=True)
        os.makedirs(self.wo_example_output_dir, exist_ok=True)


class AzureOpenAIClient:
    """Azure OpenAI client wrapper"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = AzureOpenAI(
            azure_endpoint=config.api_endpoint,
            api_key=config.api_key,
            api_version=config.api_version
        )
        self.logger = logging.getLogger(__name__)
    
    def call_api(self, prompt: str, max_retries: Optional[int] = None) -> Optional[str]:
        """call Azure OpenAI API"""
        max_retries = max_retries or self.config.max_llm_retries
        
        conversation = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=conversation
                )
                
                result = response.choices[0].message.content
                if self.config.verbose:
                    thread_id = threading.current_thread().ident
                    self.logger.info(
                        f"[thread {thread_id}] token usage - input: {response.usage.prompt_tokens}, "
                        f"output: {response.usage.completion_tokens}, total: {response.usage.total_tokens}"
                    )
                return result
                
            except Exception as e:
                retry_count += 1
                thread_id = threading.current_thread().ident
                error_msg = str(e)
                self.logger.warning(
                    f"[thread {thread_id}] Azure API call failed, retry {retry_count}/{max_retries}: {error_msg}"
                )
                
                if retry_count >= max_retries:
                    self.logger.error(
                        f"[thread {thread_id}] reached maximum retry count ({max_retries}), still failed: {error_msg}"
                    )
                    return None
                
                wait_time = 1
                self.logger.info(f"[thread {thread_id}] waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
        
        return None


class DataProcessor:
    """data processor class"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # thread lock, for thread-safe statistics
        self.lock = threading.Lock()
        self.success_count_w_example = 0
        self.success_count_wo_example = 0
    
    def load_data(self) -> Tuple[List[Dict], Dict[str, Any]]:
        """load metadata data (contains query information)"""
        self.logger.info("loading data file...")
        
        try:
            # load metadata file (contains query information)
            with open(self.config.metadata_file, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
            self.logger.info(f"successfully loaded metadata data, containing {len(metadata_data)} records")
            
            # convert to agent_query_data format and create metadata mapping
            agent_query_data = []
            metadata_dict = {}
            
            for item in metadata_data:
                dataset_id = item['dataset_id']
                agent_query = item.get('query', '')
                
                # convert to agent query format
                agent_query_data.append({
                    'dataset_id': dataset_id,
                    'agent_query': agent_query
                })
                
                # parse metadata to JSON object (supports both string and object formats)
                if 'metadata' in item:
                    metadata_data = item['metadata']
                    try:
                        if isinstance(metadata_data, str):
                            # if it is a string, parse it to a JSON object
                            metadata_obj = json.loads(metadata_data)
                        else:
                            # if it is already an object (dictionary), use it directly
                            metadata_obj = metadata_data
                        metadata_dict[dataset_id] = metadata_obj
                    except (json.JSONDecodeError, TypeError) as e:
                        self.logger.warning(f"failed to parse metadata for {dataset_id}: {e}")
            
            self.logger.info(f"successfully converted {len(agent_query_data)} agent query records")
            self.logger.info(f"successfully parsed {len(metadata_dict)} metadata records")
            return agent_query_data, metadata_dict
            
        except Exception as e:
            self.logger.error(f"failed to load data file: {e}")
            raise

    def check_file_exists_and_valid(self, dataset_id: str, output_dir: str) -> bool:
        """
        check if the JSON file of the specified dataset exists and contains valid content
        
        Args:
            dataset_id (str): dataset ID
            output_dir (str): output directory path
        
        Returns:
            bool: True if the file exists and contains valid content, False otherwise
        """
        filename = dataset_id.replace('/', '_') + '.json'
        filepath = os.path.join(output_dir, filename)
        
        try:
            # check if the file exists
            if not os.path.exists(filepath):
                return False
            
            # check file size (empty or too small file is considered invalid)
            if os.path.getsize(filepath) < self.config.min_file_size:
                return False
            
            # try to read and parse JSON file
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # check if it is a valid list and not empty
            if isinstance(data, list) and len(data) > 0:
                # check if the elements in the list have input and output fields
                if all(isinstance(item, dict) and 'input' in item and 'output' in item for item in data):
                    return True
            
            return False
            
        except (json.JSONDecodeError, IOError, Exception) as e:
            # if reading or parsing fails, consider the file invalid
            self.logger.warning(f"error occurred when checking file {filepath}: {e}")
            return False

    def preprocess_datasets(self, agent_query_data: List[Dict], metadata_dict: Dict[str, Any]) -> Tuple[List[Dict], List[Dict], Dict[str, int]]:
        """
        preprocess datasets, check file existence and generate the list of datasets to be processed
        
        Args:
            agent_query_data: agent query data
            metadata_dict: metadata dictionary
        
        Returns:
            tuple: (w_example_todo_list, wo_example_todo_list, skip_stats)
        """
        self.logger.info("🔍 preprocessing datasets, checking file existence...")
        
        w_example_todo = []
        wo_example_todo = []
        w_example_skipped = 0
        wo_example_skipped = 0
        no_metadata_count = 0
        
        for item in tqdm(agent_query_data, desc="checking file existence", disable=not self.config.verbose):
            dataset_id = item['dataset_id']
            agent_query = item['agent_query']
            
            # check if there is corresponding metadata
            if dataset_id not in metadata_dict:
                self.logger.warning(f"⚠️ no metadata found for {dataset_id}, skip")
                no_metadata_count += 1
                continue
            
            metadata = metadata_dict[dataset_id]
            example_data = metadata.get('example', '')
            
            # check w_example file (if skip_existing is enabled)
            if self.config.skip_existing:
                w_exists = self.check_file_exists_and_valid(dataset_id, self.config.w_example_output_dir)
                if w_exists:
                    self.logger.info(f"⏭️ [W] {dataset_id}: w_example file exists and is valid, skip")
                    w_example_skipped += 1
                else:
                    w_example_todo.append({
                        'dataset_id': dataset_id,
                        'agent_query': agent_query,
                        'example_data': example_data
                    })
            else:
                w_example_todo.append({
                    'dataset_id': dataset_id,
                    'agent_query': agent_query,
                    'example_data': example_data
                })
            
            # check wo_example file (if skip_existing is enabled)
            if self.config.skip_existing:
                wo_exists = self.check_file_exists_and_valid(dataset_id, self.config.wo_example_output_dir)
                if wo_exists:
                    self.logger.info(f"⏭️ [WO] {dataset_id}: wo_example file exists and is valid, skip")
                    wo_example_skipped += 1
                else:
                    wo_example_todo.append({
                        'dataset_id': dataset_id,
                        'agent_query': agent_query
                    })
            else:
                wo_example_todo.append({
                    'dataset_id': dataset_id,
                    'agent_query': agent_query
                })
        
        skip_stats = {
            'w_example_skipped': w_example_skipped,
            'wo_example_skipped': wo_example_skipped,
            'no_metadata_count': no_metadata_count
        }
        
        self.logger.info(f"\n📊 preprocessing completed:")
        self.logger.info(f"   - datasets to generate W data: {len(w_example_todo)}")
        self.logger.info(f"   - datasets to generate WO data: {len(wo_example_todo)}")
        self.logger.info(f"   - W data already exists, skipped: {w_example_skipped}")
        self.logger.info(f"   - WO data already exists, skipped: {wo_example_skipped}")
        self.logger.info(f"   - datasets with missing metadata, skipped: {no_metadata_count}")
        
        return w_example_todo, wo_example_todo, skip_stats

    def save_generated_data(self, search_dataset_id: str, generated_data: List[Dict], output_dir: str, data_type: str) -> bool:
        """save generated data to specified directory"""
        # use search_dataset_id as filename
        filename = search_dataset_id + '.json'
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(generated_data, f, ensure_ascii=False, indent=2)
            thread_id = threading.current_thread().ident
            self.logger.info(f"[thread {thread_id}] ✅ [{data_type}] {search_dataset_id}: data saved to: {filepath}")
            return True
        except Exception as e:
            thread_id = threading.current_thread().ident
            self.logger.error(f"[thread {thread_id}] ❌ [{data_type}] {search_dataset_id}: failed to save data: {e}")
            return False


class GenerationAgent:
    """generation agent main class"""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.client = AzureOpenAIClient(config)
        self.processor = DataProcessor(config)
        self.logger = logging.getLogger(__name__)
    
    def generate_prompt_with_example(self, agent_query: str, example_data: str) -> str:
        """generate prompt with example"""
        return self.config.with_example_template.format(
            agent_query=agent_query,
            example_data=example_data,
            num_data=self.config.num_data
        )

    def generate_prompt_without_example(self, agent_query: str) -> str:
        """generate prompt without example"""
        return self.config.without_example_template.format(
            agent_query=agent_query,
            num_data=self.config.num_data
        )

    def extract_json_from_response(self, response_text: str) -> Optional[List[Dict]]:
        """extract JSON list from API response"""
        try:
            # try to parse the entire response directly
            return json.loads(response_text)
        except json.JSONDecodeError:
            # if failed, try to find JSON array
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx+1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    thread_id = threading.current_thread().ident
                    self.logger.error(f"[thread {thread_id}] failed to parse JSON: {json_str[:200]}...")
                    return None
            
            thread_id = threading.current_thread().ident
            self.logger.error(f"[thread {thread_id}] no valid JSON array found in response: {response_text[:200]}...")
            return None

    def generate_data_with_retries(self, prompt_func, dataset_id: str, data_type: str) -> Tuple[bool, Optional[List[Dict]]]:
        """
        data generation function with JSON parsing retry
        
        Args:
            prompt_func: function to generate prompt
            dataset_id: dataset ID
            data_type: data type description (W or WO)
        
        Returns:
            tuple: (success_flag, generated_data)
        """
        thread_id = threading.current_thread().ident
        
        for attempt in range(self.config.max_json_retries):
            self.logger.info(
                f"[thread {thread_id}] 🔄 [{data_type}] {dataset_id}: generating data... "
                f"(attempt {attempt + 1}/{self.config.max_json_retries})"
            )
            
            # generate prompt and call API
            prompt = prompt_func()
            response = self.client.call_api(prompt)
            
            if not response:
                self.logger.warning(f"[thread {thread_id}] ❌ [{data_type}] {dataset_id}: API call failed (attempt {attempt + 1})")
                continue
            
            # try to parse JSON
            parsed_json = self.extract_json_from_response(response)
            
            if parsed_json and isinstance(parsed_json, list):
                self.logger.info(
                    f"[thread {thread_id}] ✅ [{data_type}] {dataset_id}: JSON parsing successful, containing {len(parsed_json)} samples"
                )
                return True, parsed_json
            else:
                self.logger.warning(
                    f"[thread {thread_id}] ❌ [{data_type}] {dataset_id}: format parsing failed "
                    f"(attempt {attempt + 1}/{self.config.max_json_retries})"
                )
                if attempt < self.config.max_json_retries - 1:
                    wait_time = 1
                    self.logger.info(f"[thread {thread_id}] waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
        
        self.logger.error(f"[thread {thread_id}] ❌ [{data_type}] {dataset_id}: final generation failed, reached maximum retry count")
        return False, None

    def process_w_example_dataset(self, item: Dict) -> Dict[str, Any]:
        """function to process dataset with example data"""
        dataset_id = item['dataset_id']
        agent_query = item['agent_query']
        example_data = item['example_data']
        thread_id = threading.current_thread().ident
        
        # generate synthesis type search_dataset_id
        model_name = self.config.model.replace('/', '_').replace('-', '_')
        search_dataset_id = f"synthesis_{dataset_id.replace('/', '_')}_{model_name}"
        
        self.logger.info(f"[thread {thread_id}] 🔄 [W] {dataset_id}: starting to process dataset with example data")
        
        # generate data with reference example (with retry)
        w_example_prompt_func = lambda: self.generate_prompt_with_example(agent_query, example_data)
        success, w_example_json = self.generate_data_with_retries(
            w_example_prompt_func, 
            dataset_id, 
            "W"
        )
        
        if success and w_example_json:
            if self.processor.save_generated_data(search_dataset_id, w_example_json, self.config.w_example_output_dir, "W"):
                with self.processor.lock:
                    self.processor.success_count_w_example += 1
                self.logger.info(f"[thread {thread_id}] ✅ [W] {dataset_id}: final generation successful")
                return {"dataset_id": dataset_id, "search_dataset_id": search_dataset_id, "success": True, "error": None}
            else:
                self.logger.error(f"[thread {thread_id}] ❌ [W] {dataset_id}: save failed")
                return {"dataset_id": dataset_id, "search_dataset_id": search_dataset_id, "success": False, "error": "save failed"}
        else:
            self.logger.error(f"[thread {thread_id}] ❌ [W] {dataset_id}: generation failed")
            return {"dataset_id": dataset_id, "search_dataset_id": search_dataset_id, "success": False, "error": "生成失败"}

    def process_wo_example_dataset(self, item: Dict) -> Dict[str, Any]:
        """function to process dataset without example data"""
        dataset_id = item['dataset_id']
        agent_query = item['agent_query']
        thread_id = threading.current_thread().ident
        
        # generate synthesis type search_dataset_id
        model_name = self.config.model.replace('/', '_').replace('-', '_')
        search_dataset_id = f"synthesis_{dataset_id.replace('/', '_')}_{model_name}"
        
        self.logger.info(f"[thread {thread_id}] 🔄 [WO] {dataset_id}: starting to process dataset without example data")
        
        # generate data without reference example (with retry)
        wo_example_prompt_func = lambda: self.generate_prompt_without_example(agent_query)
        success, wo_example_json = self.generate_data_with_retries(
            wo_example_prompt_func, 
            dataset_id, 
            "WO"
        )
        
        if success and wo_example_json:
            if self.processor.save_generated_data(search_dataset_id, wo_example_json, self.config.wo_example_output_dir, "WO"):
                with self.processor.lock:
                    self.processor.success_count_wo_example += 1
                self.logger.info(f"[thread {thread_id}] ✅ [WO] {dataset_id}: final generation successful")
                return {"dataset_id": dataset_id, "search_dataset_id": search_dataset_id, "success": True, "error": None}
            else:
                self.logger.error(f"[thread {thread_id}] ❌ [WO] {dataset_id}: save failed")
                return {"dataset_id": dataset_id, "search_dataset_id": search_dataset_id, "success": False, "error": "保存失败"}
        else:
            self.logger.error(f"[thread {thread_id}] ❌ [WO] {dataset_id}: generation failed")
            return {"dataset_id": dataset_id, "search_dataset_id": search_dataset_id, "success": False, "error": "generation failed"}

    def run_concurrent_generation_w(self, w_example_todo: List[Dict]) -> List[Dict]:
        """concurrently run generation with example data"""
        if not w_example_todo:
            self.logger.info("🔍 no datasets to generate W data, skip")
            return []
        
        self.logger.info(
            f"🚀 [W] starting concurrent processing with example data, using {self.config.max_workers} threads"
        )
        self.logger.info(f"📝 [W] maximum JSON parsing retry count: {self.config.max_json_retries}")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # submit all tasks and use tqdm to display progress
            future_to_item = {executor.submit(self.process_w_example_dataset, item): item for item in w_example_todo}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                              total=len(w_example_todo), 
                              desc="[W] processing dataset with example data",
                              disable=not self.config.verbose):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    self.logger.error(f"❌ [W] error occurred when processing dataset {item['dataset_id']}: {e}")
                    results.append({
                        "dataset_id": item['dataset_id'], 
                        "success": False,
                        "error": str(e)
                    })
        
        return results

    def run_concurrent_generation_wo(self, wo_example_todo: List[Dict]) -> List[Dict]:
        """concurrently run generation without example data"""
        if not wo_example_todo:
            self.logger.info("🔍 no datasets to generate WO data, skip")
            return []
        
        self.logger.info(
            f"🚀 [WO] starting concurrent processing without example data, using {self.config.max_workers} threads"
        )
        self.logger.info(f"📝 [WO] maximum JSON parsing retry count: {self.config.max_json_retries}")
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # submit all tasks and use tqdm to display progress
            future_to_item = {executor.submit(self.process_wo_example_dataset, item): item for item in wo_example_todo}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                              total=len(wo_example_todo), 
                              desc="[WO] processing dataset without example data",
                              disable=not self.config.verbose):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    item = future_to_item[future]
                    self.logger.error(f"❌ [WO] error occurred when processing dataset {item['dataset_id']}: {e}")
                    results.append({
                        "dataset_id": item['dataset_id'], 
                        "success": False,
                        "error": str(e)
                    })
        
        return results

    def run(self) -> Dict[str, Any]:
        """run generation tasks"""
        # create output directory
        model_name = self.config.model.replace('/', '_').replace('-', '_')
        self.config.create_output_directories(model_name)
        
        # load data
        agent_query_data, metadata_dict = self.processor.load_data()
        
        self.logger.info(f"successfully loaded {len(agent_query_data)} agent queries")
        self.logger.info(f"successfully loaded {len(metadata_dict)} metadata records")
        
        # preprocess datasets, check file existence
        w_example_todo, wo_example_todo, skip_stats = self.processor.preprocess_datasets(agent_query_data, metadata_dict)
        
        # concurrently process W and WO data
        w_results = self.run_concurrent_generation_w(w_example_todo)
        wo_results = self.run_concurrent_generation_wo(wo_example_todo)
        
        # print statistics
        self.logger.info(f"\n🎉 processing completed!")
        self.logger.info(f"total dataset count: {len(agent_query_data)}")
        self.logger.info(
            f"[W] with example generation successful: {self.processor.success_count_w_example} / {len(w_example_todo)} "
            f"(skipped existing files: {skip_stats['w_example_skipped']})"
        )
        self.logger.info(
            f"[WO] without example generation successful: {self.processor.success_count_wo_example} / {len(wo_example_todo)} "
            f"(skipped existing files: {skip_stats['wo_example_skipped']})"
        )
        self.logger.info(f"skipped datasets with missing metadata: {skip_stats['no_metadata_count']}")
        self.logger.info(f"with example data save path: {self.config.w_example_output_dir}")
        self.logger.info(f"without example data save path: {self.config.wo_example_output_dir}")
        
        # save detailed results
        results = {
            'w_example_results': w_results,
            'wo_example_results': wo_results,
            'skip_stats': skip_stats,
            'summary': {
                'total_datasets': len(agent_query_data),
                'w_example_success': self.processor.success_count_w_example,
                'wo_example_success': self.processor.success_count_wo_example,
                'w_example_todo_count': len(w_example_todo),
                'wo_example_todo_count': len(wo_example_todo)
            }
        }
        
        results_file = self.config.results_filename
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"detailed results saved to: {results_file}")
        
        return results


def update_metadata_with_generation_id(config: GenerationConfig) -> bool:
    """update metadata file, add search_id (synthesis type)"""
    logger = logging.getLogger(__name__)
    logger.info("starting to update metadata file, adding synthesis search_id...")
    
    # read original metadata file
    try:
        with open(config.metadata_file, 'r', encoding='utf-8') as f:
            metadata_records = json.load(f)
        logger.info(f"successfully loaded metadata file, containing {len(metadata_records)} records")
    except Exception as e:
        logger.error(f"failed to load metadata file: {e}")
        return False
    
    # get generated dataset ID list
    generated_dataset_ids = set()
    
    # check w_example directory
    if os.path.exists(config.w_example_output_dir):
        for filename in os.listdir(config.w_example_output_dir):
            if filename.endswith('.json'):
                dataset_id = filename[:-5].replace('$', '/')  # remove .json suffix and restore /
                generated_dataset_ids.add(dataset_id)
    
    # check wo_example directory
    if os.path.exists(config.wo_example_output_dir):
        for filename in os.listdir(config.wo_example_output_dir):
            if filename.endswith('.json'):
                dataset_id = filename[:-5].replace('$', '/')  # remove .json suffix and restore /
                generated_dataset_ids.add(dataset_id)
    
    logger.info(f"found {len(generated_dataset_ids)} generated datasets")
    
    # update metadata records
    updated_records = []
    for record in metadata_records:
        dataset_id = record.get('dataset_id')
        updated_record = record.copy()
        
        # if the dataset has been generated, add normalized search_id
        if dataset_id in generated_dataset_ids:
            # build normalized search_id: synthesis_{original_id}_{model}
            model_name = config.model  # keep original format like o3
            search_id = f"synthesis_{dataset_id}_{model_name}"
            updated_record['search_id'] = search_id
            logger.info(f"added search_id: {search_id} for dataset {dataset_id}")
        else:
            logger.info(f"no generated data found for dataset {dataset_id}")
        
        updated_records.append(updated_record)
    
    # save updated metadata file, using output path in configuration
    output_dir = config.results_output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # build output filename, using model name
    model_name = config.model.replace('/', '_').replace('-', '_')
    output_filename = f"generation_{model_name}_metadata.json"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_records, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ successfully saved updated metadata file to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"❌ failed to save updated metadata file: {e}")
        return False


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
            logging.FileHandler('generation_agent.log', encoding='utf-8')
        ],
        force=True
    )
    
    return logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="synthetic data generation tool, support configuration file and command line parameter call",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # use configuration file
  python generation_agent.py --config configs/generation_config.yaml
  
  # use configuration file and override some parameters
  python generation_agent.py --config configs/generation_config.yaml --num-data 100 --max-workers 10
  
  # use command line parameters only
  python generation_agent.py --agent-query-file data/queries.json --metadata-file data/metadata.json --num-data 50
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
        '--agent-query-file',
        type=str,
        help='agent query file path'
    )
    
    parser.add_argument(
        '--metadata-file',
        type=str,
        help='metadata file path'
    )
    
    # 输出路径参数
    parser.add_argument(
        '--w-example-output-dir',
        type=str,
        help='output directory with example data'
    )
    
    parser.add_argument(
        '--wo-example-output-dir',
        type=str,
        help='output directory without example data'
    )
    
    parser.add_argument(
        '--results-filename',
        type=str,
        default='generation_results.json',
        help='result filename (default: generation_results.json)'
    )
    
    # 生成参数
    parser.add_argument(
        '--num-data',
        type=int,
        help='number of data generated for each dataset'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        help='maximum number of concurrent threads'
    )
    
    parser.add_argument(
        '--max-json-retries',
        type=int,
        help='maximum number of JSON parsing retries'
    )
    
    parser.add_argument(
        '--max-llm-retries',
        type=int,
        help='maximum number of LLM API call retries'
    )
    
    # Azure OpenAI参数
    parser.add_argument(
        '--api-endpoint',
        type=str,
        help='Azure OpenAI API endpoint'
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Azure OpenAI API key'
    )
    
    parser.add_argument(
        '--api-version',
        type=str,
        help='Azure OpenAI API version'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='model name'
    )
    
    # 其他参数
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='skip existing valid files'
    )
    
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='do not skip existing files, force re-generation'
    )
    
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
    
    # set up logging
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
        
        # handle special logic for skip_existing
        if args.no_skip_existing:
            config_overrides['skip_existing'] = False
        elif args.skip_existing:
            config_overrides['skip_existing'] = True
        
        # create configuration object
        if args.config:
            logger.info(f"using configuration file: {args.config}")
            config = GenerationConfig.from_config_file(args.config, **config_overrides)
        else:
            # check required parameters
            required_params = ['agent_query_file', 'metadata_file']
            missing_params = [p for p in required_params if p not in config_overrides]
            if missing_params:
                logger.error(f"missing required parameters: {missing_params}")
                logger.error("please provide configuration file or complete command line parameters")
                sys.exit(1)
            
            logger.info("using command line parameters to create configuration")
            # set default values for missing parameters
            defaults = {
                'w_example_output_dir': './output/w_example',
                'wo_example_output_dir': './output/wo_example',
                'results_filename': 'generation_results.json',
                'results_output_dir': './datasets/results',
                'num_data': 50,
                'max_workers': 20,
                'max_json_retries': 3,
                'max_llm_retries': 3,
                'api_endpoint': 'https://guohe-apim.azure-api.net',
                'api_key': os.getenv('AZURE_OPENAI_API_KEY', ''),
                'api_version': '2025-01-01-preview',
                'model': 'o3',
                'min_file_size': 10,
                'skip_existing': True,
                'verbose': verbose,
                'system_prompt': 'You are a specialized expert in fine-tuning data synthesis. You excel at generating high-quality synthetic datasets for specific requirements.',
                'with_example_template': """You are a specialized expert in fine-tuning data synthesis. You have the following dataset search requirement: {agent_query}

Your task is to directly synthesize {num_data} corresponding examples based on this requirement. The goal is to create synthetic data that, when used for fine-tuning a large language model, will achieve better performance than fine-tuning on existing datasets found through the search.

Here is a reference example for guidance: {example_data}

You MUST output exactly {num_data} samples in JSON list format, where each sample contains only "input" and "output" fields, following this exact format:

[
  {{
    "input": "...",
    "output": "..."
  }},
  {{
    "input": "...",
    "output": "..."
  }},
  ...
]

Important requirements:
1. Generate EXACTLY {num_data} examples
2. Each example must have only "input" and "output" fields
3. Follow the task type and domain specified in the search requirement
4. Use the reference example to understand the expected format and style
5. Ensure diversity across your generated examples
6. Focus on creating high-quality data that will improve model performance through fine-tuning""",
                'without_example_template': """You are a specialized expert in fine-tuning data synthesis. You have the following dataset search requirement: {agent_query}

Your task is to directly synthesize {num_data} corresponding examples based on this requirement. The goal is to create synthetic data that, when used for fine-tuning a large language model, will achieve better performance than fine-tuning on existing datasets found through the search.

You MUST output exactly {num_data} samples in JSON list format, where each sample contains only "input" and "output" fields, following this exact format:

[
  {{
    "input": "...",
    "output": "..."
  }},
  {{
    "input": "...",
    "output": "..."
  }},
  ...
]

Important requirements:
1. Generate EXACTLY {num_data} examples
2. Each example must have only "input" and "output" fields
3. Follow the task type and domain specified in the search requirement
4. Ensure diversity across your generated examples
5. Focus on creating high-quality data that will improve model performance through fine-tuning
6. Analyze the search requirement carefully to understand the expected input/output format and content"""
            }
            
            # apply default values
            for key, default_value in defaults.items():
                if key not in config_overrides:
                    config_overrides[key] = default_value
            
            config = GenerationConfig(**config_overrides)
        
        # validate configuration
        if not config.api_key:
            logger.error("API key is not set. Please set it in the configuration file or provide it through the AZURE_OPENAI_API_KEY environment variable")
            sys.exit(1)
        
        # create and run generation agent
        agent = GenerationAgent(config)
        logger.info("🎯 starting data generation task")
        
        results = agent.run()
        
        logger.info("🎉 data generation task completed!")
        
        # update metadata file, add synthesis search_id
        logger.info("🔄 starting to update metadata file...")
        if update_metadata_with_generation_id(config):
            logger.info("✅ metadata file updated successfully")
        else:
            logger.warning("⚠️ metadata file update failed")
        
        # return exit code
        total_success = results['summary']['w_example_success'] + results['summary']['wo_example_success']
        total_attempted = results['summary']['w_example_todo_count'] + results['summary']['wo_example_todo_count']
        
        if total_success == total_attempted:
            sys.exit(0)  # all successful
        elif total_success > 0:
            sys.exit(1)  # partial success
        else:
            sys.exit(2)  # all failed
        
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