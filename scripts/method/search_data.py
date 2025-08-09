#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated search dataset pipeline
integrates four stages:
1. search_agent: search candidate datasets
2. check_exist_gated: check dataset availability
3. generate_search_train_set: generate training set
4. generate_search_set_metadata: generate metadata
"""
import os
import sys
sys.path.append(os.getcwd())
from scripts.utils.call_llm import CallLLM
from scripts.utils.config_loader import ConfigLoader
from argparse import Namespace
import sys
import json
import time
import os
import pandas as pd
from datetime import datetime, date
from openai import OpenAI, AzureOpenAI
from huggingface_hub import hf_hub_download, list_repo_files, dataset_info
from tqdm import tqdm
import concurrent.futures
import threading
from datasets import load_dataset, get_dataset_config_names
import random
import re
import argparse

# ========== global configuration ==========

# default model configuration
# concurrent configuration
# MAX_WORKERS = 20
# MAX_SEARCH_RETRIES = 5
# MAX_SAMPLES = 1000
# NUM_SEARCH = 5
# README_THRESHOLD = 100

# thread lock
lock = threading.Lock()
search_success_count = 0

# ========== general utility functions ==========

def log_message(message, prefix="[PIPELINE]"):
    """uniform logging output function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {prefix} {message}"
    print(log_entry)

def json_serialize_sample(obj):
    """custom JSON serialization function, handling datetime and other non-serializable objects"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: json_serialize_sample(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serialize_sample(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return str(obj)
    else:
        return obj

# ========== stage 1: search candidate datasets ==========

def generate_search_prompt(agent_query, num_search=5):
    """generate search dataset prompt"""
    return f"""{agent_query}

IMPORTANT: You must search for publicly accessible datasets from Hugging Face and return exactly {num_search} suitable dataset IDs.

SEARCH STRATEGY:
- First, search for datasets that closely match the query
- If you cannot find enough datasets, gradually relax the search criteria to find more potential datasets
- Include both popular and less popular datasets that might be relevant
- Consider datasets with similar tasks, domains, or data types

OUTPUT FORMAT REQUIREMENTS:
You MUST output your response in EXACTLY this JSON format - do not include any other text or explanations:

json:{{'search_datasets': ['dataset_id_1', 'dataset_id_2', 'dataset_id_3', 'dataset_id_4', 'dataset_id_5', 'dataset_id_6', 'dataset_id_7', 'dataset_id_8', 'dataset_id_9', 'dataset_id_10']}}

CRITICAL:
- Use ONLY the exact format above
- Replace dataset_id_1, dataset_id_2, etc. with actual Hugging Face dataset IDs
- Ensure you provide exactly {num_search} dataset IDs
- Do not include any text before or after the JSON
- The JSON must be valid and parseable"""

def extract_search_result(api_response):
    """extract dataset list from search API response"""
    try:
        # preprocess response, remove possible markdown format
        cleaned_response = api_response.strip()
        if '```json' in cleaned_response:
            start = cleaned_response.find('```json') + 7
            end = cleaned_response.find('```', start)
            if end != -1:
                cleaned_response = cleaned_response[start:end].strip()
        elif '```' in cleaned_response:
            start = cleaned_response.find('```') + 3
            end = cleaned_response.find('```', start)
            if end != -1:
                cleaned_response = cleaned_response[start:end].strip()
        
        # try to find json:{} format
        if 'json:' in cleaned_response:
            start = cleaned_response.find('json:') + 5
            start_brace = cleaned_response.find('{', start)
            if start_brace == -1:
                raise ValueError("No JSON object found after 'json:'")
            
            # find matching 
            brace_count = 0
            end_brace = start_brace
            for i in range(start_brace, len(cleaned_response)):
                if cleaned_response[i] == '{':
                    brace_count += 1
                elif cleaned_response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_brace = i
                        break
            
            json_str = cleaned_response[start_brace:end_brace+1]
        else:
            # if no json: format, try to find JSON object directly
            start = cleaned_response.find('{')
            end = cleaned_response.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = cleaned_response[start:end]
            else:
                raise ValueError("No JSON object found in response")
        
        # parse JSON
        result = json.loads(json_str)
        
        # verify necessary fields
        if 'search_datasets' in result and isinstance(result['search_datasets'], list):
            datasets = result['search_datasets']
            if len(datasets) > 0:
                valid_datasets = [d.strip() for d in datasets if d and d.strip()]
                if valid_datasets:
                    return valid_datasets
                else:
                    log_message("all dataset IDs are invalid")
                    return None
            else:
                log_message("search result list is empty")
                return None
        else:
            log_message(f"JSON missing search_datasets field or format error: {result}")
            return None
        
    except json.JSONDecodeError as e:
        log_message(f"JSON parsing failed: {e}")
        log_message(f"try to parse content: {api_response[:300]}...")
        return None
    except Exception as e:
        log_message(f"parse search result failed: {e}")
        log_message(f"response content: {api_response[:300]}...")
        return None

def search_single_dataset(item, llm_caller):
    """search single dataset function"""
    global search_success_count
    
    original_task_id = item['task_id']
    original_dataset_id = item['dataset_id']
    agent_query = item['query']
    thread_id = threading.current_thread().ident
    
    log_message(f"[thread {thread_id}] 🔍 start searching dataset: {original_task_id}")
    
    # generate search prompt
    search_prompt = generate_search_prompt(agent_query)
    messages = [
                {"role": "user", "content": search_prompt}
            ]
    # call search API
    search_response, prompt_tokens, completion_tokens = llm_caller.post_request(messages)
    if not search_response:
        log_message(f"[thread {thread_id}] ❌ search API call failed: {original_task_id}")
        return {
            "original_task_id": original_task_id,
            "original_dataset_id": original_dataset_id,
            "searched_datasets": []
        }
    
    # extract search results
    search_datasets = extract_search_result(search_response)
    if not search_datasets:
        log_message(f"[thread {thread_id}] ❌ search result parsing failed: {original_task_id}")
        log_message(f"[thread {thread_id}] API original response: {search_response[:500]}...")
        return {
            "original_task_id": original_task_id,
            "original_dataset_id": original_dataset_id,
            "searched_datasets": []
        }
    
    log_message(f"[thread {thread_id}] 📋 get {len(search_datasets)} candidate datasets: {search_datasets}")
    
    # clean search results, filter out empty strings
    cleaned_datasets = [dataset.strip() for dataset in search_datasets if dataset and dataset.strip()]
    
    with lock:
        search_success_count += 1
    
    log_message(f"[thread {thread_id}] ✅ search successful: {original_task_id} -> {len(cleaned_datasets)} results")
    
    return {
        "original_task_id": original_task_id,
        "original_dataset_id": original_dataset_id,
        "searched_datasets": cleaned_datasets
    }

def run_search_phase_parallel(agent_query_data, max_workers=20, llm_caller=None):
    """run search phase in parallel"""
    log_message(f"🚀 start parallel search phase, using {max_workers} threads to process {len(agent_query_data)} queries")
    
    search_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(search_single_dataset, item, llm_caller): item for item in agent_query_data}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_item), 
                          total=len(agent_query_data), 
                          desc="searching datasets"):
            try:
                result = future.result()
                search_results.append(result)
            except Exception as e:
                item = future_to_item[future]
                log_message(f"❌ searching dataset {item['task_id']} failed: {e}")
                search_results.append({
                    "original_task_id": item['task_id'],
                    "original_dataset_id": item['dataset_id'],
                    "searched_datasets": []
                })
    
    return search_results

def run_search_agent(args, llm_caller):
    """stage 1: search candidate datasets"""
    log_message("stage 1: start searching candidate datasets...")
    
    # load metadata data (contains query information)
    try:
        with open(args.metadata_file, 'r', encoding='utf-8') as f:
            metadata_records = json.load(f)
        log_message(f"successfully loaded metadata data, {len(metadata_records)} records")
    except Exception as e:
        log_message(f"load metadata data failed: {e}")
        return False
    
    # convert to agent_query_data format
    agent_query_data = []
    for record in metadata_records:
        query_item = {
            'task_id': record.get('task_id'),
            'dataset_id': record.get('dataset_id'),
            'query': record.get('query')
        }
        agent_query_data.append(query_item)
        
    log_message(f"convert to agent query format, {len(agent_query_data)} records")
    
    # run search phase
    search_results = run_search_phase_parallel(agent_query_data, args.max_workers, llm_caller)
    
    # save search results
    try:
        with open(args.search_results, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, ensure_ascii=False, indent=2)
        log_message(f"✅ search phase completed, results saved to: {args.search_results}")
    except Exception as e:
        log_message(f"❌ save search results failed: {e}")
        return False
    
    # print search statistics
    successful_searches = sum(1 for result in search_results if result.get('searched_datasets'))
    log_message(f"📊 search phase statistics: {successful_searches}/{len(search_results)} successful")
    
    return True

# ========== 阶段2: 检查数据集可用性 ==========

def check_dataset_gated_status(dataset_id):
    """check if dataset is gated"""
    if '/' not in dataset_id:
        log_message(f"dataset ID format error, missing '/': {dataset_id}")
        return None
    
    try:
        info = dataset_info(dataset_id)
        if hasattr(info, 'gated') and info.gated:
            return True
        return False
    except Exception as e:
        error_msg = str(e).lower()
        gated_keywords = ['gated', 'access', 'authentication', 'token', 'private', 'permission']
        if any(keyword in error_msg for keyword in gated_keywords):
            return True
        log_message(f"check dataset {dataset_id} gated status failed: {e}")
        return None

def check_dataset_exists(dataset_id):
    """check if dataset exists in HuggingFace and is accessible"""
    try:
        info = dataset_info(dataset_id)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        not_found_keywords = ['not found', '404', 'does not exist', 'repository not found']
        if any(keyword in error_msg for keyword in not_found_keywords):
            log_message(f"dataset {dataset_id} does not exist: {e}")
        else:
            log_message(f"check dataset {dataset_id} existence failed: {e}")
        return False

def process_single_search_result(item):
    """process single search result, find the first dataset that exists and is not gated"""
    original_task_id = item.get('original_task_id', '')
    original_dataset_id = item.get('original_dataset_id', '')
    searched_datasets = item.get('searched_datasets', [])
    
    log_message(f"🔍 start processing: {original_task_id}")
    log_message(f"📋 candidate datasets: {searched_datasets}")
    
    if not searched_datasets:
        log_message(f"⚠️ {original_task_id} has no candidate datasets")
        return {
            "original_task_id": original_task_id,
            "original_dataset_id": original_dataset_id,
            "selected_dataset": None
        }
    
    # check each dataset from front to back
    for i, dataset_name in enumerate(searched_datasets):
        dataset_name = dataset_name.strip()
        
        if not dataset_name:
            log_message(f"⚠️ the {i+1}th dataset name is empty, skip")
            continue
        
        log_message(f"🔍 check the {i+1}th dataset: {dataset_name}")
        
        # check 1: dataset existence
        if not check_dataset_exists(dataset_name):
            log_message(f"❌ dataset {dataset_name} does not exist or is not accessible, skip")
            continue
        
        log_message(f"✅ dataset {dataset_name} exists")
        
        # check 2: dataset gated status
        is_gated = check_dataset_gated_status(dataset_name)
        if is_gated is True:
            log_message(f"🔒 dataset {dataset_name} is gated, skip")
            continue
        elif is_gated is None:
            log_message(f"❓ cannot determine dataset {dataset_name} gated status, skip")
            continue
        
        log_message(f"🔓 dataset {dataset_name} is not gated")
        log_message(f"🎯 select dataset: {dataset_name}")
        
        return {
            "original_task_id": original_task_id,
            "original_dataset_id": original_dataset_id,
            "selected_dataset": dataset_name
        }
    
    # if no suitable dataset is found
    log_message(f"❌ {original_dataset_id} has no suitable dataset")
    return {
        "original_task_id": original_task_id,
        "original_dataset_id": original_dataset_id,
        "selected_dataset": None
    }

def run_check_exist_gated(args,llm_caller):
    """stage 2: check dataset availability"""
    log_message("stage 2: check dataset availability...")
    
    # load search results
    try:
        with open(args.search_results, 'r', encoding='utf-8') as f:
            search_results = json.load(f)
        log_message(f"successfully loaded search results file, {len(search_results)} records")
    except Exception as e:
        log_message(f"load search results file failed: {e}")
        return False
    
    # process each search result
    check_results = []
    
    log_message(f"🚀 start processing {len(search_results)} search results")
    
    for item in tqdm(search_results, desc="check dataset"):
        try:
            result = process_single_search_result(item)
            check_results.append(result)
            time.sleep(0.1)  # add a short delay to avoid too frequent requests
        except Exception as e:
            log_message(f"❌ processing {item.get('original_dataset_id', 'unknown')} failed: {e}")
            check_results.append({
                "original_task_id": item.get('original_task_id', 'unknown'),
                "original_dataset_id": item.get('original_dataset_id', 'unknown'),
                "selected_dataset": None
            })
    
    # save check results
    try:
        with open(args.check_result, 'w', encoding='utf-8') as f:
            json.dump(check_results, f, ensure_ascii=False, indent=2)
        log_message(f"✅ check results saved to: {args.check_result}")
    except Exception as e:
        log_message(f"❌ save check results failed: {e}")
        return False
    
    # print statistics
    total_count = len(check_results)
    successful_count = sum(1 for result in check_results if result.get('selected_dataset') is not None)
    failed_count = total_count - successful_count
    
    log_message(f"📊 check phase statistics:")
    log_message(f"   - total count: {total_count}")
    log_message(f"   - successful count: {successful_count}")
    log_message(f"   - failed count: {failed_count}")
    log_message(f"   - success rate: {successful_count/total_count*100:.1f}%")
    
    return True


# ========== stage 3: generate training set ==========

def get_readme_content(dataset_id):
    """get dataset readme content"""
    try:
        # method 1: try to download README.md file
        try:
            files = list_repo_files(repo_id=dataset_id, repo_type="dataset")
            if "README.md" in files:
                readme_path = hf_hub_download(
                    repo_id=dataset_id,
                    filename="README.md",
                    repo_type="dataset"
                )
                
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content and len(content) > 100:
                    return content
        except Exception as e:
            log_message(f"下载README.md失败: {e}")
            pass
        
        # method 2: try to get description from dataset info
        try:
            info = dataset_info(dataset_id)
            if hasattr(info, 'description') and info.description and len(info.description.strip()) > 50:
                return info.description.strip()
        except Exception as e:
            log_message(f"get dataset description failed: {e}")
            pass
            
        log_message(f"cannot get dataset {dataset_id} readme content, use empty string")
        return ""
    except Exception as e:
        log_message(f"get dataset {dataset_id} readme failed: {e}")
        return ""

def get_dataset_sample_with_config(dataset_id):
    """get a random sample from dataset, and return the selected config and split information"""
    try:
        # first try to get dataset config info
        configs = None
        selected_config = None
        dataset = None
        
        try:
            info = dataset_info(dataset_id)
            configs = list(info.config_names) if hasattr(info, 'config_names') and info.config_names else None
            if configs:
                log_message(f"get config via dataset_info: {configs}")
        except Exception as e:
            log_message(f"get dataset config info failed: {e}")
            configs = None
        
        # if dataset_info fails, try to load directly to detect if config is needed
        if not configs:
            try:
                log_message(f"try to load dataset {dataset_id} to detect config")
                dataset = load_dataset(dataset_id, streaming=True,trust_remote_code=True)
                log_message(f"load directly successful, dataset does not need config")
            except Exception as e:
                error_msg = str(e)
                log_message(f"load directly failed: {error_msg}")
                
                # check if error message contains config list
                if "Config name is missing" in error_msg and "available configs:" in error_msg:
                    try:
                        pattern = r"available configs:\s*\[(.*?)\]"
                        match = re.search(pattern, error_msg)
                        if match:
                            configs_str = match.group(1)
                            configs = [config.strip().strip("'\"") for config in configs_str.split(',')]
                            log_message(f"extract config from error message: {configs}")
                    except Exception as parse_e:
                        log_message(f"parse config list failed: {parse_e}")
                        return None, None, None
                else:
                    return None, None, None
        
        # if there is config, randomly select one
        if configs and len(configs) > 0:
            selected_config = random.choice(configs)
            log_message(f"dataset {dataset_id} has config: {configs}, randomly select: {selected_config}")
            
            # try to load config, first try streaming=True, then try streaming=False
            dataset = None
            for use_streaming in [True, False]:
                try:
                    log_message(f"try to load config {selected_config}, streaming={use_streaming}")
                    dataset = load_dataset(dataset_id, selected_config, streaming=use_streaming,trust_remote_code=True)
                    log_message(f"load config {selected_config} successfully, streaming={use_streaming}")
                    break
                except Exception as e:
                    log_message(f"load config {selected_config} failed (streaming={use_streaming}): {e}")
                    if not use_streaming:
                        return None, None, None
                    continue
        elif dataset is None:
            # if there is no config and the direct load also fails, try non-streaming mode
            log_message(f"try non-streaming mode to load dataset {dataset_id}")
            try:
                dataset = load_dataset(dataset_id, streaming=False,trust_remote_code=True)
                log_message(f"non-streaming mode load successfully")
            except Exception as e:
                log_message(f"non-streaming mode load failed: {e}")
                return None, None, None
        
        # get all available splits
        available_splits = list(dataset.keys())
        log_message(f"available dataset splits: {available_splits}")
        
        # first try train split, then test, validation, and finally other
        if 'train' in available_splits:
            selected_split = 'train'
        elif 'test' in available_splits:
            selected_split = 'test'
        elif 'validation' in available_splits:
            selected_split = 'validation'
        else:
            selected_split = available_splits[0]
        
        log_message(f"selected split: {selected_split}")
        
        # get a random sample
        split_dataset = dataset[selected_split]
        # use different sampling strategy based on whether it is a streaming dataset
        samples = []
        try:
            if hasattr(split_dataset, 'streaming') and split_dataset.streaming:
                # for streaming dataset, take a random sample from the first few samples
                log_message("use streaming mode to sample")
                for i, sample in enumerate(split_dataset):
                    samples.append(sample)
                    if i >= 20:  # take the first 20 samples
                        break
            else:
                # for non-streaming dataset, can randomly index access
                log_message("sample")
                try:
                    for i, sample in enumerate(split_dataset):
                        samples.append(sample)
                        if i >= 20:  # take the first 20 samples
                            break
                except Exception as e:
                    log_message(f"sample failed: {e}")
                    split_dataset = load_dataset(dataset_id, streaming=False,trust_remote_code=True)[selected_split]
                    for i, sample in enumerate(split_dataset):
                        samples.append(sample)
                        if i >= 20:  # take the first 20 samples
                            break
        except Exception as e:
            log_message(f"sample failed: {e}")
            # if streaming mode fails, try simple sequential sampling
            try:
                log_message("try simple sequential sampling")
                count = 0
                for sample in split_dataset:
                    samples.append(sample)
                    count += 1
                    if count >= 5:  # take the first 5 samples
                        break
            except Exception as e2:
                log_message(f"simple sequential sampling failed: {e2}")
                return None, None, None
        
        if samples:
            random_sample = random.choice(samples)
            log_message(f"successfully get random sample, contains fields: {list(random_sample.keys())}")
            return random_sample, selected_config, selected_split
        else:
            log_message("cannot get any sample")
            return None, None, None
            
    except Exception as e:
        log_message(f"get random sample from dataset {dataset_id} failed: {e}")
        return None, None, None

def generate_analysis_prompt(readme_content, sample_data, config_name, split_name):
    """generate analysis prompt, require LLM to analyze how to convert dataset to fine-tuning format"""
    config_section = f"Selected config/subset: {config_name}" if config_name else "Selected config/subset: None (single config dataset)"
    split_section = f"Selected split: {split_name}"
    
    # handle readme_content is empty
    readme_section = f"Dataset README content:\n{readme_content}" if readme_content.strip() else "Dataset README content:\n(No README content available)"
    
    try:
        serialized_data = json_serialize_sample(sample_data)
        sample_section = f"""

Sample data from the dataset:
{json.dumps(serialized_data, ensure_ascii=False, indent=2)}"""
    except Exception as e:
        log_message(f"sample data serialization failed: {e}")
        sample_section = f"""

Sample data from the dataset:
{str(sample_data)}"""
    
    prompt = f"""You are tasked with analyzing a dataset and determining how to convert it into a fine-tuning format for language models. The fine-tuning data should have exactly two fields: "input" and "output".

{readme_section}

{config_section}
{split_section}{sample_section}

Based on the README content (if available) and the sample data, please analyze and provide the following information:

1. **Selected Config**: Which config/subset was selected from the dataset (output "None" if there's only one config)
2. **Selected Split**: Which split was selected (train/test/validation/etc.)
3. **Conversion Rules**: Describe the rules for converting the dataset samples into input/output format for fine-tuning, including:
   - Which columns/fields should be combined for the "input"
   - Which columns/fields should be used for the "output"  
   - **MUST include appropriate instruction text** (e.g., "Please answer the following question:", "Classify the sentiment:", "Translate the text:", etc.)
   - The exact format and order for combining the fields
   - Any preprocessing needed for the data

**IMPORTANT REQUIREMENTS:**
- The "instruction_template" field MUST contain a clear, specific instruction appropriate for the task
- The instruction should tell the model exactly what to do (e.g., answer questions, classify, translate, summarize, etc.)
- Do NOT use "null" for instruction_template - always provide a meaningful instruction
- The input should clearly guide the model on what output is expected

Please provide your response in the following JSON format:

{{
    "selected_config": "config_name or None",
    "selected_split": "split_name",
    "conversion_rules": {{
        "input_components": ["list of field names to include in input (use exact field names from sample)"],
        "output_components": ["list of field names to include in output (use exact field names from sample)"],
        "instruction_template": "REQUIRED: Clear instruction text telling the model what to do",
        "input_format": "detailed description of how to format the input",
        "output_format": "detailed description of how to format the output",
        "example_conversion": {{
            "input": "example input with instruction based on the sample",
            "output": "example output based on the sample"
        }}
    }}
}}

Important: Make sure the conversion makes sense for language model fine-tuning and follows common patterns for instruction-following datasets. The instruction_template is MANDATORY and should be task-specific."""

    return prompt

def parse_conversion_rules(api_response):
    """parse conversion rules from API response"""
    try:
        # try to extract JSON part
        if '```json' in api_response:
            start = api_response.find('```json') + 7
            end = api_response.find('```', start)
            json_content = api_response[start:end].strip()
        elif '{' in api_response and '}' in api_response:
            start = api_response.find('{')
            end = api_response.rfind('}') + 1
            json_content = api_response[start:end]
        else:
            raise ValueError("No JSON found in response")
        
        rules = json.loads(json_content)
        return rules
    except Exception as e:
        log_message(f"parse conversion rules failed: {e}")
        return None

def extract_field_name(component_str):
    """extract actual field name from component string"""
    component_str = component_str.strip()
    
    if '(' in component_str:
        field_name = component_str.split('(')[0].strip()
    else:
        field_name = component_str
    
    field_name = field_name.strip('.,;:!?')
    
    return field_name

def extract_field_value_with_parsing(sample, component_str):
    """smartly extract field value, support complex field parsing"""
    component_str = component_str.strip()
    
    # check if there is special parsing instruction
    if '(' in component_str and ')' in component_str:
        field_part = component_str.split('(')[0].strip()
        instruction_part = component_str[component_str.find('(')+1:component_str.rfind(')')].strip()
        
        # handle special parsing instruction
        if 'derived from' in instruction_part and 'after' in instruction_part:
            source_field = None
            delimiter = None
            
            if 'from text' in instruction_part:
                source_field = 'text'
            if "after '*'" in instruction_part:
                delimiter = '*'
            elif 'after ":"' in instruction_part:
                delimiter = ':'
            elif 'after "|"' in instruction_part:
                delimiter = '|'
            
            if source_field and delimiter and source_field in sample:
                source_value = str(sample[source_field])
                if delimiter in source_value:
                    extracted = source_value.split(delimiter)[-1].strip()
                    return extracted
        
        elif 'derived from' in instruction_part and 'before' in instruction_part:
            source_field = None
            delimiter = None
            
            if 'from text' in instruction_part:
                source_field = 'text'
            if "before '*'" in instruction_part:
                delimiter = '*'
            elif 'before ":"' in instruction_part:
                delimiter = ':'
            
            if source_field and delimiter and source_field in sample:
                source_value = str(sample[source_field])
                if delimiter in source_value:
                    extracted = source_value.split(delimiter)[0].strip()
                    return extracted
    
    # if there is no special instruction, directly extract field name
    field_name = extract_field_name(component_str)
    if field_name in sample:
        return str(sample[field_name]).strip()
    
    return None

def smart_instruction_design(sample, rules):
    """smartly design instruction template"""
    conversion_rules = rules.get('conversion_rules', {})
    instruction_template = conversion_rules.get('instruction_template', '')
    
    # if there is valid instruction template, directly use it
    if instruction_template and instruction_template.strip() and instruction_template.lower() not in ["null", "none", ""]:
        return instruction_template.strip()
    
    # smartly generate instruction based on dataset features
    sample_keys = list(sample.keys())
    sample_values = [str(v) for v in sample.values()]
    
    # check if it is a classification task
    if any('*' in str(v) for v in sample.values()):
        return "Please classify the following content and provide the corresponding label:"
    
    # check if it is a question-answering task
    if any(key.lower() in ['question', 'query', 'q'] for key in sample_keys):
        return "Please answer the following question:"
    
    # check if it is a translation task
    if any(key.lower() in ['source', 'target', 'en', 'zh', 'english', 'chinese'] for key in sample_keys):
        return "Please translate the following content:"
    
    # check if it is a summarization task
    if any(key.lower() in ['text', 'article', 'document'] and len(str(sample.get(key, ''))) > 200 for key in sample_keys):
        return "Please generate a summary for the following content:"
    
    # default instruction
    return "Please complete the following task based on the given content:"

def apply_conversion_rules(sample, rules):
    """apply conversion rules to convert single sample to input/output format"""
    try:
        conversion_rules = rules.get('conversion_rules', {})
        input_components = conversion_rules.get('input_components', [])
        output_components = conversion_rules.get('output_components', [])
        input_format = conversion_rules.get('input_format', '')
        
        # smartly design instruction
        instruction = smart_instruction_design(sample, rules)
        
        # build input
        input_parts = [instruction]
        input_field_added = False
        
        for component in input_components:
            if isinstance(component, str):
                field_value = extract_field_value_with_parsing(sample, component)
                if field_value:
                    if '*' in field_value:
                        clean_value = field_value.split('*')[0].strip()
                        input_parts.append(clean_value)
                        input_field_added = True
                    else:
                        input_parts.append(field_value)
                        input_field_added = True
        
        # if there is no valid input field, use fallback logic
        if not input_field_added:
            for key, value in sample.items():
                if key.lower() not in ['label', 'answer', 'target', 'output', 'response', 'completion']:
                    str_value = str(value).strip()
                    if str_value:
                        if '*' in str_value:
                            clean_value = str_value.split('*')[0].strip()
                            input_parts.append(clean_value)
                        else:
                            input_parts.append(str_value)
                        input_field_added = True
                        break
        
        # build output
        output_parts = []
        output_field_added = False
        for component in output_components:
            if isinstance(component, str):
                field_value = extract_field_value_with_parsing(sample, component)
                if field_value:
                    output_parts.append(field_value)
                    output_field_added = True
        
        # if there is no valid output field, use fallback logic
        if not output_field_added:
            # first try to extract from fields with delimiter
            for key, value in sample.items():
                str_value = str(value).strip()
                if '*' in str_value:
                    parts = str_value.split('*')
                    if len(parts) > 1:
                        label = parts[-1].strip()
                        if label:
                            output_parts.append(label)
                            output_field_added = True
                            break
            
            # if still not found, try common output fields
            if not output_field_added:
                output_keys = ['label', 'answer', 'target', 'output', 'response', 'completion']
                for key in output_keys:
                    if key in sample:
                        str_value = str(sample[key]).strip()
                        if str_value:
                            output_parts.append(str_value)
                            output_field_added = True
                            break
        
        # make sure input and output are not empty
        if not input_field_added:
            all_content = []
            for key, value in sample.items():
                str_value = str(value).strip()
                if str_value:
                    if '*' in str_value:
                        clean_value = str_value.split('*')[0].strip()
                        all_content.append(clean_value)
                    else:
                        all_content.append(str_value)
            if all_content:
                input_parts.extend(all_content)
                input_field_added = True
        
        if not output_field_added:
            for key, value in sample.items():
                str_value = str(value).strip()
                if str_value.isdigit():
                    output_parts.append(str_value)
                    output_field_added = True
                    break
                elif len(str_value) < 20 and not any(char in str_value for char in ['。', '.', '?', '？']):
                    output_parts.append(str_value)
                    output_field_added = True
                    break
        
        # final assembly
        if 'two newline characters' in input_format.lower() or 'newline' in input_format.lower():
            final_input = '\n\n'.join(input_parts).strip()
        else:
            final_input = '\n'.join(input_parts).strip()
        
        final_output = '\n'.join(output_parts).strip()
        
        # make sure both are not empty
        if not final_input.strip() or not final_output.strip():
            return None
        
        return {
            "system":instruction,
            "input": final_input,
            "output": final_output
        }
    except Exception as e:
        return None

def convert_entire_dataset(dataset_id, config_name, split_name, rules, max_samples=1000):
    """convert entire dataset to fine-tuning format, limit max samples"""
    try:
        log_message(f"start converting entire dataset {dataset_id}")
        # load dataset
        if config_name and config_name != "None":
            log_message(f"use config {config_name} to load dataset")
            dataset = load_dataset(dataset_id, config_name, streaming=True,trust_remote_code=True)
        else:
            log_message(f"load dataset directly (no config)")
            dataset = load_dataset(dataset_id, streaming=True,trust_remote_code=True)

        # get specified split
        split_data = dataset[split_name]

        # use counter to get samples
        log_message(f"start getting samples...")
        samples = []
        try:
            for i, example in enumerate(split_data):
                converted = apply_conversion_rules(example, rules)
                if converted:
                    samples.append(converted)
                else:
                    log_message(f"conversion failed, skip sample {i}")
                    continue
                samples.append(example)
                if i >= max_samples - 1:  # get max_samples samples
                    break
                
                # optional: show progress
                if (i + 1) % 100 == 0:
                    log_message(f"get {i + 1} samples")

            actual_samples = len(samples)
            log_message(f"actually get {actual_samples} samples")
        except Exception as e:
            log_message(f"get samples data failed, use non-streaming mode to get")
            dataset = load_dataset(dataset_id, streaming=False,trust_remote_code=True)
            split_data = dataset[split_name]
            for i, example in enumerate(split_data):
                converted = apply_conversion_rules(example, rules)
                if converted:
                    samples.append(converted)
            return samples, len(samples)
        
        # # load dataset
        # if config_name and config_name != "None":
        #     log_message(f"use config {config_name} to load dataset")
        #     dataset = load_dataset(dataset_id, config_name)
        # else:
        #     log_message(f"load dataset directly (no config)")
        #     dataset = load_dataset(dataset_id,streaming=True)
        
        # # get specified split
        # split_data = dataset[split_name]
        
        # # determine actual number of samples to process
        # total_samples = len(split_data)
        # actual_samples = min(total_samples, max_samples)
        # log_message(f"dataset has {total_samples} samples, will process {actual_samples} samples")
        
        # converted_samples = []
        
        # # convert samples (only take the first MAX_SAMPLES samples)
        # for i in tqdm(range(actual_samples), desc="convert samples"):
        #     sample = split_data[i]
        #     converted = apply_conversion_rules(sample, rules)
        #     if converted:
        #         converted_samples.append(converted)
            
        #     # record progress every 1000 samples
        #     if (i + 1) % 1000 == 0:
        #         log_message(f"converted {i + 1}/{actual_samples} samples")
        
        # log_message(f"successfully converted {len(converted_samples)}/{actual_samples} samples")
        return samples, actual_samples
        
    except Exception as e:
        log_message(f"convert dataset failed: {e}")
        return None, 0

def save_converted_dataset(output_dir, original_dataset_id, search_dataset_id, converted_data):
    """save converted dataset, use search_dataset_id as filename"""
    try:
        # directly use search_dataset_id as filename, no replacement
        filename = f"{search_dataset_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        log_message(f"successfully save converted dataset to {filepath}")
        return filepath
        
    except Exception as e:
        log_message(f"save dataset failed: {e}")
        return None

def run_generate_search_train_set(args, llm_caller):
    """phase 3: generate training set"""
    log_message("phase 3: generate training set...")
    
    # build model sub-directory path
    model_name = args.search_model_name.replace('/', '_').replace('-', '_')
    output_dir = os.path.join(args.train_set_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    log_message(f"output directory: {output_dir}")
    
    # initialize conversion_log.txt file
    txt_log_path = os.path.join(output_dir, 'conversion_log.txt')
    with open(txt_log_path, 'w', encoding='utf-8') as f:
        f.write("search dataset conversion log\n")
        f.write("=" * 50 + "\n")
    
    # read input JSON file
    try:
        with open(args.check_result, 'r', encoding='utf-8') as f:
            dataset_pairs = json.load(f)
    except Exception as e:
        log_message(f"load check result file failed: {e}")
        return False
    
    # filter out valid dataset pairs (selected_dataset is not null)
    valid_pairs = []
    for pair in dataset_pairs:
        if pair.get('selected_dataset') is not None:
            valid_pairs.append(pair)
    
    log_message(f"find {len(valid_pairs)} valid dataset pairs to process")
    
    # add global variable to record dataset information
    dataset_info_list = []
    
    # process each dataset pair
    success_count = 0
    for i, pair in enumerate(valid_pairs):
        original_task_id = pair['original_task_id']
        original_dataset_id = pair['original_dataset_id']
        selected_dataset = pair['selected_dataset']
        
        # generate search_dataset_id: search_{selected_dataset}_{model}
        model_name_clean = args.search_model_name.replace('/', '_').replace('-', '_')
        # if there is no selected_dataset, use original_dataset_id
        dataset_name = selected_dataset if selected_dataset else original_dataset_id.replace('/', '_')
        search_dataset_id = f"search_{dataset_name.replace('/', '_')}_{model_name_clean}"
        
        # first check if selected_dataset has been downloaded
        if os.path.exists(os.path.join(output_dir, f'{search_dataset_id.replace("/", "_")}.json')):
            log_message(f"dataset {search_dataset_id} has been downloaded, skip")
            log_message(f"\n{'='*80}")
            # read from csv, but find search_dataset_id but task_id is different
            df = pd.read_csv(os.path.join(output_dir, f'search_dataset_info_o3-mini.csv'), encoding='utf-8')
            # find all records with the same search_dataset_id
            matching_rows = df[df['search_dataset_id'] == search_dataset_id]
            
            if len(matching_rows) > 0:
                # check if there are different task_ids
                existing_task_ids = set(matching_rows['original_task_id'].values)
                
                # if current task_id is not in the existing records, add new record
                if original_task_id not in existing_task_ids:
                    # use the configuration information of the first matching record
                    first_row = matching_rows.iloc[0]
                    config_name = first_row['config']
                    split_name = first_row['split']
                    samples_count = first_row['samples_count']
                    
                    dataset_info_list.append({
                        'original_task_id': original_task_id,
                        'original_dataset_id': original_dataset_id,
                        'search_dataset_id': search_dataset_id,
                        'config': config_name if config_name else 'None',
                        'split': split_name if split_name else 'N/A',
                        'status': 'Success',
                        'samples_count': samples_count,
                        'saved_path': os.path.join(output_dir, f'{search_dataset_id.replace("/", "_")}.json')
                    })
                    log_message(f"find records with the same search_dataset_id but different task_id: {search_dataset_id}, task_id: {original_task_id}")
                else:
                    log_message(f"skip duplicate record: {search_dataset_id}, task_id: {original_task_id}")
            else:
                # if no corresponding record is found in CSV, use default value
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': 'N/A',
                    'split': 'N/A',
                    'status': 'Success',
                    'samples_count': 0,
                    'saved_path': os.path.join(output_dir, f'{search_dataset_id.replace("/", "_")}.json')
                })
            continue
        
        log_message(f"\n{'='*80}")
        log_message(f"start processing {i+1}/{len(valid_pairs)} dataset pairs")
        log_message(f"original dataset: {original_dataset_id}")
        log_message(f"search dataset: {search_dataset_id}")
        log_message(f"{'='*80}")
        
        try:
            # 1. get README content (can be empty)
            readme_content = get_readme_content(search_dataset_id)
            log_message(f"get README content, length: {len(readme_content)} characters")
            
            # 2. get random sample and configuration information
            sample_data, config_name, split_name = get_dataset_sample_with_config(search_dataset_id)
            if not sample_data:
                log_message(f"cannot get samples from dataset {search_dataset_id}, skip")
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': 'N/A',
                    'split': 'N/A',
                    'status': 'Failed - No Sample',
                    'samples_count': 0
                })
                continue
            
            # 3. generate analysis prompt and call LLM
            prompt = generate_analysis_prompt(readme_content, sample_data, config_name, split_name)
            messages = [
                {"role": "system", "content": "You are an expert in dataset analysis and language model fine-tuning. Analyze datasets and provide clear, structured conversion rules."},
                {"role": "user", "content": prompt}
            ]
            api_response, prompt_tokens, completion_tokens = llm_caller.post_request(messages)
            if not api_response:
                log_message(f"API call failed, skip dataset {search_dataset_id}")
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': config_name if config_name else 'None',
                    'split': split_name if split_name else 'N/A',
                    'status': 'Failed - API Error',
                    'samples_count': 0
                })
                continue
            
            # 4. parse conversion rules
            rules = parse_conversion_rules(api_response)
            if not rules:
                log_message(f"parse conversion rules failed, skip dataset {search_dataset_id}")
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': config_name if config_name else 'None',
                    'split': split_name if split_name else 'N/A',
                    'status': 'Failed - Parse Rules',
                    'samples_count': 0
                })
                continue
            
            log_message(f"successfully parse conversion rules")
            
            # 5. test conversion of a sample first
            test_converted = apply_conversion_rules(sample_data, rules)
            if not test_converted:
                log_message(f"test sample conversion failed, skip dataset {search_dataset_id}")
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': config_name if config_name else 'None',
                    'split': split_name if split_name else 'N/A',
                    'status': 'Failed - Test Conversion',
                    'samples_count': 0
                })
                continue
            
            log_message(f"test conversion success - Input: {test_converted['input'][:100]}...")
            log_message(f"test conversion success - Output: {test_converted['output'][:100]}...")
            
            # 6. convert entire dataset (limit max samples)
            converted_data, actual_samples = convert_entire_dataset(search_dataset_id, config_name, split_name, rules)
            if not converted_data or len(converted_data) == 0:
                log_message(f"convert dataset failed or no valid samples, skip dataset {search_dataset_id}")
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': config_name if config_name else 'None',
                    'split': split_name if split_name else 'N/A',
                    'status': 'Failed - No Valid Samples',
                    'samples_count': 0
                })
                continue
            
            # 7. save converted dataset
            saved_path = save_converted_dataset(output_dir, original_dataset_id, search_dataset_id, converted_data)
            if saved_path:
                success_count += 1
                log_message(f"successfully process dataset pair, save to {saved_path}")
                log_message(f"converted samples: {len(converted_data)}")
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': config_name if config_name else 'None',
                    'split': split_name,
                    'status': 'Success',
                    'samples_count': len(converted_data),
                    'saved_path': saved_path
                })
            else:
                dataset_info_list.append({
                    'original_task_id': original_task_id,
                    'original_dataset_id': original_dataset_id,
                    'search_dataset_id': search_dataset_id,
                    'config': config_name if config_name else 'None',
                    'split': split_name if split_name else 'N/A',
                    'status': 'Failed - Save Error',
                    'samples_count': 0
                })
            
        except Exception as e:
            log_message(f"error when processing dataset pair: {e}")
            import traceback
            log_message(f"detailed error information: {traceback.format_exc()}")
            dataset_info_list.append({
                'original_task_id': original_task_id,
                'original_dataset_id': original_dataset_id,
                'search_dataset_id': search_dataset_id,
                'config': 'N/A',
                'split': 'N/A',
                'status': f'Failed - {str(e)[:50]}',
                'samples_count': 0
            })
            continue
    
    # save dataset information to CSV file - incremental write mode
    try:
        csv_file = os.path.join(output_dir, f'search_dataset_info_o3-mini.csv')
        
        # check existing file and read existing data
        existing_data = []
        if os.path.exists(csv_file):
            try:
                existing_df = pd.read_csv(csv_file, encoding='utf-8')
                existing_data = existing_df.to_dict('records')
                log_message(f"find existing CSV file, contains {len(existing_data)} records")
            except Exception as e:
                log_message(f"read existing CSV file failed: {e}")
                existing_data = []
        
        # merge existing data and new data
        all_data = existing_data + dataset_info_list
        
        # deduplicate (based on the combination of original_dataset_id and search_dataset_id)
        seen_keys = set()
        unique_data = []
        for item in all_data:
            key = (item.get('original_dataset_id', ''), item.get('search_dataset_id', ''))
            if key not in seen_keys:
                seen_keys.add(key)
                unique_data.append(item)
        
        # save merged data
        df = pd.DataFrame(unique_data)
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        new_count = len(dataset_info_list)
        total_count = len(unique_data)
        log_message(f"successfully save dataset information to {csv_file}")
        log_message(f"add {new_count} records, total {total_count} records")
        
    except Exception as e:
        log_message(f"save CSV file failed: {e}")
    
    log_message(f"\n{'='*80}")
    log_message(f"process completed! successfully process {success_count}/{len(valid_pairs)} dataset pairs")
    log_message(f"converted dataset saved in {output_dir} directory")
    log_message(f"dataset information recorded in {csv_file}")
    log_message(f"detailed log saved in {os.path.join(output_dir, 'conversion_log.txt')}")
    log_message(f"{'='*80}")
    
    return True

# ========== phase 4: generate metadata ==========

def run_generate_search_set_metadata(args, llm_caller: CallLLM):
    """phase 4: generate metadata"""
    log_message("phase 4: generate metadata...")
    
    # read CSV file
    original_data_path = args.metadata_file  # directly read from metadata_file
    original_data = json.load(open(original_data_path, 'r', encoding='utf-8'))
    csv_file_path = os.path.join(args.train_set_dir, f'search_dataset_info_o3-mini.csv')
    json_dir_path = args.train_set_dir
    output_file = args.metadata_output
    
    try:
        df = pd.read_csv(csv_file_path)
        log_message(f"successfully read CSV file, contains {len(df)} records")
    except Exception as e:
        log_message(f"read CSV file failed: {e}")
        return False
    
    # load existing results
    existing_results = []
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
            log_message(f"find existing results file, contains {len(existing_results)} processed datasets")
        except Exception as e:
            log_message(f"read existing results file failed: {e}")
            existing_results = []
    
    # get processed dataset ID combinations
    processed_datasets = set()
    for result in existing_results:
        key = (result['original_dataset_id'], result['search_dataset_id'])
        processed_datasets.add(key)
    
    log_message(f"processed dataset count: {len(processed_datasets)}")
    
    # store results (start from existing results)
    results = existing_results.copy()
    
    # count datasets to process
    total_datasets = len(df)
    remaining_datasets = []
    
    for idx, row in df.iterrows():
        original_task_id = row['original_task_id']
        original_dataset_id = row['original_dataset_id'].replace('/', '_')
        search_dataset_id = row['search_dataset_id'].replace('/', '_')
        key = (original_dataset_id, search_dataset_id)
        
        if key not in processed_datasets:
            remaining_datasets.append((idx, row))
    
    log_message(f"remaining datasets to process: {len(remaining_datasets)}")
    
    if len(remaining_datasets) == 0:
        log_message("✅ all datasets have been processed!")
        return True
    
    log_message(f"start processing remaining {len(remaining_datasets)} datasets...")
    
    # process remaining datasets
    for idx, (original_idx, row) in enumerate(tqdm(remaining_datasets, desc="process remaining datasets")):
        original_task_id = row['original_task_id']
        original_dataset_id = row['original_dataset_id']
        search_dataset_id = row['search_dataset_id']
        config = row['config']
        split = row['split']
        samples_count = row['samples_count']
        
        log_message(f"\nprocess {idx+1}/{len(remaining_datasets)} remaining dataset: {search_dataset_id}")
        log_message(f"(total progress: {len(results)}/{total_datasets})")
        
        # 1. get README content
        readme_content = get_readme_content(search_dataset_id)
        if readme_content:
            log_message(f"successfully get README content, length: {len(readme_content)} characters")
        else:
            log_message(f"cannot get README content")
        
        # 2. construct JSON file path and load sample data
        original_filename = original_dataset_id.replace('/', '_')
        search_filename = search_dataset_id.replace('/', '_')
        json_filename = f"{search_filename}.json"
        json_file_path = os.path.join(json_dir_path, json_filename)
        
        sample_data = None
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    first_sample = data[0]
                    if 'input' in first_sample and 'output' in first_sample:
                        sample_data = first_sample
                if sample_data:
                    log_message(f"successfully load sample data")
                else:
                    log_message(f"JSON file exists but cannot load sample data")
            except Exception as e:
                log_message(f"load JSON file failed: {e}")
        else:
            log_message(f"JSON file does not exist: {json_filename}")
        
        # 3. generate metadata prompt
        def generate_metadata_prompt(readme_content, sample_data, samples_count):
            if not readme_content and not sample_data:
                return f"""You need to analyze a dataset, but both README content and sample data are missing. Since no information is available about the dataset content, please output the following in English stating that metadata cannot be generated due to lack of both sample and README information.

Please output in JSON format:
{{
    "introduction": "Unable to generate metadata due to lack of both README and sample data",
    "task_type": "unknown", 
    "question": "unknown",
    "input": "unknown",
    "output": "unknown", 
    "source": "unknown",
    "example": "No sample available",
    "samples_count": {samples_count}
}}"""
            
            sample_section = ""
            if sample_data:
                sample_section = f"""

Sample from the dataset:
Input: {sample_data['input']}
Output: {sample_data['output']}"""
            
            readme_section = f"README content:\n{readme_content}" if readme_content else "README content: No README available"
            
            prompt = f"""You will see a README file introduction from a HuggingFace dataset and one sample from it. You need to output the following content based on these materials. Please output in JSON format.

{readme_section}{sample_section}

Please analyze the dataset based on both the README and the sample (prioritize the sample when there are conflicts, as the README might be vague while we've ensured all samples in the dataset are similar to the provided one), and output the following metadata in JSON format:

{{
    "introduction": "A one-sentence introduction of the dataset content, concise and clear, including key information about task, domain, input and output",
    "task_type": "Directly output a task type: include but not limited to multiple-choice, question-answering, summarization, text-classification, text-generation, translation",
    "question": "Directly output Question Content Type - Describes the primary knowledge domains and content types covered by the questions in the test dataset, such as open-domain knowledge of film and entertainment, scientific common sense, history and geography, literature and arts, sports news, and professional technical fields.",
    "input": "Directly output the dataset's input content, including its language, such as: an English news text for translation, a multiple-choice question in French philosophy domain, etc. (Consider both sample and README, don't be limited by single sample's domain, but also don't be too broad like README)",
    "output": "Directly output the dataset's output content, including its language, such as: a number 0 or 1, a letter A/B/C/D, translated Italian text, etc.",
    "source": "Directly output the dataset's source: real-world, human-generated, machine-generated, etc.",
    "example": "Directly extract the sample provided in the prompt and put it here",
    "samples_count": {samples_count}
}}

Important: Please strictly follow the above JSON format and provide a comprehensive analysis based on both README and sample data."""
            
            return prompt
        
        metadata_prompt = generate_metadata_prompt(readme_content, sample_data, samples_count)
        messages = [
            {"role": "system", "content": "You are an expert in dataset analysis and language model fine-tuning. Analyze datasets and provide clear, structured conversion rules."},
            {"role": "user", "content": metadata_prompt}
        ]
        search_metadata, prompt_tokens, completion_tokens = llm_caller.post_request(messages)
        
        if not search_metadata:
            log_message(f"API call failed, skip dataset {search_dataset_id}")
            continue
        
        # parse metadata to JSON format
        try:
            # try to extract JSON part
            if '```json' in search_metadata:
                start = search_metadata.find('```json') + 7
                end = search_metadata.find('```', start)
                json_content = search_metadata[start:end].strip()
            elif '{' in search_metadata and '}' in search_metadata:
                start = search_metadata.find('{')
                end = search_metadata.rfind('}') + 1
                json_content = search_metadata[start:end]
            else:
                log_message(f"cannot extract JSON format from API response, skip dataset {search_dataset_id}")
                json_content = search_metadata
            # parse JSON
            parsed_metadata = json.loads(json_content)
            log_message(f"successfully parse metadata to JSON format")
        except json.JSONDecodeError as e:
            log_message(f"JSON parse failed: {e}")
            log_message(f"try to parse content: {search_metadata[:300]}...")
            continue
        except Exception as e:
            log_message(f"parse metadata failed: {e}")
            continue
        # search for original_task_id in the dictionary list
        origin_metadata = None
        for item in original_data:
            if item['task_id'] == original_task_id:
                origin_metadata = item['metadata']
                break
        # 4. store results
        result = {
            "task_id": original_task_id,
            "search_dataset_id": search_filename,
            "original_dataset_id": original_dataset_id,
            "search_metadata": search_metadata,
            "original_metadata": origin_metadata
        }
        
        results.append(result)
        
        # 5. Save results
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            log_message(f"✅ successfully process and save dataset {search_dataset_id}")
        except Exception as e:
            log_message(f"save metadata failed: {e}")
    
    log_message(f"\n🎉 process completed!")
    log_message(f"successfully process {len(results)} datasets")
    log_message(f"results saved to {output_file}")
    
    # print statistics
    log_message(f"\n📊 statistics:")
    log_message(f"total dataset count: {total_datasets}")
    log_message(f"successfully process count: {len(results)}")
    log_message(f"new processed count: {len(results) - len(existing_results)}")
    log_message(f"skipped processed count: {len(processed_datasets)}")
    
    return True

def update_metadata_with_search_id(args, llm_caller):
    """update metadata file, add search_id"""
    log_message("phase 5: update metadata file...")
    
    # read original metadata file
    try:
        with open(args.metadata_file, 'r', encoding='utf-8') as f:
            metadata_records = json.load(f)
        log_message(f"successfully load metadata file, contains {len(metadata_records)} records")
    except Exception as e:
        log_message(f"load metadata file failed: {e}")
        return False
    
    # read search metadata results
    try:
        with open(args.metadata_output, 'r', encoding='utf-8') as f:
            search_metadata_results = json.load(f)
        log_message(f"successfully load search metadata results, contains {len(search_metadata_results)} records")
    except Exception as e:
        log_message(f"load search metadata results failed: {e}")
        return False
    
    # create task_id to search_dataset_id mapping
    task_to_search_id = {}
    for result in search_metadata_results:
        task_id = result.get('task_id')
        search_dataset_id = result.get('search_dataset_id')
        if task_id and search_dataset_id:
            task_to_search_id[task_id] = search_dataset_id
    
    # update metadata records
    updated_records = []
    for record in metadata_records:
        task_id = record.get('task_id')
        original_dataset_id = record.get('dataset_id')
        updated_record = record.copy()
        
        # if find corresponding search dataset, use search_id in the mapping
        if task_id in task_to_search_id:
            search_id = task_to_search_id[task_id]
            updated_record['search_id'] = search_id
            log_message(f"add search_id: {search_id} for task {task_id}")
        else:
            log_message(f"cannot find corresponding search dataset for task {task_id}")
        
        updated_records.append(updated_record)
    
    # save updated metadata file, use output path in configuration
    output_dir = args.results_output_dir if hasattr(args, 'results_output_dir') else os.path.join(args.train_set_dir, '..', '..', 'datasets', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    # build output filename, use search model name
    model_name = args.search_model_name.replace('/', '_').replace('-', '_')
    output_filename = f"search_{model_name}_metadata.json"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(updated_records, f, ensure_ascii=False, indent=2)
        log_message(f"✅ successfully save updated metadata file to: {output_path}")
        return True
    except Exception as e:
        log_message(f"❌ save updated metadata file failed: {e}")
        return False

# ========== argparse entry ==========

def main():
    parser = argparse.ArgumentParser(description="one-click dataset search and pipeline")
    parser.add_argument('--config', type=str,required=True, help='configuration file path')

    
    args = parser.parse_args()
    config = ConfigLoader.load_config(args.config)
    # build output directory with time and model name
    time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    parms = Namespace(**config)
    model_name = parms.search_model_name.replace('/','_')
    output_dir = f'./experiments/search/{time_str}_{model_name}'
    
    if 'experiments' not in parms.search_results:
        parms.search_results = os.path.join(output_dir, parms.search_results)
    else:
        output_dir = os.path.dirname(parms.search_results)
    if 'experiments' not in parms.check_result:
        parms.check_result = os.path.join(output_dir, parms.check_result)
    if 'experiments' not in parms.metadata_output:
        parms.metadata_output = os.path.join(output_dir, parms.metadata_output)
    os.makedirs(output_dir, exist_ok=True)
    log_message("🚀 start executing dataset search and pipeline")
    log_message(f"execute phase: {parms.step}")
    
    success = True
    search_llm_caller = CallLLM(
        model=parms.search_model_name,
        api_base=parms.search_api_base,
        api_key=parms.search_api_key,
    )
    llm_caller = CallLLM(
        model=parms.model_name, 
        api_base=parms.model_api_base, 
        api_key=parms.model_api_key
    )
    if parms.step in ['all', 'search']:
        log_message("=" * 50)
        success = run_search_agent(parms, search_llm_caller) and success
    
    if parms.step in ['all', 'check']:
        log_message("=" * 50)
        success = run_check_exist_gated(parms, llm_caller) and success
    
    if parms.step in ['all', 'generate']:
        log_message("=" * 50)
        success = run_generate_search_train_set(parms, llm_caller) and success
    
    if parms.step in ['all', 'metadata']:
        log_message("=" * 50)
        success = run_generate_search_set_metadata(parms, llm_caller) and success
    
    if parms.step in ['all', 'update_metadata']:
        log_message("=" * 50)
        success = update_metadata_with_search_id(parms, llm_caller) and success
    
    if success:
        log_message("🎉 pipeline executed successfully!")
    else:
        log_message("❌ error occurred during pipeline execution")
        sys.exit(1)

if __name__ == '__main__':
    main()
        