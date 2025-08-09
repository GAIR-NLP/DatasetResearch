import json
import time
from datetime import datetime
from huggingface_hub import dataset_info
from tqdm import tqdm

# we use gpt-4o-search-preview.json as input file
input_file = 'search_results_gpt-4o-search-preview.json'
output_file = 'check_exist_gated_gpt-4o-search-preview.json'

def log_message(message):
    """simplified log output function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)

def check_dataset_gated_status(dataset_id):
    """
    check if the dataset is gated (requires access permission)
    
    Args:
        dataset_id (str): dataset ID
    
    Returns:
        bool: True if gated, False if not gated, None if check failed
    """
    # first check if the dataset_id format is correct (must contain /)
    if '/' not in dataset_id:
        log_message(f"dataset ID format error, missing '/'：{dataset_id}")
        return None
    
    try:
        # try to get dataset information
        info = dataset_info(dataset_id)
        
        # check if there are gated related attributes
        if hasattr(info, 'gated') and info.gated:
            return True
        
        return False  # successfully accessed, not gated
                
    except Exception as e:
        error_msg = str(e).lower()
        # check error message
        gated_keywords = ['gated', 'access', 'authentication', 'token', 'private', 'permission']
        if any(keyword in error_msg for keyword in gated_keywords):
            return True
        
        log_message(f"check dataset {dataset_id} gated status failed: {e}")
        return None

def check_dataset_exists(dataset_id):
    """
    check if the dataset exists and is accessible on HuggingFace
    
    Args:
        dataset_id (str): dataset ID
    
    Returns:
        bool: True if exists and accessible, False if not exists or inaccessible
    """
    try:
        # try to get dataset basic information
        info = dataset_info(dataset_id)
        return True
    except Exception as e:
        error_msg = str(e).lower()
        # check if it is an error related to non-existence
        not_found_keywords = ['not found', '404', 'does not exist', 'repository not found']
        if any(keyword in error_msg for keyword in not_found_keywords):
            log_message(f"dataset {dataset_id} does not exist: {e}")
        else:
            log_message(f"check dataset {dataset_id} existence failed: {e}")
        return False

def process_single_search_result(item):
    """
    process a single search result, find the first dataset that exists and is not gated
    
    Args:
        item (dict): dictionary containing original_dataset_id and searched_datasets
    
    Returns:
        dict: dictionary containing original_dataset_id and selected_dataset
    """
    original_dataset_id = item.get('original_dataset_id', '')
    searched_datasets = item.get('searched_datasets', [])
    
    log_message(f"🔍 start processing: {original_dataset_id}")
    log_message(f"📋 candidate datasets: {searched_datasets}")
    
    if not searched_datasets:
        log_message(f"⚠️ {original_dataset_id} has no candidate datasets")
        return {
            "original_dataset_id": original_dataset_id,
            "selected_dataset": None
        }
    
    # check each dataset one by one from front to back
    for i, dataset_name in enumerate(searched_datasets):
        dataset_name = dataset_name.strip()
        
        if not dataset_name:
            log_message(f"⚠️ the {i+1}th dataset name is empty, skip")
            continue
        
        log_message(f"🔍 check the {i+1}th dataset: {dataset_name}")
        
        # check 1: dataset existence
        if not check_dataset_exists(dataset_name):
            log_message(f"❌ dataset {dataset_name} does not exist or is inaccessible, skip")
            continue
        
        log_message(f"✅ dataset {dataset_name} exists")
        
        # check 2: whether gated
        is_gated = check_dataset_gated_status(dataset_name)
        if is_gated is True:
            log_message(f"🔒 dataset {dataset_name} is gated, skip")
            continue
        elif is_gated is None:
            log_message(f"❓ cannot determine the gated status of dataset {dataset_name}, skip")
            continue
        
        log_message(f"🔓 dataset {dataset_name} is not gated")
        log_message(f"🎯 select dataset: {dataset_name}")
        
        return {
            "original_dataset_id": original_dataset_id,
            "selected_dataset": dataset_name
        }
    
    # if no suitable dataset is found
    log_message(f"❌ {original_dataset_id} has no dataset that exists and is not gated")
    return {
        "original_dataset_id": original_dataset_id,
        "selected_dataset": None
    }

def load_search_results(file_path):
    """
    load search results file
    
    Args:
        file_path (str): search results file path
    
    Returns:
        list: search results list, return None if failed
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            search_results = json.load(f)
        
        log_message(f"successfully loaded search results file, total {len(search_results)} records")
        return search_results
    except Exception as e:
        log_message(f"failed to load search results file: {e}")
        return None

def save_check_results(results, output_file):
    """
    save check results to file
    
    Args:
        results (list): check results list
        output_file (str): output file path
    
    Returns:
        bool: whether saving is successful
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        log_message(f"✅ check results saved to: {output_file}")
        return True
    except Exception as e:
        log_message(f"❌ failed to save check results: {e}")
        return False

def main():
    """
    main function: check the existence and gated status of datasets in search results
    """
    
    log_message("🎯 start checking the existence and gated status of datasets in search results")
    
    # load search results
    search_results = load_search_results(input_file)
    if not search_results:
        log_message("❌ failed to load search results, task terminated")
        return
    
    # process each search result
    check_results = []
    
    log_message(f"🚀 start processing {len(search_results)} search results")
    
    for item in tqdm(search_results, desc="check datasets"):
        try:
            result = process_single_search_result(item)
            check_results.append(result)
            
            # add a short delay to avoid too frequent requests
            time.sleep(0.1)
            
        except Exception as e:
            log_message(f"❌ failed to process {item.get('original_dataset_id', 'unknown')}: {e}")
            check_results.append({
                "original_dataset_id": item.get('original_dataset_id', 'unknown'),
                "selected_dataset": None
            })
    
    # save check results
    if save_check_results(check_results, output_file):
        log_message("✅ check task completed")
    else:
        log_message("❌ failed to save results")
        return
    
    # print statistics
    total_count = len(check_results)
    successful_count = sum(1 for result in check_results if result.get('selected_dataset') is not None)
    failed_count = total_count - successful_count
    
    log_message(f"📊 final statistics:")
    log_message(f"   - total number: {total_count}")
    log_message(f"   - successfully found dataset: {successful_count}")
    log_message(f"   - not found suitable dataset: {failed_count}")
    log_message(f"   - success rate: {successful_count/total_count*100:.1f}%")
    log_message(f"   - results saved to: {output_file}")

if __name__ == "__main__":
    main()