import os
import json
import sys
sys.path.append(os.getcwd())
from typing import Dict, List, Optional, Any
from scripts.utils.call_llm import CallLLM

def load_existing_results(output_file: str) -> List[Dict]:
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
                print(f"Failed to load existing results from {output_file}: {str(e)}")
                return []
        return []

def save_single_result(result: Dict, output_file: str) -> None:
        """
        Save a single result to output file by appending to existing results
        
        Args:
            result: Single processed benchmark result
            output_file: Path to output JSON file
        """
        try:
            # Load existing results
            existing_results = load_existing_results(output_file)
            
            # Add new result
            existing_results.append(result)
            
            # Save updated results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_results, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Failed to save result to {output_file}: {str(e)}")
def transfer_metadata_paperwithcode(sample:Dict[str, Any]) -> Dict[str, Any]:
    """
    transfer metadata to json format
    """
    processed_sample = {}
    if sample['bench_id'] == 'long-form-narrative-summarization-on-booksum':
        return None
    data_path = f"./batch_downloaded_datasets/{sample['dataset']}/process.json"
    print(data_path)
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f)
        samples_count = len(data)
    else:
        samples_count = 0
    processed_sample['task_id'] = sample['bench_id']
    processed_sample['dataset_id'] = sample['dataset']
    processed_sample['query'] = sample['natural_language_query']
    processed_sample['introduction'] = f'The task area is: {sample["structured_query"]["area"]}, and the task description is: {sample["structured_query"]["task"]}. The dataset scale is: {sample["structured_query"]["scale"]}'

    
    
    # 使用llm来判断属于哪个task_type
    llm_caller = CallLLM(api_base="https://guohe-apim.azure-api.net", api_key="f847dd7d5eff4fc0bff57d061813a4ab", model="o3-mini")
    messages = [
        {"role": "system", "content": "You are a helpful assistant that can help me classify the task type of a dataset. The task types only are: multiple-choice, question-answering, summarization, text-classification, text-generation, translation. You should only return the task type, no other text."},
        {"role": "user", "content": f"The dataset is {sample['dataset']}, the introduction is {processed_sample['introduction']}, the question is {sample['structured_query']['question']} and the sample example is {sample['scheme']}"}
    ]
    response, prompt_tokens, completion_tokens = llm_caller.post_request(messages)
    processed_sample['task_type'] = response
    processed_sample['question'] = sample['structured_query']['question']
    processed_sample['input'] = sample['structured_query']['input']
    processed_sample['output'] = sample['structured_query']['output']
    processed_sample['source'] = "real-world"
    processed_sample['example'] = sample['scheme']
    processed_sample['samples_count'] = samples_count
    return processed_sample

def transfer_metadata_huggingface(sample:Dict[str, Any]) -> Dict[str, Any]:
    """
    transfer metadata to json format
    """
    processed_sample = {}
    data_path = f"./batch_download_datasets/{sample['dataset_id']}/process.json"
    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            data = json.load(f) 


if __name__ == "__main__":
    with open("./datasets/generated_queries.json", "r") as f:
        data = json.load(f)
    processed_data = []
    output_file = "./datasets/metadata.json"
    with open(output_file, "w") as f:
        json.dump(processed_data, f, indent=4)
    
    for sample in data:
        print(sample['bench_id'])
        processed_sample = transfer_metadata_paperwithcode(sample)
        if processed_sample == None:
            continue
        save_single_result(processed_sample,output_file)