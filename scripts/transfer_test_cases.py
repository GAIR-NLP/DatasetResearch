#!/usr/bin/env python3
"""
convert the test case dataset to a classification task format

input: JSON data containing test scenarios and cases
output: classification dataset, input is scenario+case, output is success/failure
"""

import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple

# configure the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_test_cases(data: List[Dict]) -> List[Dict]:
    """
    extract test cases from the original data, labeled as success/failure
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: converted classification dataset
    """
    classification_data = []
    
    for item in data:
        try:
            category = item.get('Category', '')
            scenario = item.get('Scenario', '')
            github_url = item.get('Github URL', '')
            
            # extract all test cases
            test_cases = [
                ('Success Case 1', 'success'),
                ('Failure Case 1', 'failure'),
                ('Success Case 2', 'success'),
                ('Failure Case 2', 'failure')
            ]
            
            for case_key, label in test_cases:
                case_content = item.get(case_key, '')
                
                if case_content.strip():
                    # create a classification data item
                    classification_item = {
                        "system": "You are a test case classifier. Given a scenario and test case, classify whether it represents a success or failure case.",
                        "input": f"Scenario: {scenario}\n\nTest Case: {case_content}",
                        "output": label,
                        "metadata": {
                            "category": category,
                            "scenario": scenario,
                            "case_type": case_key,
                            "github_url": github_url
                        }
                    }
                    
                    classification_data.append(classification_item)
                    
        except Exception as e:
            logger.error(f"❌ error processing data item: {e}")
            continue
    
    return classification_data

def create_simple_classification_format(data: List[Dict]) -> List[Dict]:
    """
    create a simplified classification format, only containing scenario and case content as input
    
    Args:
        data: original JSON data
        
    Returns:
        List[Dict]: simplified classification dataset
    """
    simple_data = []
    
    for item in data:
        try:
            scenario = item.get('Scenario', '')
            
            # extract all test cases
            test_cases = [
                ('Success Case 1', 'success'),
                ('Failure Case 1', 'failure'),
                ('Success Case 2', 'success'),
                ('Failure Case 2', 'failure')
            ]
            
            for case_key, label in test_cases:
                case_content = item.get(case_key, '')
                
                if case_content.strip():
                    # create a simplified classification data item
                    simple_item = {
                        "scenario": scenario,
                        "case": case_content,
                        "label": label
                    }
                    
                    simple_data.append(simple_item)
                    
        except Exception as e:
            logger.error(f"❌ error processing data item: {e}")
            continue
    
    return simple_data

def load_data(input_file: str) -> List[Dict]:
    """
    load JSON data
    
    Args:
        input_file: input file path
        
    Returns:
        List[Dict]: loaded data
    """
    logger.info(f"📖 reading file: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            else:
                data = [data]
                
        logger.info(f"✅ successfully loaded {len(data)} data items")
        return data
        
    except Exception as e:
        logger.error(f"❌ file reading failed: {e}")
        return []

def convert_dataset(input_file: str, output_file: str, format_type: str = "instruction"):
    """
    convert the dataset
    
    Args:
        input_file: input file path
        output_file: output file path
        format_type: output format type ("instruction" or "simple")
    """
    # read data
    data = load_data(input_file)
    
    if not data:
        logger.error("❌ no valid data read, exiting conversion")
        return
    
    logger.info(f"📊 read {len(data)} original data items")
    
    # convert data
    if format_type == "instruction":
        converted_data = extract_test_cases(data)
        logger.info("📝 using instruction tuning format")
    else:
        converted_data = create_simple_classification_format(data)
        logger.info("📝 using simplified classification format")
    
    if not converted_data:
        logger.error("❌ no valid classification data generated")
        return
    
    # 统计标签分布
    label_counts = {}
    for item in converted_data:
        if format_type == "instruction":
            label = item.get("output", "unknown")
        else:
            label = item.get("label", "unknown")
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 保存结果
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ conversion completed!")
        logger.info(f"📊 statistics:")
        logger.info(f"   - original data items: {len(data)}")
        logger.info(f"   - generated classification samples: {len(converted_data)}")
        logger.info(f"   - label distribution: {label_counts}")
        logger.info(f"💾 results saved to: {output_file}")
        
        # display examples
        if converted_data:
            logger.info("\n📝 conversion results example:")
            example = converted_data[0]
            
            if format_type == "instruction":
                logger.info(f"System: {example.get('system', '')[:100]}...")
                logger.info(f"Input: {example.get('input', '')[:200]}...")
                logger.info(f"Output: {example.get('output', '')}")
                logger.info(f"Metadata: {example.get('metadata', {})}")
            else:
                logger.info(f"Scenario: {example.get('scenario', '')}")
                logger.info(f"Case: {example.get('case', '')[:200]}...")
                logger.info(f"Label: {example.get('label', '')}")
            
    except Exception as e:
        logger.error(f"❌ saving results failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='convert test case dataset to a classification task format')
    parser.add_argument('--input', '-i', required=True, help='input JSON file path')
    parser.add_argument('--output', '-o', required=True, help='output file path')
    parser.add_argument('--format', '-f', choices=['instruction', 'simple'], 
                       default='instruction', help='output format (instruction/simple)')
    
    args = parser.parse_args()
    convert_dataset(args.input, args.output, args.format)

if __name__ == "__main__":
    main() 