#!/usr/bin/env python3
"""
JSON dataset conversion script based on templates

use string templates to convert JSON data to instruction tuning format
support processing multiple question-answer pairs in a single item

usage:
python scripts/template_converter.py --input data.json --output instruction_data.json --template template.py

template.py content example:

# template for single question-answer pair
TEMPLATE = {
    "instruction": "answer the question: {sample.Question}",
    "input": "we have the following search results: {sample.SearchResults}",
    "output": "{sample.Answer.Value}"
}

# template for multiple question-answer pairs
TEMPLATE_MULTI = {
    "questions_field": "Questions",  # specify the field name containing multiple questions
    "template": {
        "instruction": "answer the question: {sample.Question}",
        "input": "context: {sample.Context}",  # shared context information
        "output": "{sample.Answer}"
    }
}
"""

import json
import argparse
import sys
import os
from types import SimpleNamespace
import re
from typing import Dict, List, Any, Optional

def dict_to_namespace(d):
    """recursively convert dictionary to SimpleNamespace, support nested access"""
    if isinstance(d, dict):
        namespace = SimpleNamespace()
        for key, value in d.items():
            # process special characters in dictionary keys
            safe_key = key.replace(' ', '_').replace('-', '_')
            if safe_key.isidentifier():
                setattr(namespace, safe_key, dict_to_namespace(value))
            else:
                # for keys that cannot be used as attribute names, put them in a special dictionary
                if not hasattr(namespace, '_dict_items'):
                    namespace._dict_items = {}
                namespace._dict_items[key] = dict_to_namespace(value)
        return namespace
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

def get_nested_value(obj, path):
    """get the value of nested objects, support dot-separated paths"""
    try:
        keys = path.split('.')
        value = obj
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                return ""
        
        # if it is a complex object, convert it to a string
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False)
        
        return str(value) if value is not None else ""
    except:
        return ""

def load_template(template_file):
    """load template file"""
    if not os.path.exists(template_file):
        print(f"template file does not exist: {template_file}")
        return None
    
    # dynamically import template file
    sys.path.insert(0, os.path.dirname(os.path.abspath(template_file)))
    module_name = os.path.splitext(os.path.basename(template_file))[0]
    
    try:
        template_module = __import__(module_name)
        if hasattr(template_module, 'TEMPLATE'):
            return template_module.TEMPLATE
        else:
            print(f"TEMPLATE variable not found in template file")
            return None
    except Exception as e:
        print(f"failed to load template file: {e}")
        return None

def eval_expression(expression: str, sample_data: Dict) -> str:
    """
    safely execute Python expressions, support list comprehensions and other complex expressions
    
    Args:
        expression: Python expression to execute
        sample_data: sample data dictionary
        
    Returns:
        str: expression execution result converted to a string
    """
    try:
        # create a safe execution environment
        safe_globals = {
            '__builtins__': {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'sorted': sorted,
                'sum': sum,
                'max': max,
                'min': min,
                'any': any,
                'all': all,
                'enumerate': enumerate,
                'zip': zip,
                'range': range,
                'filter': filter,
                'map': map,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'getattr': getattr,
                'json': json,
            }
        }
        
        # add sample data to the execution context
        safe_locals = {
            'sample': sample_data,
            'd': sample_data,  # provide d as an alias for sample
            'data': sample_data,  # provide data as an alias for sample
        }
        
        # execute expression
        result = eval(expression, safe_globals, safe_locals)
        
        # convert result to a string
        if isinstance(result, (list, dict)):
            return json.dumps(result, ensure_ascii=False)
        else:
            return str(result) if result is not None else ""
            
    except Exception as e:
        print(f"⚠️ expression execution failed: {expression}")
        print(f"    error information: {e}")
        return f"[ERROR: {str(e)}]"

def replace_template_vars(template_str, sample):
    """replace variables and execute code expressions in the template string"""
    def replace_match(match):
        content = match.group(1)  # get the content inside {}
        
        # check if it contains Python code features (spaces, for, if, in, etc.)
        code_indicators = [' for ', ' if ', ' in ', '(', ')', '[', ']', '==', '!=', '>', '<', ' and ', ' or ']
        is_code = any(indicator in content for indicator in code_indicators)
        
        if is_code:
            # if it is a code expression, execute it
            return eval_expression(content, sample)
        else:
            # if it is a simple field reference
            if content.startswith('sample.'):
                field_path = content[7:]  # remove the 'sample.' prefix
                return get_nested_value(sample, field_path)
            else:
                # try to be a sample field
                return get_nested_value(sample, content)
    
    # find all {} patterns
    pattern = r'\{([^}]+)\}'
    result = re.sub(pattern, replace_match, template_str)
    return result

def apply_template(template, sample_data):
    """apply template to sample data, support multiple question-answer pairs"""
    try:
        # check if it is a multiple question-answer pair mode
        if isinstance(template, dict) and 'questions_field' in template:
            return apply_multi_qa_template(template, sample_data)
        
        # single question-answer pair mode
        sample = sample_data
        result = {}
        for key, template_str in template.items():
            if isinstance(template_str, str):
                result[key] = replace_template_vars(template_str, sample)
            else:
                result[key] = str(template_str)
        
        return result
    except Exception as e:
        print(f"failed to apply template: {e}")
        print(f"sample data: {sample_data}")
        return None

def apply_multi_qa_template(template_config: Dict, sample_data: Dict) -> List[Dict]:
    """
    process multiple question-answer pairs
    
    Args:
        template_config: configuration dictionary containing questions_field and template
        sample_data: original data
        
    Returns:
        List[Dict]: converted multiple question-answer pairs
    """
    try:
        questions_field = template_config['questions_field']
        template = template_config['template']
        
        # get the question-answer pairs list
        qa_list = sample_data.get(questions_field, [])
        if not isinstance(qa_list, list):
            print(f"warning: {questions_field} field is not a list format")
            return []
            
        results = []
        # process each question-answer pair
        for qa_item in qa_list:
            # create new sample data, merge shared information and current question-answer pair
            current_sample = {
                **sample_data,  # keep shared information in the original data
                **(qa_item if isinstance(qa_item, dict) else {'qa': qa_item})  # add current question-answer pair
            }
            
            # apply template
            result = {}
            for key, template_str in template.items():
                if isinstance(template_str, str):
                    result[key] = replace_template_vars(template_str, current_sample)
                else:
                    result[key] = str(template_str)
            
            results.append(result)
            
        return results
        
    except Exception as e:
        print(f"failed to process multiple question-answer pairs: {e}")
        return []

def load_data(input_file: str) -> List[Dict]:
    """
    load data file, support JSON and JSONL format
    
    Args:
        input_file: input file path
        
    Returns:
        List[Dict]: loaded data list
    """
    print(f"📖 reading file: {input_file}")
    
    try:
        # first try to load as JSON
        with open(input_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
                print("✅ successfully loaded file as JSON")
                # ensure the data is a list format
                if isinstance(data, dict):
                    if 'Data' in data:
                        data = data['Data']
                    else:
                        data = [data]
                    # data = list(zip(data.keys(),data.values()))
                        
                return data
                
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON format parsing failed: {e}")
                print("🔄 trying to load as JSONL...")
                
                # file pointer back to the beginning
                f.seek(0)
                
                # try to load as JSONL
                data = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # skip empty lines
                        continue
                        
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"⚠️ JSONL parsing failed at line {line_num}: {e}")
                        print(f"    problematic line content: {line[:100]}...")
                        continue
                
                if data:
                    print(f"✅ successfully loaded file as JSONL (total {len(data)} data)")
                    return data
                else:
                    raise ValueError("no data loaded")
                    
    except Exception as e:
        print(f"❌ file reading failed: {e}")
        return []

def convert_dataset(input_file, output_file, template):
    """convert dataset"""
    # read data
    data = load_data(input_file)
    
    if not data:
        print("❌ no valid data read, exiting conversion")
        return
    
    print(f"📊 read {len(data)} data items")
    
    # convert data
    converted_data = []
    
    for i, item in enumerate(data):
        try:
            results = apply_template(template, item)
            if results:
                # process multiple question-answer pairs
                if isinstance(results, list):
                    converted_data.extend(results)
                else:
                    converted_data.append(results)
            else:
                print(f"⚠️ skip item {i}: template application failed")
        except Exception as e:
            print(f"❌ error processing item {i}: {e}")
    
    if not converted_data:
        print("❌ no valid data generated")
        return
    
    # save results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ conversion completed: {len(converted_data)} valid items")
        print(f"💾 results saved to: {output_file}")
        
        # display examples
        print("\n📝 conversion results example:")
        example = converted_data[0]
        for key, value in example.items():
            display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            print(f"{key}: {display_value}")
            
    except Exception as e:
        print(f"❌ saving results failed: {e}")

def create_sample_template():
    """create sample template file"""
    template_content = '''# template file example
# use {sample.field_name} to reference fields in JSON

# template for single question-answer pair
TEMPLATE = {
    "instruction": "answer the question: {sample.Question}",
    "input": "context: {sample.Context}",
    "output": "{sample.Answer}"
}

# template for multiple question-answer pairs
TEMPLATE_MULTI = {
    # specify the field name containing multiple questions
    "questions_field": "Questions",
    
    # template for each question-answer pair
    "template": {
        "instruction": "answer the question: {sample.Question}",
        "input": "context: {sample.Context}\\ndocument ID: {sample.DocID}",  # can reference shared fields
        "output": "{sample.Answer}"
    }
}

# use which template (uncomment the template you want to use)
TEMPLATE = TEMPLATE_MULTI  # use multiple question-answer pairs template
# TEMPLATE = TEMPLATE  # use single question-answer pair template
'''
    
    with open('template_sample.py', 'w', encoding='utf-8') as f:
        f.write(template_content)
    
    print("sample template file created: template_sample.py")

def main():
    parser = argparse.ArgumentParser(description='JSON dataset conversion based on templates')
    parser.add_argument('--input', '-i', required=True, help='input JSON file path')
    # parser.add_argument('--output', '-o', required=True, help='output JSON file path')
    parser.add_argument('--template', '-t', required=True, help='template file path (.py file)')
    parser.add_argument('--create-sample', action='store_true', help='create sample template file')
    
    args = parser.parse_args()
    directory, old_file_name = os.path.split(args.input)
    name = 'process.json'
    output = os.path.join(directory, old_file_name)
    # if args.create_sample:
    #     create_sample_template()
    #     return
    
    # load template
    template = load_template(args.template)
    if not template:
        return
    
    # convert dataset
    convert_dataset(args.input, output, template)

if __name__ == "__main__":
    main()