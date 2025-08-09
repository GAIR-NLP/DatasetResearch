#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSON format check and clean script
check all JSON files in the directory whether they are in the correct format: 
the list contains dictionaries, and each dictionary only has three keys: system, input, output
unconformant keys or elements will be deleted
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

def is_valid_dict(item: Any) -> bool:
    """
    check if the item is a valid dictionary (only contains system, input, output three keys)
    """
    if not isinstance(item, dict):
        return False
    
    required_keys = {"system", "input", "output"}
    return set(item.keys()) == required_keys

def clean_dict(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    clean the dictionary, only keep system, input, output three keys
    """
    required_keys = {"system", "input", "output"}
    return {k: v for k, v in item.items() if k in required_keys}

def process_json_data(data: Any) -> Tuple[List[Dict[str, Any]], bool]:
    """
    process the JSON data, return the cleaned data and whether it has been modified
    """
    if not isinstance(data, list):
        print(f"   warning: the root element is not a list, cannot be processed")
        return [], True
    
    cleaned_data = []
    modified = False
    
    for i, item in enumerate(data):
        if isinstance(item, dict):
            # check if the dictionary's key is conformant
            required_keys = {"system", "input", "output"}
            item_keys = set(item.keys())
            
            if item_keys == required_keys:
                # the dictionary is conformant
                cleaned_data.append(item)
            elif required_keys.issubset(item_keys):
                # the dictionary contains the required keys, but has extra keys, need to clean
                cleaned_item = clean_dict(item)
                cleaned_data.append(cleaned_item)
                modified = True
                print(f"  clean dictionary[{i}]: deleted extra keys {item_keys - required_keys}")
            else:
                # the dictionary is missing the required keys, delete the whole item
                missing_keys = required_keys - item_keys
                modified = True
                print(f"  delete dictionary[{i}]: missing required keys {missing_keys}")
        else:
            # the item is not a dictionary, delete it
            modified = True
            print(f"  delete item[{i}]: not a dictionary, but {type(item).__name__}")
    
    return cleaned_data, modified

def process_json_file(file_path: Path) -> bool:
    """
    process a single JSON file
    return True if the file is modified, False if not
    """
    try:
        # read the JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"process file: {file_path}")
        
        # process the data
        cleaned_data, modified = process_json_data(data)
        
        if modified:
            # backup the original file
            backup_path = file_path.with_suffix('.json.backup')
            if not backup_path.exists():
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  backup created: {backup_path}")
            
            # write the cleaned data
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            
            print(f"  file modified, {len(cleaned_data)} valid items retained")
            return True
        else:
            print(f"  file format correct, no modification needed")
            return False
            
    except json.JSONDecodeError as e:
        print(f"  error: JSON format error - {e}")
        return False
    except Exception as e:
        print(f"  error: error occurred while processing the file - {e}")
        return False

def main():
    """
    main function
    """
    if len(sys.argv) != 2:
        print("usage: python script.py <directory path>")
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.exists():
        print(f"error: directory '{directory}' does not exist")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"error: '{directory}' is not a directory")
        sys.exit(1)
    
    # find all JSON files
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        print(f"no JSON files found in directory '{directory}'")
        return
    
    print(f"found {len(json_files)} JSON files")
    print("=" * 50)
    
    modified_count = 0
    
    for json_file in json_files:
        if process_json_file(json_file):
            modified_count += 1
        print("-" * 30)
    
    print("=" * 50)
    print(f"processing completed!")
    print(f"total files: {len(json_files)}")
    print(f"modified files: {modified_count}")
    print(f"correct files: {len(json_files) - modified_count}")
    
    if modified_count > 0:
        print(f"\nwarning: modified files have been automatically created backup (.json.backup)")

if __name__ == "__main__":
    main()