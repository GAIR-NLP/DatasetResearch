#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset registration and file renaming tool
1. batch rename files, replace dollar symbol $ with underscore _
2. register datasets to a unified JSON file
"""

import os
import json
import argparse
from pathlib import Path


def rename_files_in_directory(directory_path, dry_run=True, mode='dollar_to_underscore'):
    """
    batch rename files in the directory
    
    Args:
        directory_path (str): directory path
        dry_run (bool): whether to run in dry run mode, True means only display the operations, not actually execute
        mode (str): rename mode, 'dollar_to_underscore' or 'remove_before_ampersand'
    
    Returns:
        tuple: (number of successfully renamed files, number of failed files)
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"❌ directory does not exist: {directory_path}")
        return 0, 0
    
    if not directory.is_dir():
        print(f"❌ path is not a directory: {directory_path}")
        return 0, 0
    
    success_count = 0
    failed_count = 0
    
    print(f"📁 scanning directory: {directory_path}")
    print(f"{'🔍 dry run mode' if dry_run else '🚀 execution mode'}")
    print(f"🎯 rename mode: {mode}")
    print("-" * 60)
    
    # iterate through the first level files and directories
    for item_path in directory.iterdir():
        old_name = item_path.name
        
        # rename according to the mode
        if mode == 'dollar_to_underscore':
            # new_name = old_name.replace('$', '_').replace('.json','_generation.json')
            new_name = f"{old_name.replace('_generation.json','')}2{old_name}"
        elif mode == 'remove_before_ampersand':
            if '&' in old_name:
                new_name = old_name.split('&', 1)[1]  # take the part after &
            else:
                new_name = old_name  # if there is no &, keep it unchanged
        
        # if the file name needs to be renamed
        if old_name != new_name:
            new_path = item_path.parent / new_name
            
            # check if the new file name already exists
            if new_path.exists():
                print(f"⚠️  skip {old_name} -> {new_name} (target file already exists)")
                failed_count += 1
                continue
            
            if dry_run:
                print(f"📝 will rename: {old_name} -> {new_name}")
            else:
                try:
                    item_path.rename(new_path)
                    print(f"✅ renamed: {old_name} -> {new_name}")
                    success_count += 1
                except Exception as e:
                    print(f"❌ rename failed: {old_name} -> {new_name} (error: {e})")
                    failed_count += 1
        else:
            # file name does not need to be renamed
            if not dry_run:
                if mode == 'dollar_to_underscore':
                    print(f"⏭️  skip: {old_name} (does not contain $ symbol)")
                elif mode == 'remove_before_ampersand':
                    print(f"⏭️  skip: {old_name} (does not contain & symbol)")
    
    return success_count, failed_count


def register_datasets(input_json_path, output_json_path, key_field='search_dataset_id', dir_name='deep_research_dataset/gemini', model='gemini', template_type='generation', train_set_field='original_dataset_id', test_set_field='search_dataset_id'):
    """
    register datasets in the input JSON file to the output JSON file
    
    Args:
        input_json_path (str): input JSON file path, containing dataset list
        output_json_path (str): output JSON file path, for storing registered datasets
        key_field (str): field name used as dictionary key, default is 'search_dataset_id'
        dir_name (str): dataset file directory name
        model (str): model name
        template_type (str): template type, 'test_set' or 'generation'
        train_set_field (str): train set field name, default is 'original_dataset_id'
        test_set_field (str): test set field name, default is 'search_dataset_id'
    
    Returns:
        tuple: (number of successfully registered datasets, number of failed datasets)
    """
    try:
        # read input JSON file
        with open(input_json_path, 'r', encoding='utf-8') as f:
            datasets = json.load(f)
        
        if not isinstance(datasets, list):
            print(f"❌ input JSON file should contain a list, current type: {type(datasets)}")
            return 0, 0
        
        print(f"📋 loaded {len(datasets)} datasets from {input_json_path}")
        
        # read or create output JSON file
        output_data = {}
        if os.path.exists(output_json_path):
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                print(f"📁 loaded existing output file: {output_json_path}")
            except json.JSONDecodeError as e:
                print(f"⚠️  output file format error, will create new file: {e}")
                output_data = {}
        else:
            print(f"📁 create new output file: {output_json_path}")
        
        success_count = 0
        failed_count = 0
        
        print("-" * 60)
        print("🔄 start registering datasets...")
        
        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                print(f"❌ skip the {i+1}th element (not a dictionary)")
                failed_count += 1
                continue
            
            # check required key field
            if key_field not in dataset:
                print(f"❌ skip the {i+1}th dataset (missing key field: {key_field})")
                failed_count += 1
                continue
            
            # generate dataset name and key according to the template type
            if template_type == 'test_set':
                # test_set template: file name and key are the original dataset name (/ converted to _)
                dataset_name = dataset[test_set_field].replace('/', '_')
                dataset_key = dataset_name
            else:
                # generation template: {train_set}2{test_set}_{model}
                train_name = dataset[train_set_field].replace('/', '_') if train_set_field in dataset else 'unknown'
                test_name = dataset[test_set_field].replace('_generation', '').replace('/', '_')
                dataset_name = test_name
                dataset_key = f"{train_name}2{test_name}_{model}"
            
            # check if it already exists
            if dataset_key in output_data:
                print(f"⚠️  skip {dataset_key} (already exists)")
                failed_count += 1
                continue
                
            json_file_name = f'{dataset_name}.json'
            # register dataset
            output_data[dataset_key] = {
                "file_name": os.path.join(dir_name, json_file_name),
                "columns": {
                    # "system": "system",
                    "prompt": "input",
                    "response": "output",
                }
            }
            print(f"✅ register: {dataset_key}")
            success_count += 1
        
        # save to output file
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("-" * 60)
        print(f"💾 saved to: {output_json_path}")
        
        return success_count, failed_count
        
    except FileNotFoundError:
        print(f"❌ input file does not exist: {input_json_path}")
        return 0, 0
    except json.JSONDecodeError as e:
        print(f"❌ input JSON file format error: {e}")
        return 0, 0
    except Exception as e:
        print(f"❌ error occurred during processing: {e}")
        return 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="dataset registration and file renaming tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
usage examples:
  file renaming:
    python register_and_unified.py rename /path/to/directory          # dry run mode ($ replaced with _)
    python register_and_unified.py rename /path/to/directory --execute # execute renaming ($ replaced with _)
    python register_and_unified.py rename /path/to/directory --mode remove_before_ampersand # remove content before &
    python register_and_unified.py rename /path/to/directory --mode remove_before_ampersand --execute # execute removing content before &
  
  dataset registration:
    # basic usage
    python register_and_unified.py register input.json output.json
    
    # specify model and directory
    python register_and_unified.py register input.json output.json --model gpt4o --dir-name data/gpt4o
    
    # use test_set template (file name and key are the dataset name)
    python register_and_unified.py register input.json output.json --template-type test_set
    
    # use generation template (train2test_model format)
    python register_and_unified.py register input.json output.json --template-type generation --model gemini
    
    # custom field names
    python register_and_unified.py register input.json output.json --train-set-field train_data --test-set-field test_data
    
    # complete example
    python register_and_unified.py register datasets/metadata.json LLaMA-Factory/data/dataset_info.json \\
        --template-type generation --model gemini --dir-name deep_research_dataset/gemini \\
        --train-set-field original_dataset_id --test-set-field search_dataset_id
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='available commands')
    
    # rename command
    rename_parser = subparsers.add_parser('rename', help='batch rename files')
    rename_parser.add_argument(
        "directory",
        help="directory path to process"
    )
    rename_parser.add_argument(
        "--mode",
        choices=['dollar_to_underscore', 'remove_before_ampersand'],
        default='dollar_to_underscore',
        help="rename mode: dollar_to_underscore($ replaced with _) or remove_before_ampersand(remove content before &)"
    )
    rename_parser.add_argument(
        "--execute",
        action="store_true",
        help="execute renaming (default is dry run mode)"
    )
    rename_parser.add_argument(
        "--recursive",
        action="store_true",
        help="recursive processing of subdirectories (default is only process the specified directory)"
    )
    
    # register command
    register_parser = subparsers.add_parser('register', help='register datasets to a unified JSON file')
    register_parser.add_argument(
        "input_json",
        help="input JSON file path, containing dataset list"
    )
    register_parser.add_argument(
        "output_json",
        help="output JSON file path, for storing registered datasets"
    )
    register_parser.add_argument(
        "--key-field",
        default="search_dataset_id",
        help="field name used as dictionary key, default is 'search_dataset_id'"
    )
    register_parser.add_argument(
        "--dir-name",
        default="deep_research_dataset/gemini",
        help="dataset file directory name, default is 'deep_research_dataset/gemini'"
    )
    register_parser.add_argument(
        "--model",
        default="gemini",
        help="model name, default is 'gemini'"
    )
    register_parser.add_argument(
        "--template-type",
        choices=['test_set', 'generation'],
        default="generation",
        help="template type: test_set (file name and key are the dataset name) or generation (train2test_model format)"
    )
    register_parser.add_argument(
        "--train-set-field",
        default="original_dataset_id",
        help="train set field name, default is 'original_dataset_id'"
    )
    register_parser.add_argument(
        "--test-set-field",
        default="search_dataset_id",
        help="test set field name, default is 'search_dataset_id'"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    if args.command == 'rename':
        # get absolute path
        directory_path = os.path.abspath(args.directory)
        
        print("🔄 batch file renaming tool")
        print("=" * 60)
        
        if args.execute:
            # confirm execution
            print(f"⚠️  will execute actual renaming operation!")
            print(f"📁 target directory: {directory_path}")
            response = input("confirm to continue? (y/N): ").strip().lower()
            
            if response not in ['y', 'yes']:
                print("❌ operation cancelled")
                return
        
        # execute renaming
        success, failed = rename_files_in_directory(
            directory_path, 
            dry_run=not args.execute,
            mode=args.mode
        )
        
        print("-" * 60)
        print("📊 operation result summary:")
        print(f"✅ success: {success} files")
        print(f"❌ failed: {failed} files")
        print(f"📁 total: {success + failed} files")
        
        if not args.execute and success > 0:
            print("\n💡 hint: use --execute parameter to actually execute renaming")
    
    elif args.command == 'register':
        print("🔄 dataset registration tool")
        print("=" * 60)
        
        # execute dataset registration
        success, failed = register_datasets(
            args.input_json,
            args.output_json,
            args.key_field,
            args.dir_name,
            args.model,
            args.template_type,
            args.train_set_field,
            args.test_set_field
        )
        
        print("-" * 60)
        print("📊 registration result summary:")
        print(f"✅ success: {success} datasets")
        print(f"❌ failed: {failed} datasets")
        print(f"📁 total: {success + failed} datasets")


if __name__ == "__main__":
    main()