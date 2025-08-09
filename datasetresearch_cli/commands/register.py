#!/usr/bin/env python3
"""
Register command module for Dataset Research CLI
Register datasets from search/synthesis metadata to dataset_info.json
"""

import json
import os
from typing import Tuple


def register_datasets(
    metadata_file: str,
    output_file: str,
    dataset_id_field: str = "search_dataset_id",
    base_dir: str = "search_dataset",
    model: str = "gemini",
    template_type: str = "generation",
    original_dataset_field: str = "original_dataset_id"
) -> Tuple[int, int]:
    """
    Register search datasets to dataset_info.json
    
    Args:
        metadata_file: Path to metadata file from search/synthesis
        output_file: Path to dataset_info.json
        dataset_id_field: Field name for dataset ID, default is 'search_dataset_id'
        base_dir: Base directory name (search_dataset or synthesis)
        model: Model name for registration
        template_type: Template type for dataset naming
        original_dataset_field: Field name for original dataset ID
        
    Returns:
        Tuple of (success_count, failed_count)
    """
    try:
        # Load metadata file
        with open(metadata_file, 'r', encoding='utf-8') as f:
            datasets = json.load(f)
        
        if not isinstance(datasets, list):
            print(f"❌ Metadata file should contain a list, got {type(datasets)}")
            return 0, 0
        
        print(f"📋 Loaded {len(datasets)} datasets from {metadata_file}")
        
        # Load or create output dataset_info.json
        output_data = {}
        if os.path.exists(output_file):
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
                print(f"📁 Loaded existing dataset_info.json: {output_file}")
            except json.JSONDecodeError as e:
                print(f"⚠️  Output file format error, creating new file: {e}")
                output_data = {}
        else:
            print(f"📁 Creating new dataset_info.json: {output_file}")
        
        success_count = 0
        failed_count = 0
        
        print("-" * 60)
        print("🔄 Starting dataset registration...")
        
        for i, dataset in enumerate(datasets):
            if not isinstance(dataset, dict):
                print(f"❌ Skipping {i+1}th item (not a dictionary)")
                failed_count += 1
                continue
            
            # Check required fields
            if dataset_id_field not in dataset:
                print(f"❌ Skipping {i+1}th dataset (missing field: {dataset_id_field})")
                failed_count += 1
                continue
            
            dataset_id = dataset[dataset_id_field]
            
            # Use search_dataset_id directly as the key and file name
            dataset_key = dataset_id  # search_dataset_id becomes the key directly
            json_file_name = f"{dataset_id}.json"  # file name is search_dataset_id.json
            
            # Check if already exists
            if dataset_key in output_data:
                print(f"⚠️  Skipping {dataset_key} (already exists)")
                failed_count += 1
                continue
            
            # Register dataset - file_name相对于LLaMA-Factory/data目录
            output_data[dataset_key] = {
                "file_name": os.path.join(base_dir, model, json_file_name),
                "columns": {
                    "prompt": "input",
                    "response": "output",
                }
            }
            print(f"✅ Registered: {dataset_key}")
            success_count += 1
        
        # Save to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print("-" * 60)
        print(f"💾 Saved to: {output_file}")
        
        return success_count, failed_count
        
    except FileNotFoundError:
        print(f"❌ Input file not found: {metadata_file}")
        return 0, 0
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return 0, 0
    except Exception as e:
        print(f"❌ Processing error: {e}")
        return 0, 0


def add_register_parser(subparsers):
    """Add register command parser"""
    parser = subparsers.add_parser(
        'register',
        help='Register search datasets to dataset_info.json'
    )
    parser.add_argument(
        '--metadata-file',
        required=True,
        help='Path to metadata file from search/synthesis results'
    )
    parser.add_argument(
        '--output-file',
        default='LLaMA-Factory/data/dataset_info.json',
        help='Path to output dataset_info.json (default: LLaMA-Factory/data/dataset_info.json)'
    )
    parser.add_argument(
        '--dataset-id-field',
        default='search_dataset_id',
        help='Field name for dataset ID in metadata (default: search_dataset_id)'
    )
    parser.add_argument(
        '--base-dir',
        choices=['search_dataset', 'synthesis'],
        default='search_dataset',
        help='Base directory name: search_dataset (for search results) or synthesis (for synthesis results)'
    )
    parser.add_argument(
        '--model',
        default='gemini',
        help='Model name for registration (default: gemini)'
    )
    parser.add_argument(
        '--template-type',
        choices=['generation', 'simple'],
        default='generation',
        help="Template type: 'generation' (original2search_model) or 'simple' (just dataset_id)"
    )
    parser.add_argument(
        '--original-dataset-field',
        default='original_dataset_id',
        help='Field name for original dataset ID (default: original_dataset_id)'
    )
    
    return parser


def handle_register(cli_manager, args):
    """Handle register command"""
    print("🔄 Dataset Registration Tool")
    print("=" * 60)
    
    # Validate input file exists
    if not os.path.exists(args.metadata_file):
        if hasattr(cli_manager, 'cli'):
            cli_manager.cli.print_error(f"Metadata file not found: {args.metadata_file}")
        else:
            print(f"❌ Metadata file not found: {args.metadata_file}")
        return 1
    
    # Show registration info
    print(f"📋 Metadata file: {args.metadata_file}")
    print(f"📁 Output file: {args.output_file}")
    print(f"🎯 Dataset ID field: {args.dataset_id_field}")
    print(f"📂 Base directory: {args.base_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"📋 Template type: {args.template_type}")
    print(f"🔗 Original dataset field: {args.original_dataset_field}")
    print("-" * 60)
    
    # Execute registration
    success, failed = register_datasets(
        metadata_file=args.metadata_file,
        output_file=args.output_file,
        dataset_id_field=args.dataset_id_field,
        base_dir=args.base_dir,
        model=args.model,
        template_type=args.template_type,
        original_dataset_field=args.original_dataset_field
    )
    
    print("-" * 60)
    print("📊 Registration Summary:")
    print(f"✅ Success: {success} datasets")
    print(f"❌ Failed: {failed} datasets")
    print(f"📁 Total: {success + failed} datasets")
    
    if success > 0:
        print(f"\n🎉 Successfully registered {success} search datasets!")
    
    return 0 if failed == 0 else 1