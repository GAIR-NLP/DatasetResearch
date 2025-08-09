#!/usr/bin/env python3
"""
a script to download datasets from Hugging Face
"""
import datasets
from datasets import load_dataset
import os
import json
import pandas as pd
from pathlib import Path
import argparse
import logging
from huggingface_hub import HfApi

# configure the logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_huggingface_cache():
    """
    clear the Hugging Face cache, solve the file corruption problem
    """
    try:
        from datasets import clear_cache
        import os
        import shutil
        
        logger.info("🧹 start to clear the Hugging Face cache...")
        
        # clear the datasets cache
        clear_cache()
        logger.info("✅ datasets cache cleared")
        
        # clear the huggingface_hub cache
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        if os.path.exists(cache_dir):
            try:
                shutil.rmtree(cache_dir)
                logger.info("✅ huggingface_hub cache cleared")
            except Exception as e:
                logger.warning(f"⚠️ failed to clear the huggingface_hub cache: {e}")
        
        logger.info("🎉 cache cleared")
        return True
        
    except Exception as e:
        logger.error(f"❌ failed to clear the cache: {e}")
        return False


def list_dataset_versions(dataset_id: str):
    """
    list all available versions of the dataset
    
    Args:
        dataset_id: dataset ID
        
    Returns:
        list: version list, sorted by time (latest first)
    """
    try:
        api = HfApi()
        repo_info = api.repo_info(repo_id=dataset_id, repo_type="dataset")
        
        # get all tags (versions)
        tags = repo_info.tags if hasattr(repo_info, 'tags') else []
        
        # get all branches
        refs = api.list_repo_refs(repo_id=dataset_id, repo_type="dataset")
        branches = [ref.name for ref in refs.branches] if refs.branches else []
        
        versions = []
        
        # add tags as versions
        for tag in tags:
            versions.append({
                'name': tag.name,
                'type': 'tag',
                'commit': tag.target_commit if hasattr(tag, 'target_commit') else None
            })
        
        # add branches as versions (if there are special branch names)
        special_branches = [b for b in branches if b not in ['main', 'master']]
        for branch in special_branches:
            versions.append({
                'name': branch,
                'type': 'branch',
                'commit': None
            })
        
        # sort by name, version number first
        def version_sort_key(v):
            name = v['name']
            # if the version number format (e.g. v1.0, v2.1, etc.), sort first
            if name.startswith('v') and any(c.isdigit() for c in name):
                return (0, name)
            return (1, name)
        
        versions.sort(key=version_sort_key, reverse=True)
        
        logger.info(f"📋 available versions of the dataset {dataset_id}:")
        for i, version in enumerate(versions):
            logger.info(f"   {i+1}. {version['name']} ({version['type']})")
        
        return versions
        
    except Exception as e:
        logger.warning(f"⚠️ failed to get the version information: {e}")
        return []


def get_latest_version(dataset_id: str):
    """
    get the latest version of the dataset
    
    Args:
        dataset_id: dataset ID
        
    Returns:
        str: latest version name, return None if there is no version
    """
    versions = list_dataset_versions(dataset_id)
    if versions:
        latest = versions[0]['name']
        logger.info(f"🎯 latest version: {latest}")
        return latest
    return None


def select_version_interactive(dataset_id: str):
    """
    interactive version selection
    
    Args:
        dataset_id: dataset ID
        
    Returns:
        str: selected version name
    """
    versions = list_dataset_versions(dataset_id)
    
    if not versions:
        logger.info("ℹ️ no specific version found, use the default version")
        return None
    
    print("\nplease select the version:")
    print("0. use the default version (no version specified)")
    for i, version in enumerate(versions):
        print(f"{i+1}. {version['name']} ({version['type']})")
    
    while True:
        try:
            choice = input(f"\nplease input the choice (0-{len(versions)}): ").strip()
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(versions):
                selected = versions[choice_num - 1]['name']
                logger.info(f"✅ selected version: {selected}")
                return selected
            else:
                print(f"❌ please input the number between 0 and {len(versions)}")
        except ValueError:
            print("❌ please input the valid number")
        except KeyboardInterrupt:
            print("\n\n💡 use the default version")
            return None


def download_dataset_with_datasets_library(dataset_id: str, output_dir: str = "downloaded_datasets", 
                                          subset: str = None, split: str = None, 
                                          version: str = None, auto_latest: bool = False,
                                          interactive_version: bool = False,
                                          trust_remote_code: bool = False,
                                          streaming: bool = True,  # default use streaming
                                          max_samples: int = 1000):  # default limit 1000 samples
    """
    use the datasets library to download the dataset
    
    Args:
        dataset_id: dataset ID
        output_dir: output directory
        subset: subset name
        split: data split
        version: version number
        auto_latest: whether to automatically use the latest version
        interactive_version: whether to interactively select the version
        trust_remote_code: whether to trust remote code (security related)
        streaming: whether to use streaming processing (default True, reduce memory usage)
        max_samples: maximum number of samples (default 1000, for memory limit)
    """
    import gc
    import psutil
    import os
    
    try:
        logger.info(f"🔽 start to download the dataset: {dataset_id}")
        
        # force use streaming mode
        streaming = True
        logger.info("🌊 force use streaming mode, reduce memory usage")
        
        # check the current memory usage
        try:
            memory_info = psutil.virtual_memory()
            logger.info(f"💾 current memory usage: {memory_info.percent}% ({memory_info.used / 1024**3:.1f}GB / {memory_info.total / 1024**3:.1f}GB)")
        except ImportError:
            logger.warning("⚠️ failed to check the memory usage")
        
        # security reminder
        if trust_remote_code:
            logger.warning("⚠️ use trust_remote_code, please ensure the dataset source is trusted")
        
        # version selection logic
        selected_version = None
        if interactive_version:
            selected_version = select_version_interactive(dataset_id)
        elif auto_latest:
            selected_version = get_latest_version(dataset_id)
        elif version:
            selected_version = version
        
        if selected_version:
            logger.info(f"🏷️ use version: {selected_version}")
        
        # create the output directory
        version_suffix = f"_{selected_version}" if selected_version else ""
        output_path = Path(output_dir) / f"{dataset_id.replace('/', '_')}{version_suffix}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        # download the dataset - force use streaming
        download_args = {
            "path": dataset_id,
            # "name": "default",
            "split": "train",
            "trust_remote_code": trust_remote_code,
            "streaming": True,  # force use streaming
            "token": "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"
        }
        
        if subset:
            download_args["name"] = subset
        if selected_version:
            download_args["revision"] = selected_version
        
        logger.info(f"🔧 download parameters: {download_args}")
        logger.info(f"📊 maximum number of samples: {max_samples}")
        
        # try to download, if failed and because of trust_remote_code problem, give a hint
        try:
            dataset = load_dataset(**download_args)
        except ValueError as e:
            if "trust_remote_code" in str(e):
                logger.error("❌ this dataset needs to execute custom code")
                logger.error("💡 solution: add --trust-remote-code parameter")
                logger.error("💡 or set trust_remote_code=True")
                logger.error("⚠️ please ensure the dataset source is trusted before use!")
                raise
            else:
                raise
        
        # save the dataset information
        logger.info(f"📊 dataset information:")
        if isinstance(dataset, dict):
            for split_name, split_data in dataset.items():
                logger.info(f"   - {split_name}: streaming dataset")
        else:
            logger.info(f"   - streaming dataset")
        
        # save the version information
        version_info = {
            "dataset_id": dataset_id,
            "version": selected_version,
            "subset": subset,
            "download_time": pd.Timestamp.now().isoformat(),
            "download_args": download_args,
            "streaming": True,  # record as True
            "max_samples": max_samples
        }
        
        version_file = output_path / "version_info.json"
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(version_info, f, indent=2, ensure_ascii=False)
        logger.info(f"📋 version information saved to: {version_file}")
        
        # save as different formats - only use streaming processing
        if isinstance(dataset, dict):
            # multiple splits
            for split_name, split_data in dataset.items():
                logger.info(f"📝 start to save the split: {split_name}")
                
                # force use streaming processing to save data
                _save_streaming_dataset(split_data, output_path, f"{split_name}", max_samples)
                
                # force garbage collection
                del split_data
                gc.collect()
        else:
            # single dataset
            logger.info(f"📝 start to save the dataset")
            
            # force use streaming processing
            _save_streaming_dataset(dataset, output_path, "dataset", max_samples)
            
            # force garbage collection
            del dataset
            gc.collect()
        
        logger.info(f"✅ dataset downloaded, saved to: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ download failed: {e}")
        # clear the memory
        gc.collect()
        return False


def _save_streaming_dataset(dataset, output_path: Path, filename: str, max_samples: int = 1000):
    """
    save the streaming dataset (memory friendly)
    """
    import json
    import random
    from collections import deque
    
    logger.info(f"🌊 use streaming way to save: {filename}")
    logger.info(f"📊 sample number limit: {max_samples}")
    
    jsonl_file = output_path / f"{filename}.jsonl"
    json_file = output_path / f"{filename}.json"
    
    # reservoir sampling algorithm for random sampling
    def reservoir_sampling(stream, k):
        reservoir = []
        for i, item in enumerate(stream):
            if len(reservoir) < k:
                reservoir.append(item)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = item
            if (i + 1) % 1000 == 0:
                logger.info(f"   processed {i + 1} records")
        return reservoir
    
    # save as JSONL
    try:
        # first calculate the dataset size (if possible)
        total_count = None
        try:
            total_count = len(dataset)
            logger.info(f"📊 dataset total size: {total_count}")
        except:
            logger.info("⚠️ failed to get the dataset total size, will count during processing")
        
        # if the total size is known and need to sample, use reservoir sampling
        if total_count and total_count > max_samples:
            logger.info(f"🎲 use reservoir sampling algorithm to randomly select {max_samples} records")
            sampled_data = reservoir_sampling(dataset, max_samples)
            
            # save the sampled data
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for item in sampled_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            logger.info(f"💾 saved {len(sampled_data)} random sampled records to: {jsonl_file}")
        else:
            # regular streaming processing, strictly follow the max_samples limit
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                count = 0
                for example in dataset:
                    if count >= max_samples:
                        logger.info(f"⏹️ reached the maximum sample number limit: {max_samples}")
                        break
                    
                    json.dump(example, f, ensure_ascii=False)
                    f.write('\n')
                    count += 1
                    
                    # 每100条记录显示进度
                    if count % 100 == 0:
                        logger.info(f"   processed {count}/{max_samples} records")
            
            logger.info(f"💾 saved {count} records to: {jsonl_file}")
    
    except Exception as e:
        logger.warning(f"⚠️ streaming processing interrupted: {e}")
        return
    
    # save as JSON format (compatibility)
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f_in:
            with open(json_file, 'w', encoding='utf-8') as f_out:
                data = []
                for line in f_in:
                    if line.strip():
                        data.append(json.loads(line))
                json.dump(data, f_out, ensure_ascii=False, indent=2)
        logger.info(f"💾 saved JSON format to: {json_file}")
    except Exception as e:
        logger.warning(f"⚠️ failed to save JSON format: {e}")
    
    # try to save as CSV (may fail because of the complex data structure)
    try:
        import pandas as pd
        
        # read the saved JSONL file to generate CSV
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            data = []
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        if data:
            df = pd.DataFrame(data)
            csv_file = output_path / f"{filename}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"💾 saved CSV format to: {csv_file}")
        
    except Exception as e:
        logger.warning(f"⚠️ failed to save as CSV format: {e}")


def _save_regular_dataset(dataset, output_path: Path, filename: str, max_samples: int = None):
    """
    save the regular dataset
    """
    logger.info(f"📝 save the regular dataset: {filename}")
    
    # limit the sample number
    if max_samples and hasattr(dataset, '__len__'):
        try:
            actual_length = len(dataset)
            if actual_length > max_samples:
                logger.info(f"🔢 dataset size ({actual_length}) exceeds the limit ({max_samples}), use random sampling")
                # use random sampling instead of simple truncation
                import random
                indices = random.sample(range(actual_length), max_samples)
                indices.sort()  # keep the order, reduce memory fragmentation
                dataset = dataset.select(indices)
                logger.info(f"✂️ sampled to {max_samples} records")
            else:
                logger.info(f"✅ dataset size ({actual_length}) is within the limit ({max_samples}), no need to sample")
        except Exception as e:
            logger.warning(f"⚠️ failed to sample: {e}")
    
    jsonl_file = output_path / f"{filename}.jsonl"
    json_file = output_path / f"{filename}.json"
    
    # save as JSONL
    try:
        dataset.to_json(jsonl_file, orient='records', lines=True, force_ascii=False)
        logger.info(f"💾 saved JSONL to: {jsonl_file}")
    except Exception as e:
        logger.error(f"❌ failed to save JSONL: {e}")
    
    # save as JSON
    try:
        dataset.to_json(json_file, orient='records', force_ascii=False)
        logger.info(f"💾 saved JSON to: {json_file}")
    except Exception as e:
        logger.error(f"❌ failed to save JSON: {e}")
    
    # save as CSV
    try:
        csv_file = output_path / f"{filename}.csv"
        dataset.to_csv(csv_file, index=False)
        logger.info(f"💾 saved CSV to: {csv_file}")
    except Exception as e:
        logger.warning(f"⚠️ failed to save as CSV format: {e}")


def download_with_git_lfs(dataset_id: str, output_dir: str = "downloaded_datasets"):
    """
    use git lfs to download the dataset (suitable for large files)
    
    Args:
        dataset_id: dataset ID
        output_dir: output directory
    """
    import subprocess
    
    try:
        logger.info(f"🔽 use git lfs to download: {dataset_id}")
        
        # create the output directory
        output_path = Path(output_dir) / dataset_id.replace('/', '_')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # build the git clone command
        repo_url = f"https://huggingface.co/datasets/{dataset_id}"
        
        # clone the repository
        cmd = ["git", "clone", repo_url, str(output_path)]
        logger.info(f"🔧 execute the command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=output_dir)
        
        if result.returncode == 0:
            logger.info(f"✅ git clone successfully, data saved to: {output_path}")
            return True
        else:
            logger.error(f"❌ git clone failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ download failed: {e}")
        return False


def download_specific_files(dataset_id: str, file_patterns: list, output_dir: str = "downloaded_datasets"):
    """
    download specific files
    
    Args:
        dataset_id: dataset ID
        file_patterns: file pattern list, like ["*.json", "*.parquet"]
        output_dir: output directory
    """
    from huggingface_hub import snapshot_download
    
    try:
        logger.info(f"🔽 download specific files: {dataset_id}")
        logger.info(f"📁 file pattern: {file_patterns}")
        
        # create the output directory
        output_path = Path(output_dir) / dataset_id.replace('/', '_')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # download specific files
        downloaded_path = snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            local_dir=str(output_path),
            allow_patterns=file_patterns
        )
        
        logger.info(f"✅ file downloaded, saved to: {downloaded_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ download failed: {e}")
        return False


def main():
    """main function"""
    parser = argparse.ArgumentParser(description='download dataset from Hugging Face')
    parser.add_argument('--id', help='dataset ID (e.g. bigcode/bigcodebench)')
    parser.add_argument('--output_dir', '-o', default='batch_downloaded_datasets', 
                       help='output directory (default: downloaded_datasets)')
    parser.add_argument('--method', '-m', choices=['datasets', 'git', 'files'], 
                       default='datasets', help='download method')
    parser.add_argument('--subset', '-s', help='subset name')
    parser.add_argument('--split', help='data split (train/test/validation)')
    parser.add_argument('--file_patterns', nargs='+', default=['*.json', '*.parquet'], 
                       help='file pattern (only used for files method)')
    
    # version related parameters
    parser.add_argument('--version', '-v', help='specify specific version (e.g. v1.0, v2.1)')
    parser.add_argument('--latest', action='store_true', 
                       help='automatically use the latest version')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='interactive version selection')
    parser.add_argument('--list-versions', action='store_true',
                       help='list all available versions, not download')
    
    # security related parameters
    parser.add_argument('--trust-remote-code', action='store_true',
                       help='trust and execute remote code (⚠️ security risk)')
    
    # new: sample number limit parameter
    parser.add_argument('--max-samples', type=int, default=1000,
                       help='maximum sample number limit (default: 1000)')
    
    # new: clear cache option
    parser.add_argument('--clear-cache', action='store_true',
                       help='clear Hugging Face cache before download')
    
    args = parser.parse_args()
    
    if not args.id:
        parser.print_help()
        print("\n❌ please specify the dataset ID")
        return
    
    # 如果指定了清理缓存，先执行清理
    if args.clear_cache:
        logger.info("🧹 user request to clear Hugging Face cache...")
        if clear_huggingface_cache():
            logger.info("✅ huggingface cache cleared")
        else:
            logger.info("❌ failed to clear huggingface cache, but the program will continue to execute")
    
    # security warning
    if args.trust_remote_code:
        print("⚠️" * 20)
        print("🚨 security warning: you are about to enable trust_remote_code")
        print("🚨 this will allow executing arbitrary Python code from the dataset")
        print("🚨 malicious code may harm your system")
        print("⚠️" * 20)
        
        confirm = input("continue? (input 'yes' to continue): ").strip().lower()
        if confirm != 'yes':
            print("💡 operation cancelled")
            return
    
    # if only list the versions
    if args.list_versions:
        logger.info(f"🔍 query the version information of the dataset {args.id}...")
        versions = list_dataset_versions(args.id)
        if not versions:
            logger.info("ℹ️ the dataset has no specific version")
        return
    
    logger.info(f"🎯 dataset ID: {args.id}")
    logger.info(f"📂 output directory: {args.output_dir}")
    logger.info(f"🔧 download method: {args.method}")
    logger.info(f"📊 maximum sample number: {args.max_samples}")
    
    success = False
    
    if args.method == 'datasets':
        success = download_dataset_with_datasets_library(
            dataset_id=args.id,
            output_dir=args.output_dir,
            subset=args.subset,
            split=args.split,
            version=args.version,
            auto_latest=args.latest,
            interactive_version=args.interactive,
            trust_remote_code=args.trust_remote_code,
            max_samples=args.max_samples  # pass the maximum sample number parameter
        )
    elif args.method == 'git':
        success = download_with_git_lfs(args.id, args.output_dir)
    elif args.method == 'files':
        success = download_specific_files(
            args.id, args.file_patterns, args.output_dir
        )
    
    if success:
        logger.info("🎉 download completed!")
    else:
        logger.error("💥 download failed!")


# example usage
def example_usage():
    """example usage"""
    
    # example 1: download the latest version, limit 1000 samples
    print("=" * 50)
    print("example 1: download the latest version, limit 1000 samples")
    download_dataset_with_datasets_library("squad", auto_latest=True, max_samples=1000)
    
    # example 2: download the specified version, limit 500 samples
    print("\n" + "=" * 50)
    print("example 2: download the specified version, limit 500 samples")
    download_dataset_with_datasets_library("squad", version="v2.0", max_samples=500)
    
    # example 3: interactive version selection, limit 2000 samples
    print("\n" + "=" * 50)
    print("example 3: interactive version selection, limit 2000 samples")
    download_dataset_with_datasets_library("squad", interactive_version=True, max_samples=2000)
    
    # example 4: list all versions
    print("\n" + "=" * 50)
    print("example 4: list all versions")
    versions = list_dataset_versions("squad")
    for version in versions:
        print(f"- {version['name']} ({version['type']})")
    
    # example 5: use command line parameters
    print("\n" + "=" * 50)
    print("example 5: use command line parameters")
    print("python huggingface_download.py --id squad --max-samples 1000")
    print("python huggingface_download.py --id squad --max-samples 500 --trust-remote-code")
    print("python huggingface_download.py --id squad --max-samples 2000 --latest")
    print("python huggingface_download.py --id squad --max-samples 1000 --clear-cache")


if __name__ == "__main__":
    main()