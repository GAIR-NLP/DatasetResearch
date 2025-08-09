"""filter all datasets on huggingface
initial filtering (python filter.py)
function description:
first part: generate dataset_2.txt, containing 422 datasets
filtering: gated dataset and the following six task types:
question-answering, summarization, text-classification, text-generation, multiple-choice, translation

second part: generate dataset_3.txt, containing 261 datasets
filtering: README contains more than 1000 characters
"""

from huggingface_hub import list_datasets
from huggingface_hub import dataset_info
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm
import pdb
import itertools
from datetime import datetime

file_name = 'dataset_4.txt'
readme_threshold = 1000
only_filter_readme = False

# if only_filter_readme is True, then the format of dataset_2.txt is needed, and the final format of dataset_3.txt will be generated
# if only_filter_readme is False, then the blank file is enough

def has_readme_file(dataset_id):
    """
    check if the dataset has a README.md file and the content is not empty
    
    Args:
        dataset_id (str): dataset ID
    
    Returns:
        bool: True if the dataset has a README.md file and the content is not empty, otherwise False
    """
    try:
        # method 1: check if the README.md file exists and the content is not empty
        try:
            files = list_repo_files(repo_id=dataset_id, repo_type="dataset")
            if "README.md" in files:
                # if the README.md file exists, download and check if the content is empty
                try:
                    readme_path = hf_hub_download(
                        repo_id=dataset_id,
                        filename="README.md",
                        repo_type="dataset"
                    )
                    
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    # check if the content is empty or only contains whitespace characters
                    if content and len(content) > readme_threshold:  # at least 10 characters
                        return True
                    else:
                        print(f"The README.md file of dataset {dataset_id} exists but the content is empty")
                        return False
                        
                except Exception as e:
                    print(f"Failed to download or read the README.md file of dataset {dataset_id}: {e}")
                    # if the download fails, continue to check the description information
                    pass
        except Exception:
            pass
        
        # method 2: check if the dataset information has a valid description
        try:
            info = dataset_info(dataset_id)
            if hasattr(info, 'description') and info.description and len(info.description.strip()) > 50:  # at least 50 characters
                return True
        except Exception:
            pass
            
        return False
    except Exception:
        return False


if not only_filter_readme:

    # Limit to first 100000 items directly on generator, without converting to list
    total = 1000000
    datasets = itertools.islice(list_datasets(), total)
    # pdb.set_trace()

    # Collect all unique task types
    unique_task_types = set()

    # Open file for writing (will be created automatically if it doesn't exist)
    with open(file_name, 'w', encoding='utf-8') as f:
        for dataset in tqdm(datasets, desc="Processing datasets", total=total):
            # pdb.set_trace()
            # First check if tags exist
            if not (hasattr(dataset, 'tags') and dataset.tags):
                continue

            # # Check if creation time is 2020 or later
            # if hasattr(dataset, 'created_at') and dataset.created_at:
            #     if dataset.created_at.year < 2020:
            #         continue
            
            # # Check if downloads are greater than or equal to 100
            # if hasattr(dataset, 'downloads') and dataset.downloads is not None:
            #     if dataset.downloads < 100:
            #         continue

            # print(dataset_info(dataset.id))
            # pdb.set_trace()
                
            # Get all modality-related tags
            modality_tags = [tag for tag in dataset.tags if tag.startswith('modality:')]
            # Continue only when modality_tags contains only 'modality:text'
            if modality_tags != ['modality:text']:
                continue
                
            # # Get all language-related tags
            # language_tags = [tag for tag in dataset.tags if tag.startswith('language:')]
            # # Continue only when language_tags contains only 'language:en'
            # if language_tags != ['language:en']:
            #     continue
                
            # Get all task_categories-related tags
            task_category_tags = [tag for tag in dataset.tags if tag.startswith('task_categories:')]
            # Must contain task_categories tags
            if not task_category_tags:
                continue
                
            # Only select datasets with exactly one task category
            if len(task_category_tags) != 1:
                continue
                
            # Only keep datasets with text-generation or summarization task categories
            allowed_task_categories = ['task_categories:text-generation', 'task_categories:summarization', 'task_categories:question-answering', 'task_categories:text-classification', 'task_categories:multiple-choice', 'task_categories:translation']
            if task_category_tags[0] not in allowed_task_categories:
                continue
                
            # Check if dataset requires access approval (gated)
            if getattr(dataset, 'gated', False):
                # pdb.set_trace()
                print(f"{dataset.id} requires access approval and contains text data")
                # Extract task types (remove 'task_categories:' prefix)
                task_types = [tag.replace('task_categories:', '') for tag in task_category_tags]
                # Add task types to set
                unique_task_types.update(task_types)
                # Write dataset name and task types to file
                f.write(f"{dataset.id} | Task types: {', '.join(task_types)}\n")
                # Immediately flush buffer to ensure data is written to file
                f.flush()

        # Print all unique task types at the end of file
        f.write(f"\nAll unique task types: {', '.join(sorted(unique_task_types))}\n")

    print(f"initial filtering completed, generated {file_name} file")

# read the play_3.txt file and perform readme filtering
print("start readme filtering...")

# read the original file content
with open(file_name, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# separate dataset lines and stats lines
dataset_lines = []
stats_lines = []
for line in lines:
    if line.strip() and not line.startswith("All unique task types:"):
        dataset_lines.append(line.strip())
    elif line.startswith("All unique task types:"):
        stats_lines.append(line)

print(f"before readme filtering: {len(dataset_lines)} datasets")

# filter datasets with readme and non-empty content
filtered_lines = []
empty_readme_count = 0
no_readme_count = 0

for line in tqdm(dataset_lines, desc="check readme file"):
    # extract dataset ID (format: dataset_id | Task types: xxx)
    dataset_id = line.split(' | ')[0]
    
    if has_readme_file(dataset_id):
        filtered_lines.append(line)
        print(f"keep dataset {dataset_id}: has valid readme content")
    else:
        # further distinguish between no readme or empty readme
        try:
            files = list_repo_files(repo_id=dataset_id, repo_type="dataset")
            if "README.md" in files:
                empty_readme_count += 1
                print(f"remove dataset {dataset_id}: README.md exists but content is empty")
            else:
                no_readme_count += 1
                print(f"remove dataset {dataset_id}: no README.md file")
        except:
            no_readme_count += 1
            print(f"remove dataset {dataset_id}: cannot access or no readme file")

print(f"\nafter readme filtering: {len(filtered_lines)} datasets")
print(f"removed {len(dataset_lines) - len(filtered_lines)} datasets")
print(f"where:")
print(f"  - README.md exists but content is empty: {empty_readme_count} datasets")
print(f"  - no README.md file: {no_readme_count} datasets")

# rewrite the file, only keep datasets with readme and non-empty content
with open(file_name, 'w', encoding='utf-8') as f:
    # write the filtered datasets
    for line in filtered_lines:
        f.write(line + '\n')
    
    # 写入统计信息
    for line in stats_lines:
        f.write(line)
    
    # add readme filtering statistics
    f.write(f"\nreadme filtering statistics:\n")
    f.write(f"before filtering: {len(dataset_lines)} datasets\n")
    f.write(f"after filtering: {len(filtered_lines)} datasets\n")
    f.write(f"removed: {len(dataset_lines) - len(filtered_lines)} datasets\n")
    f.write(f"README.md exists but content is empty: {empty_readme_count}\n")
    f.write(f"no README.md file: {no_readme_count}\n")

print(f"readme filtering completed, updated {file_name} file")