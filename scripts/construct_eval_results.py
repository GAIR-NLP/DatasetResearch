import json

# with open('datasets/test_set_metadata.json', 'r') as f:
#     test_set_metadata = json.load(f)

# with open('datasets/search_set_metadata_gpt-4o-mini-search-preview.json', 'r') as f:
#     search_set_metadata = json.load(f)

with open('datasets/metadata_paperwithcode.json', 'r') as f:
    test_set_metadata = json.load(f)

with open('datasets/generation_metadata_wo.json', 'r') as f:
    search_set_metadata = json.load(f)

final_metadatas = []

for sample in test_set_metadata:
    # first determine task_id
    final_metadata = {}
    find_flag = False
    final_metadata['task_id'] = sample['task_id']
    final_metadata['original_dataset_id'] = sample['dataset_id']
    final_metadata['original_metadata'] = sample['metadata']
    for search_sample in search_set_metadata:
        if search_sample['task_id'] == sample['task_id']:
            final_metadata['search_dataset_id'] = search_sample['search_dataset_id']
            final_metadata['search_metadata'] = search_sample['search_metadata']
            find_flag = True
            break
    if not find_flag:
        final_metadata['search_dataset_id'] = 'Generation Failed'
        final_metadata['search_metadata'] = 'Generation Failed'
    final_metadatas.append(final_metadata)

    # final_metadata['task_id'] = sample['dataset_id'].replace('/', '_')
    # final_metadata['original_dataset_id'] = sample['dataset_id']
    # metadata = json.loads(sample['metadata'])
    # # find corresponding search_dataset_id from search_set_metadata
    # for search_sample in search_set_metadata:
    #     if search_sample['original_dataset_id'] == sample['dataset_id']:
    #         final_metadata['search_dataset_id'] = search_sample['search_dataset_id']
    #         final_metadata['original_metadata'] = search_sample['original_metadata']
    #         final_metadata['search_metadata'] = search_sample['search_metadata']
    #         find_flag = True
    #         break
    # if not find_flag:
    #     final_metadata['original_metadata'] = metadata
    #     final_metadata['search_dataset_id'] = 'Search Failed'
    #     final_metadata['search_metadata'] = 'Search Failed'
    # final_metadatas.append(final_metadata)

with open('datasets/results/generation_wo2.json', 'w') as f:
    json.dump(final_metadatas, f, indent=4, ensure_ascii=False)
