import os
import pickle
from path_config import DataConfig

def test_loading():
    label_root = DataConfig.LABEL_ROOT
    print(f"LABEL_ROOT = {label_root}")
    print(f"LABEL_ROOT exists? {os.path.exists(label_root)}")
    
    # Step 1: Check pkl file
    pkl_path = os.path.join(label_root, 'csl2020ct_v2.pkl')
    print(f"\n--- Step 1: Check PKL ---")
    print(f"pkl_path = {pkl_path}")
    print(f"pkl exists? {os.path.exists(pkl_path)}")
    
    csl_labels = {}
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        print(f"pkl type = {type(data)}")
        if isinstance(data, list):
            print(f"pkl list length = {len(data)}")
            if len(data) > 0:
                print(f"First item keys = {list(data[0].keys())}")
                print(f"First item name = {data[0].get('name', 'NO NAME KEY')}")
                for item in data:
                    csl_labels[item['name']] = item
        elif isinstance(data, dict):
            print(f"pkl dict length = {len(data)}")
            first_key = list(data.keys())[0] if data else None
            print(f"First key = {first_key}")
            csl_labels = data
        print(f"Total labels loaded: {len(csl_labels)}")
    else:
        print("PKL FILE NOT FOUND!")
    
    # Step 2: Check split file
    split_path = os.path.join(label_root, 'split_1.txt')
    print(f"\n--- Step 2: Check Split ---")
    print(f"split_path = {split_path}")
    print(f"split exists? {os.path.exists(split_path)}")
    
    split_map = {}
    if os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print(f"Total lines in split_1.txt = {len(lines)}")
        print(f"First 5 lines:")
        for line in lines[:5]:
            print(f"  '{line.strip()}'")
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('name|'):
                continue
            parts = line.split('|')
            if len(parts) >= 2:
                split_map[parts[0]] = parts[1]
        print(f"Total entries in split_map: {len(split_map)}")
        # Show distribution
        from collections import Counter
        split_counts = Counter(split_map.values())
        print(f"Split distribution: {dict(split_counts)}")
    else:
        print("SPLIT FILE NOT FOUND!")
    
    # Step 3: Cross-reference
    print(f"\n--- Step 3: Cross-reference ---")
    matched = 0
    unmatched_pkl = 0
    for vid_name in csl_labels:
        if vid_name in split_map:
            matched += 1
        else:
            unmatched_pkl += 1
    print(f"Labels in pkl: {len(csl_labels)}")
    print(f"Entries in split_map: {len(split_map)}")
    print(f"Matched (in both): {matched}")
    print(f"In pkl but not in split: {unmatched_pkl}")
    
    # Step 4: Test the actual function
    print(f"\n--- Step 4: Test load_label_mappings ---")
    from data_loader_rotation import load_label_mappings
    mappings = load_label_mappings()
    for split in ['train', 'dev', 'test']:
        count = len(mappings.get(split, {}))
        print(f"Split: {split}, Count: {count}")
    
    # Check a sample
    sample_id = "S000048_P0000_T00"
    for split in mappings:
        if sample_id in mappings[split]:
            print(f"\nSample {sample_id} found in '{split}'")
            print(f"Info: {mappings[split][sample_id]}")
            break
    else:
        print(f"\nWarning: Sample {sample_id} not found in any split.")
        # Check if it's in csl_labels
        if sample_id in csl_labels:
            print(f"  BUT it IS in csl_labels with keys: {list(csl_labels[sample_id].keys())}")
            if sample_id in split_map:
                print(f"  AND it IS in split_map with split: {split_map[sample_id]}")
            else:
                print(f"  BUT it is NOT in split_map")
        else:
            print(f"  AND it is NOT in csl_labels either")

if __name__ == "__main__":
    test_loading()
