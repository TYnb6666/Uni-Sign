import pickle
import os
from path_config import DataConfig

pkl_path = os.path.join(DataConfig.LABEL_ROOT, 'csl2020ct_v2.pkl')
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print(f"Type: {type(data)}")
print(f"Keys: {list(data.keys())}")

for key in data.keys():
    val = data[key]
    print(f"\n--- Key: '{key}' ---")
    print(f"  Type: {type(val)}")
    if isinstance(val, list):
        print(f"  Length: {len(val)}")
        if len(val) > 0:
            print(f"  First item type: {type(val[0])}")
            if isinstance(val[0], dict):
                print(f"  First item keys: {list(val[0].keys())}")
                print(f"  First item: {val[0]}")
            else:
                print(f"  First item: {val[0]}")
    elif isinstance(val, dict):
        print(f"  Length: {len(val)}")
        first_k = list(val.keys())[:3]
        print(f"  First 3 keys: {first_k}")
        if first_k:
            print(f"  Value of first key: {val[first_k[0]]}")
    elif isinstance(val, str):
        print(f"  Value: {val[:200]}")
    else:
        print(f"  Value: {val}")
