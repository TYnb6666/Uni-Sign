import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from path_config import DataConfig
from data_loader_multigraph import SignLanguageMultiGraphDataset, collate_fn_multigraph
from models import Uni_Sign
from data_loader_rotation import build_vocab
import pickle
import os

class MockArgs:
    def __init__(self):
        self.hidden_dim = 256
        self.dataset = 'CSL-Daily'
        self.rgb_support = False # Should be false now
        self.label_smoothing = 0.1

def verify_pipeline():
    print("--- Verifying Pipeline ---")
    
    # 1. Setup Mappings/Vocab (Mock or Load)
    # We need to make sure we can load the split file and vocab
    # split_1.txt path: D:\Datasets\sentence_label\split_1.txt
    # data_loader expects mappings dict: mappings['train'] = {sample_id: info, ...}
    
    print("Loading mappings...")
    # Mocking mapping for one sample we know exists
    sample_id = "S000048_P0000_T00"
    mappings = {
        'train': {
            sample_id: {
                'group': 'default', # Config logic might need adjustment if logic relies on folder structure
                'gloss': ['test', 'gloss'],
                'data_path': 'mock_path' 
            }
        }
    }
    
    # Needs vocab
    vocab = {'<sos>': 0, '<eos>': 1, '<unk>': 2, 'test': 3, 'gloss': 4, '<pad>': 5}
    
    # 2. Instantiate Dataset
    print("Instantiating Dataset...")
    dataset = SignLanguageMultiGraphDataset(split='train', mappings=mappings, vocab=vocab, cache_data=False)
    
    # 3. Load One Sample
    print(f"Loading sample {sample_id}...")
    sample = dataset[0]
    
    print("Sample Loaded keys:", sample.keys())
    print("Left Shape:", sample['left'].shape)
    print("Right Shape:", sample['right'].shape)
    print("Body Shape:", sample['body'].shape)
    print("Face Shape:", sample['face'].shape)
    
    # Verify Dimensions (should be 4)
    assert sample['left'].shape[-1] == 4, f"Left hand last dim should be 4, got {sample['left'].shape[-1]}"
    assert sample['right'].shape[-1] == 4, f"Right hand last dim should be 4, got {sample['right'].shape[-1]}"
    assert sample['body'].shape[-1] == 4, f"Body last dim should be 4, got {sample['body'].shape[-1]}"
    assert sample['face'].shape[-1] == 4, f"Face last dim should be 4, got {sample['face'].shape[-1]}"
    
    # 4. Instantiate Model
    print("Instantiating Model...")
    args = MockArgs()
    model = Uni_Sign(args)
    # Move to CPU for test
    model = model.to('cpu')
    
    # 5. Forward Pass
    print("Running Forward Pass...")
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn_multigraph)
    batch = next(iter(loader))
    src_input, tgt_input = batch
    
    # Models often check lengths for masking
    output = model(src_input, tgt_input)
    
    print("Forward Pass Successful.")
    print("Loss:", output['lines'] if 'loss' in output else output.keys())
    print("Output shapes verified.")

if __name__ == "__main__":
    verify_pipeline()
