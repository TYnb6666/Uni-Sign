# utils/data_loader_multigraph.py
"""
Data Loader for Multi-Graph ST-GCN.

Loads raw 3D coordinates only (no derived features).
Outputs dict with keys: 'left', 'right', 'body', 'face'
Each is (T, V, 3) where V is number of nodes for that modality.
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from path_config import DataConfig

# Import existing utilities
from data_loader_rotation import load_label_mappings, build_vocab

# Index mappings
# Index mappings (PKL -> MediaPipe)
# Pose Mapping (9 nodes used in graph)
POSE_MAPPING = {
    0: 1,   # Left Eye (Inner) - Used as head root in graph
    1: 7,   # Left Ear
    2: 8,   # Right Ear
    3: 11,  # Left Shoulder
    4: 12,  # Right Shoulder
    5: 13,  # Left Elbow
    6: 14,  # Right Elbow
    7: 15,  # Left Wrist
    8: 16,  # Right Wrist
}

# Face Mapping (18 nodes used in graph, excludes 33 and 263)
FACE_MAPPING = {
    0: 1,   # Nose Tip
    # 1: 33,  # Excluded
    2: 152, # Chin Bottom
    # 3: 263, # Excluded
    4: 13,  # Upper Lip Center
    5: 14,  # Lower Lip Center
    6: 58,  # Right Cheek
    7: 78,  # Left Mouth Corner
    8: 81,  # Upper Lip Left
    9: 136, # Left Cheek
    10: 149, # Lower Lip Left
    11: 178, # Lower Lip Left Edge
    12: 234, # Left Face Edge
    13: 288, # Right Cheek
    14: 308, # Right Mouth Corner
    15: 311, # Upper Lip Right
    16: 365, # Right Cheek
    17: 378, # Lower Lip Right
    18: 402, # Lower Lip Right Edge
    19: 454, # Right Face Edge
}


def crop_scale_3d(motion, thr=0.3):
    """
    Normalize 3D motion to [-1, 1] using dynamic bounding box.
    
    Args:
        motion: (T, N, 4) where last dim is [x, y, z, confidence]
        thr: confidence threshold
        
    Returns:
        result: (T, N, 4) normalized coordinates + confidence
        scale: float, the scale factor used
    """
    result = copy.deepcopy(motion)
    
    # Find valid coordinates (confidence > threshold)
    valid_mask = motion[..., 3] > thr
    valid_coords = motion[valid_mask][:, :3]
    
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0.0
    
    # Compute bounding box
    xmin, ymin, zmin = valid_coords.min(axis=0)
    xmax, ymax, zmax = valid_coords.max(axis=0)
    
    # Scale is max extent
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin)
    
    if scale == 0:
        return np.zeros(motion.shape), 0.0
    
    # Center
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    
    # Normalize to [-1, 1]
    result_3d = (motion[..., :3] - np.array([cx, cy, cz])) / scale * 2
    result_3d = np.clip(result_3d, -1, 1)
    
    # Combine with original confidence
    result[..., :3] = result_3d
    
    # Mask invalid points (optional, but keep consistent with previous logic)
    # Note: Previous logic zeroed out invalid points. 
    # Here we keep 0s where invalid, but now we have 4 channels.
    # If confidence is low, the normalized coords might be noisy but valid_mask handles it downstream if needed.
    # For now, let's zero out xyz if invalid, but keep confidence? 
    # actually previous code did: result_3d[~valid_mask] = 0
    result[~valid_mask, :3] = 0
    
    return result, scale


def load_pickle_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_multigraph_data(sample_id, info, split='train'):
    """
    Load multimodal data for a single sample from pickle.
    
    Args:
        sample_id: Sample identifier
        info: Dict with 'data_path', 'group', 'gloss'
        split: Dataset split ('train', 'dev', 'test')
        
    Returns:
        Dict with keys 'left', 'right', 'body', 'face', each (T, V, 4)
    """
    group = info.get('group', 'default') # Adjust if 'group' is not present in new info
    
    # All data is in one pickle file, so we can just use one path getter
    # or just construct it directly. Using get_hand_data_path as generic getter.
    pkl_path = DataConfig.get_hand_data_path(split, group, sample_id)
    
    if not os.path.exists(pkl_path):
        return None
        
    data = load_pickle_data(pkl_path)
    
    # All modalities should have same sequence length T
    # We can check 'pose' for T
    if 'pose' not in data: 
         return None
         
    pose_raw = np.array(data['pose']) # (T, 17, 4)
    T = pose_raw.shape[0]

    # === Process Pose ===
    # Map PKL indices to MediaPipe sorted indices
    # We need 17 nodes. MP indices are just for reference/graph construction.
    # We should arrange them in a fixed order. 
    # Let's assume the graph strategy uses the order defined in POSE_MAPPING values sorted.
    # Wait, the Graph class usually expects a fixed number of nodes 0..V-1.
    # The 'indices' in previous code were picking specific columns.
    # Here we have 17 points. We should just return them as (T, 17, 4).
    # BUT, we need to ensure they match the adjacency matrix order if the graph relies on specific MP IDs.
    # The graph usually relies on 0..N indices. If we provide 17 points, the graph should be built for 17 points.
    # If the user wants "MediaPipe IDs", it implies the Graph definition uses MP IDs.
    # Let's sort keys of POSE_MAPPING to get consistent ordering if needed?
    # Or just return the 17 points re-ordered by MP ID? 
    # Let's sort by MediaPipe ID to be safe and consistent.
    sorted_pose_mp_ids = sorted(POSE_MAPPING.values())
    mp_to_pkl_pose = {v: k for k, v in POSE_MAPPING.items()}
    
    body = np.zeros((T, 9, 4))
    for i, mp_idx in enumerate(sorted_pose_mp_ids):
        pkl_idx = mp_to_pkl_pose[mp_idx]
        body[:, i] = pose_raw[:, pkl_idx]
        
    # Scale Pose
    # Use crop_scale_3d on the body data
    # Note: crop_scale_3d expects (T, N, 4) and returns (T, N, 4)
    body, _ = crop_scale_3d(body, thr=0.3)

    # === Process Left Hand ===
    left_hand = np.zeros((T, 21, 4))
    if 'left_hand' in data:
        lh_raw = np.array(data['left_hand']) # (T, 21, 7)
        if lh_raw.shape[0] > 0:
            # Slice first 4 channels: x, y, z, conf
            left_hand = lh_raw[..., :4]
            T_hand = left_hand.shape[0]
            # Handle length mismatch
            min_T = min(T, T_hand)
            left_hand = left_hand[:min_T]
            # If hand is shorter? assume aligned for now.
    
    # Center Left Hand at Wrist (Index 0)
    # Wrist is index 0 in MediaPipe Hand
    for t in range(T):
        if np.any(left_hand[t, :, 3] > 0): # Check confidence > 0
            wrist = left_hand[t, 0, :3]
            left_hand[t, :, :3] = left_hand[t, :, :3] - wrist
            
    # Scale Left Hand
    # Calc scale from valid points
    l_mask = left_hand[..., 3] > 0
    if np.any(l_mask):
         l_max = np.max(np.abs(left_hand[..., :3][l_mask])) # 手部动作在空间中伸展的最大范围
         l_scale = max(l_max, 1e-6)
         left_hand[..., :3] /= l_scale

    # === Process Right Hand ===
    right_hand = np.zeros((T, 21, 4))
    if 'right_hand' in data:
        rh_raw = np.array(data['right_hand']) # (T, 21, 7)
        if rh_raw.shape[0] > 0:
            right_hand = rh_raw[..., :4]
            min_T = min(T, right_hand.shape[0])
            right_hand = right_hand[:min_T]

    # Center Right Hand at Wrist
    for t in range(T):
        if np.any(right_hand[t, :, 3] > 0):
            wrist = right_hand[t, 0, :3]
            right_hand[t, :, :3] = right_hand[t, :, :3] - wrist
            
    # Scale Right Hand
    r_mask = right_hand[..., 3] > 0
    if np.any(r_mask):
         r_max = np.max(np.abs(right_hand[..., :3][r_mask]))
         r_scale = max(r_max, 1e-6)
         right_hand[..., :3] /= r_scale

    # === Process Face ===
    face = np.zeros((T, 18, 4))
    if 'face' in data:
        face_raw = np.array(data['face']) # (T, 20, 3)
        # Append confidence 1.0
        conf = np.ones((face_raw.shape[0], face_raw.shape[1], 1))
        face_4ch = np.concatenate([face_raw, conf], axis=-1)
        
        # Map to Sorted MediaPipe IDs
        sorted_face_mp_ids = sorted(FACE_MAPPING.values())
        mp_to_pkl_face = {v: k for k, v in FACE_MAPPING.items()}
        
        for i, mp_idx in enumerate(sorted_face_mp_ids):
            pkl_idx = mp_to_pkl_face[mp_idx]
            face[:, i] = face_4ch[:, pkl_idx]
            
    # Center Face at Nose (MP Index 1)
    # We need to find which index in our sorted 20 points corresponds to Nose (MP 1)
    # sorted_face_mp_ids[0] should be 1.
    nose_idx = sorted_face_mp_ids.index(1)
    
    for t in range(T):
        # Center if nose is present
        if np.any(face[t, nose_idx, :3] != 0):
            nose = face[t, nose_idx, :3]
            face[t, :, :3] = face[t, :, :3] - nose
            
    # Scale Face
    f_mask = face[..., 3] > 0
    if np.any(f_mask):
        f_max = np.max(np.abs(face[..., :3][f_mask]))
        f_scale = max(f_max, 1e-6)
        face[..., :3] /= f_scale

    return {
        'left': left_hand.astype(np.float32),
        'right': right_hand.astype(np.float32),
        'body': body.astype(np.float32),
        'face': face.astype(np.float32),
    }


def worker_load_multigraph_sample(args):
    """
    Helper function for parallel data loading.
    args: (sample_id, info, split, vocab)
    """
    sample_id, info, split, vocab = args
    
    data = load_multigraph_data(sample_id, info, split)
    
    if data is None:
        # Return empty tensors
        data = {
            'left': np.zeros((1, 21, 4), dtype=np.float32),
            'right': np.zeros((1, 21, 4), dtype=np.float32),
            'body': np.zeros((1, 9, 4), dtype=np.float32),
            'face': np.zeros((1, 18, 4), dtype=np.float32),
        }
    
    gloss_list = info['gloss']
    gloss_indices = (
        [vocab['<sos>']] +
        [vocab.get(tok, vocab['<unk>']) for tok in gloss_list] +
        [vocab['<eos>']]
    )
    
    gloss_string = " ".join(gloss_list)
    return {
        'left': torch.tensor(data['left'], dtype=torch.float32),
        'right': torch.tensor(data['right'], dtype=torch.float32),
        'body': torch.tensor(data['body'], dtype=torch.float32),
        'face': torch.tensor(data['face'], dtype=torch.float32),
        'gloss': torch.tensor(gloss_indices, dtype=torch.long),
        'gloss_string': gloss_string,
        'sample_id': sample_id,
    }


class SignLanguageMultiGraphDataset(Dataset):
    """Dataset for Multi-Graph ST-GCN."""
    
    def __init__(self, split, mappings, vocab, cache_data=True, num_processes=8):
        self.split = split
        self.mappings = mappings[split]
        self.vocab = vocab
        self.sample_ids = list(self.mappings.keys())
        self.cache_data = cache_data
        self.cached_samples = {}
        
        if cache_data:
            print(f"Pre-loading {split} dataset ({len(self.sample_ids)} samples) using {num_processes} processes...")
            
            if len(self.sample_ids) == 0:
                print(f"Warning: No samples found for split '{split}'. Check mappings.")
            else:
                load_args = []
                for sid in self.sample_ids:
                    load_args.append((sid, self.mappings[sid], self.split, self.vocab))
                
                # Use sequential for debugging if needed or if list is short
                results = []
                try:
                    with ProcessPoolExecutor(max_workers=num_processes) as executor:
                        results = list(tqdm(executor.map(worker_load_multigraph_sample, load_args), total=len(load_args), desc=f"Caching {split}"))
                except Exception as e:
                    print(f"Multiprocessing failed: {e}. Falling back to sequential.")
                    results = [worker_load_multigraph_sample(arg) for arg in tqdm(load_args, desc=f"Caching {split} (Sequential)")]

                for res in results:
                    if res is not None:
                        self.cached_samples[res['sample_id']] = res

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        if self.cache_data:
            return self.cached_samples[sample_id]
        
        # Fallback for on-the-fly loading
        info = self.mappings[sample_id]
        return worker_load_multigraph_sample((sample_id, info, self.split, self.vocab))


def collate_fn_multigraph(batch):
    """Collate function for multi-graph data."""
    batch = [b for b in batch if b['left'].shape[0] > 0]
    if not batch:
        return {}
    
    # Get max lengths
    max_seq_len = max(b['left'].shape[0] for b in batch)
    max_gloss_len = max(len(b['gloss']) for b in batch)
    
    batch_size = len(batch)
    
    # Prepare padded tensors
    left = torch.zeros(batch_size, max_seq_len, 21, 4)
    right = torch.zeros(batch_size, max_seq_len, 21, 4)
    body = torch.zeros(batch_size, max_seq_len, 9, 4)
    face = torch.zeros(batch_size, max_seq_len, 18, 4)
    glosses = torch.zeros(batch_size, max_gloss_len, dtype=torch.long)
    
    seq_lengths = []
    gloss_lengths = []
    sample_ids = []
    
    # Store raw text for MT5
    gt_sentences = []
    
    for i, b in enumerate(batch):
        T = b['left'].shape[0]
        L = len(b['gloss'])
        
        left[i, :T] = b['left']
        right[i, :T] = b['right']
        body[i, :T] = b['body']
        face[i, :T] = b['face']
        glosses[i, :L] = b['gloss']
        
        seq_lengths.append(T)
        gloss_lengths.append(L)
        sample_ids.append(b['sample_id'])
        gt_sentences.append(b['gloss_string'])
    
    src_input = {
        'left': left,
        'right': right,
        'body': body,
        'face': face,
        'name_batch': sample_ids, # models.py uses 'name_batch'
        'attention_mask': (left.sum(dim=(-1,-2)) != 0).long(),
        # Actually models.py expects 'attention_mask'. 
        # In original datasets.py: attention_mask is length-based mask.
    }
    
    # Create attention mask based on sequence length
    # Assuming all modalities have same length T (which they do by construction in load_multigraph_data)
    # We can just use seq_lengths
    mask_gen = []
    for l in seq_lengths:
        mask_gen.append(torch.ones(l))
    src_input['attention_mask'] = torch.nn.utils.rnn.pad_sequence(mask_gen, batch_first=True, padding_value=0).long()

    tgt_input = {
        'gt_sentence': gt_sentences,
        'gt_gloss': glosses, # fine_tuning.py might expect this for CSLR
    }

    return src_input, tgt_input
