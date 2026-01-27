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
POSE_INDICES = [1, 7, 8, 11, 12, 13, 14, 15, 16]  # 9 nodes for body
FACE_INDICES = [
    # Jaw (9 points): 454, 288, 365, 378, 152, 149, 136, 58, 234
    454, 288, 365, 378, 152, 149, 136, 58, 234,
    # Inner Mouth (8 points): 13, 14, 78, 308, 81, 178, 311, 402
    13, 14, 78, 308, 81, 178, 311, 402,
    # Nose (1 point): 1
    1
]


def crop_scale_3d(motion, thr=0.3):
    """
    Normalize 3D motion to [-1, 1] using dynamic bounding box.
    
    Args:
        motion: (T, N, 4) where last dim is [x, y, z, confidence]
        thr: confidence threshold
        
    Returns:
        result: (T, N, 3) normalized coordinates
        scale: float, the scale factor used
    """
    result = copy.deepcopy(motion)
    
    # Find valid coordinates (confidence > threshold)
    valid_mask = motion[..., 3] > thr
    valid_coords = motion[valid_mask][:, :3]
    
    if len(valid_coords) < 4:
        return np.zeros(motion.shape[:-1] + (3,)), 0.0
    
    # Compute bounding box
    xmin, ymin, zmin = valid_coords.min(axis=0)
    xmax, ymax, zmax = valid_coords.max(axis=0)
    
    # Scale is max extent
    scale = max(xmax - xmin, ymax - ymin, zmax - zmin)
    
    if scale == 0:
        return np.zeros(motion.shape[:-1] + (3,)), 0.0
    
    # Center
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    cz = (zmin + zmax) / 2
    
    # Normalize to [-1, 1]
    result_3d = (motion[..., :3] - np.array([cx, cy, cz])) / scale * 2
    result_3d = np.clip(result_3d, -1, 1)
    
    # Mask invalid points
    result_3d[~valid_mask] = 0
    
    return result_3d, scale


def clean_dataframe(df):
    """
    Apply linear interpolation to fill missing values.
    Ref: x[t] = (x[t-1] + x[t+1]) / 2
    Edge cases (start/end) are filled with nearest valid value.
    """
    # Sort just in case, though usually sorted
    if 'frame' in df.columns:
        df = df.sort_values('frame')
        
    # Interpolate internal gaps
    df = df.interpolate(method='linear', limit_direction='both')
    
    # Fill edges (if any NaNs remain at start or end)
    df = df.ffill().bfill()
    
    # Final safety: fill any remaining NaNs (e.g. if *all* are NaN) with 0
    df = df.fillna(0)
    
    return df


def load_multigraph_data(sample_id, info, split='train'):
    """
    Load multimodal data for a single sample.
    
    Args:
        sample_id: Sample identifier
        info: Dict with 'data_path', 'group', 'gloss'
        split: Dataset split ('train', 'dev', 'test')
        
    Returns:
        Dict with keys 'left', 'right', 'body', 'face', each (T, V, 3)
    """
    group = info['group']
    
    # === Load Hand Data ===
    # === Load Hand Data ===
    hand_path = DataConfig.get_hand_data_path(split, group, sample_id)
    
    if not os.path.exists(hand_path):
        return None
        
    df_hand = pd.read_csv(hand_path)
    df_hand = clean_dataframe(df_hand)
    
    frame_groups = df_hand.groupby('frame')
    sorted_frames = sorted(frame_groups.groups.keys())
    T = len(sorted_frames)
    
    left_hand = np.zeros((T, 21, 3))
    right_hand = np.zeros((T, 21, 3))
    
    for t, frame_num in enumerate(sorted_frames):
        # In wide format, each frame usually has one row containing both hands
        group_df = frame_groups.get_group(frame_num)
        if group_df.empty:
            continue
            
        row = group_df.iloc[0]
        
        # Extract Left Hand (LH_world_i_x)
        for i in range(21):
            left_hand[t, i] = [
                row.get(f'LH_world_{i}_x', 0),
                row.get(f'LH_world_{i}_y', 0),
                row.get(f'LH_world_{i}_z', 0),
            ]
            
        # Extract Right Hand (RH_world_i_x)
        for i in range(21):
            right_hand[t, i] = [
                row.get(f'RH_world_{i}_x', 0),
                row.get(f'RH_world_{i}_y', 0),
                row.get(f'RH_world_{i}_z', 0),
            ]
    
    # Normalize hands (center at wrist)
    for t in range(T):
        if np.any(left_hand[t] != 0):
            left_hand[t] = left_hand[t] - left_hand[t, 0:1]
        if np.any(right_hand[t] != 0):
            right_hand[t] = right_hand[t] - right_hand[t, 0:1]
    
    # Scale hands to unit
    left_scale = np.max(np.abs(left_hand)) if np.any(left_hand != 0) else 1.0
    right_scale = np.max(np.abs(right_hand)) if np.any(right_hand != 0) else 1.0
    hand_scale = max(left_scale, right_scale, 1e-6)
    
    left_hand = left_hand / hand_scale
    right_hand = right_hand / hand_scale
    
    # === Load Pose Data ===
    pose_path = DataConfig.get_pose_data_path(split, group, sample_id)
    body = np.zeros((T, 9, 3))
    
    if os.path.exists(pose_path):
        df_pose = pd.read_csv(pose_path)
        df_pose = clean_dataframe(df_pose)
        
        pose_groups = {f: g for f, g in df_pose.groupby('frame')}
        
        # Build confidence array for crop_scale_3d
        pose_raw = np.zeros((T, 33, 4))  # 33 full pose points with confidence
        
        for t, frame_num in enumerate(sorted_frames):
            if frame_num in pose_groups:
                row = pose_groups[frame_num].iloc[0]
                for i in range(33):
                    pose_raw[t, i] = [
                        row.get(f'world_landmark_{i}_x', 0),
                        row.get(f'world_landmark_{i}_y', 0),
                        row.get(f'world_landmark_{i}_z', 0),
                        row.get(f'world_landmark_{i}_visibility', 1.0),
                    ]
        
        # Apply crop_scale_3d to full pose
        pose_normed, _ = crop_scale_3d(pose_raw, thr=0.3)
        
        # Extract subset indices
        for i, pose_idx in enumerate(POSE_INDICES):
            body[:, i, :] = pose_normed[:, pose_idx, :]
    
    # === Load Face Data ===
    face_path = DataConfig.get_face_data_path(split, group, sample_id)
    face = np.zeros((T, 18, 3))
    
    if os.path.exists(face_path):
        df_face = pd.read_csv(face_path)
        df_face = clean_dataframe(df_face)


        
        # Check if face has landmarks
        if 'frame' in df_face.columns:
            face_groups = {f: g for f, g in df_face.groupby('frame')}
            
            for t, frame_num in enumerate(sorted_frames):
                if frame_num in face_groups:
                    row = face_groups[frame_num].iloc[0]
                    
                    # Extract face landmarks (center at nose)
                    face_coords = np.zeros((18, 3))
                    for i, face_idx in enumerate(FACE_INDICES):
                        x = row.get(f'world_landmark_{face_idx}_x', 0)
                        y = row.get(f'world_landmark_{face_idx}_y', 0)
                        z = row.get(f'world_landmark_{face_idx}_z', 0)
                        face_coords[i] = [x, y, z]
                    
                    # Center at nose (last index = 17)
                    if np.any(face_coords != 0):
                        face_coords = face_coords - face_coords[17:18]
                    
                    face[t] = face_coords
            
            # Scale face to unit
            face_scale = np.max(np.abs(face)) if np.any(face != 0) else 1.0
            face = face / max(face_scale, 1e-6)
    
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
            'left': np.zeros((1, 21, 3), dtype=np.float32),
            'right': np.zeros((1, 21, 3), dtype=np.float32),
            'body': np.zeros((1, 9, 3), dtype=np.float32),
            'face': np.zeros((1, 18, 3), dtype=np.float32),
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
            
            load_args = []
            for sid in self.sample_ids:
                load_args.append((sid, self.mappings[sid], self.split, self.vocab))
            
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                results = list(tqdm(executor.map(worker_load_multigraph_sample, load_args), total=len(load_args), desc=f"Caching {split}"))
            
            for res in results:
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
    left = torch.zeros(batch_size, max_seq_len, 21, 3)
    right = torch.zeros(batch_size, max_seq_len, 21, 3)
    body = torch.zeros(batch_size, max_seq_len, 9, 3)
    face = torch.zeros(batch_size, max_seq_len, 18, 3)
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
        'attention_mask': (left.sum(dim=(-1,-2)) != 0).long(), # Approximate mask if not provided? 
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
