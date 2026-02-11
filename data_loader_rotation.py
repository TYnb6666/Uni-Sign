import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

from path_config import DataConfig

# 配置参数
BASE_PATH = DataConfig.PROJECT_ROOT
DATA_SPLITS = ['train', 'test', 'dev']
HAND_GROUPS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

# 构建标签文件路径
label_files = {
    split: DataConfig.get_label_file(split)
    for split in DATA_SPLITS
}


# 1. 读取标签文件并创建数据映射
# 1. 读取标签文件并创建数据映射
def load_label_mappings(task='SLT'):
    mappings = {}
    
    # CSL-Daily specific logic
    label_root = DataConfig.LABEL_ROOT
    
    # Load CSL-Daily Labels (csl2020ct_v2.pkl)
    pkl_path = os.path.join(label_root, 'csl2020ct_v2.pkl') 
    csl_labels = {}
    if os.path.exists(pkl_path):
        import pickle
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            # Structure: {'info': [{'name': '...', 'label_gloss': [...], ...}, ...], 
            #             'gloss_map': [...], 'char_map': [...], ...}
            if isinstance(data, dict) and 'info' in data:
                for item in data['info']:
                    csl_labels[item['name']] = item
            elif isinstance(data, list):
                for item in data:
                    csl_labels[item['name']] = item
            else:
                print(f"Warning: Unknown data format in {pkl_path}, keys={list(data.keys()) if isinstance(data, dict) else type(data)}")
    else:
         print(f"Warning: {pkl_path} not found.")

    # Load Split Info (split_1.txt)
    split_path = os.path.join(label_root, 'split_1.txt')
    split_map = {}
    if os.path.exists(split_path):
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Format: "S000048_P0000_T00|train"
                line = line.strip()
                if not line or line.startswith('name|'): continue
                
                parts = line.split('|')
                if len(parts) < 2: 
                     parts = line.split() # Try whitespace
                
                if len(parts) >= 2:
                    vid_name = parts[0]
                    split_name = parts[1]
                    split_map[vid_name] = split_name
    else:
        print(f"Warning: {split_path} not found.")
    
    # Construct mappings
    for split in ['train', 'dev', 'test']:
        mappings[split] = {}
        
    for vid_name, info in csl_labels.items():
        if vid_name not in split_map:
            continue
            
        split = split_map[vid_name]
        if split not in mappings:
            continue
            
        data_path = DataConfig.get_hand_data_path(split, 'default', vid_name)
        
        # Use 'label_gloss' for CSL-Daily
        gloss = info.get('label_gloss', info.get('gloss', []))
        
        mappings[split][vid_name] = {
            'gloss': gloss,
            'text': " ".join(gloss), # Simple join for text
            'data_path': data_path
        }

    return mappings


# 2. 构建词汇表（从训练集创建）
def build_vocab(mappings):
    word_counts = defaultdict(int)
    train_mappings = mappings['train']

    for data in train_mappings.values():
        for word in data['gloss']:
            word_counts[word] += 1

    # 创建词汇表（添加特殊符号）
    vocab = {
        '<pad>': 0,  # 填充符
        '<sos>': 1,  # 序列开始
        '<eos>': 2,  # 序列结束
        '<unk>': 3  # 未知词
    }

    # 添加实际词汇
    for idx, word in enumerate(word_counts.keys(), start=len(vocab)):
        vocab[word] = idx

    return vocab


# 计算两个向量之间的角度（弧度）
def calculate_angle(v1, v2):
    """计算两个向量之间的角度（弧度）"""
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # 处理除零情况
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0

    cos_theta = dot_product / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在有效范围内
    return np.arccos(cos_theta)

def get_canonical_transform(hand_points):
    """
    计算基于手掌平面的Canonical Frame旋转矩阵
    hand_points: (21, 3)
    返回: R (3, 3) 旋转矩阵, valid (bool) 是否有效
    """
    # 0: wrist, 5: index_mcp, 17: pinky_mcp
    p0 = hand_points[0]
    p5 = hand_points[5]
    p17 = hand_points[17]
    
    # Check validity (if all zeros)
    if np.all(p0 == 0) and np.all(p5 == 0) and np.all(p17 == 0):
        return np.eye(3), False

    # Check for NaNs or Infs
    if np.isnan(hand_points).any() or np.isinf(hand_points).any():
        return np.eye(3), False

    # X axis: wrist -> index_mcp
    x_axis = p5 - p0
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-6:
        return np.eye(3), False
    x_axis = x_axis / norm_x
    
    # Temp vector for Z calculation: wrist -> pinky_mcp
    v_temp = p17 - p0
    
    # Z axis: Normal to palm plane (cross product of X and wrist->pinky)
    z_axis = np.cross(x_axis, v_temp)
    norm_z = np.linalg.norm(z_axis)
    if norm_z < 1e-6:
        # Fallback if colinear
        return np.eye(3), False
    z_axis = z_axis / norm_z
    
    # Consistency check: If Z points "inwards" (z-component negative assuming camera is +Z looking at subject? 
    # Or subject facing camera... User said: "如果 z_axis 与 “相机朝向/人体朝向”点乘为负，就把 z 取反"
    # Usually camera is at origin looking towards -Z or +Z? 
    # Let's assume standard intuitive frame: Camera at origin, looking at hand. 
    # User said: "在没有人体/相机方向时，简单做法是：保证 z 的某一维（比如 z 分量）为正。"
    if z_axis[2] < 0:
        z_axis = -z_axis
        # And we need to flip Y to maintain right-hand rule later?
        # User said: "如果是左手系...就把 z 取反，同时 y 也取反（保持右手系）"
        # Wait, cross(X, Z) -> -Y if Z is flipped. 
        # If we flip Z, we SHOULD recalculate Y using new Z to be sure.
        
    # Y axis: cross(Z, X)
    y_axis = np.cross(z_axis, x_axis)
    # y_axis should be normalized already if Z and X are orthogonal and normalized
    
    # Construct Rotation Matrix [x, y, z] (Col vectors or Row? User said "p_aligned = R^T * p")
    # If R = [x, y, z] (3x3), then R * e1 = x. 
    # We want x-axis to map to e1=[1,0,0]. 
    # So we need R_align such that R_align * x = e1.
    # If Rot_matrix columns are new basis axes expressed in old basis: M = [x', y', z']
    # Then v_new = M^T * v_old.
    # So yes, we construct Matrix with columns X, Y, Z.
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    return rotation_matrix, True

# 3. 三维数据处理函数（添加关节角度特征 & 旋转对齐 & 四元数）
def load_hand_data(file_path):
    # 读取手部数据CSV文件
    df = pd.read_csv(file_path)

    # 筛选有效的手部数据（hand_id不为-1）
    valid_frames = df[df['hand_id'] != -1]

    # 按帧分组处理数据
    frame_data = {}
    for frame_num, group in valid_frames.groupby('frame_num'):
        hands = defaultdict(lambda: np.zeros(63))  # 21关键点 * 3坐标

        for _, row in group.iterrows():
            hand_type = row['handedness']
            coords = []

            # 提取所有关键点坐标
            for i in range(21):
                x = row[f'world_landmark_{i}_x']
                y = row[f'world_landmark_{i}_y']
                z = row[f'world_landmark_{i}_z']
                coords.extend([x, y, z])

            # 存储左右手数据
            hands[hand_type] = np.array(coords)

        # 确保左右手数据顺序一致：左手在前，右手在后
        frame_vector = np.concatenate([
            hands.get('Left', np.zeros(63)),
            hands.get('Right', np.zeros(63))
        ])

        frame_data[frame_num] = frame_vector

    # 按帧号排序并组成序列
    sequence = np.array([frame_data[k] for k in sorted(frame_data.keys())])
    
    # Handle empty sequence
    if sequence.size == 0:
         return np.zeros((10, 422)) # Return dummy small seq

    # === 数据完整性检查 ===
    if np.isnan(sequence).any() or np.isinf(sequence).any():
        print(f"警告: 文件 {os.path.basename(file_path)} 包含NaN/Inf值")
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)

    # === 数据预处理 ===
    if sequence.shape[0] < 3:
        extended_seq = np.zeros((sequence.shape[0], 422))
        return extended_seq

    # 1. 归一化处理：寻找第一个有效帧（非全零）作为平移基准
    first_frame = sequence[0]
    for i in range(sequence.shape[0]):
        if np.any(sequence[i] != 0):
            first_frame = sequence[i]
            break
            
    # 平移对齐 (Global translation alignment still useful?)
    # User said: "只替换坐标为 aligned（旋转不变）"
    # Actually, if we do per-frame alignment to canonical frame (Wrist at origin), 
    # we implicitly do translation alignment per frame for the wrist.
    # The user instruction implies: "每帧每只手计算 R... p_aligned = R^T * (p - wrist)"
    # So we should probably do per-frame Wrist-centering AND Rotation.
    
    # Let's perform the rotation alignment per frame first.
    
    aligned_sequence_list = []
    quaternion_list = []
    
    # Store previous valid rotation for filling missing frames
    prev_left_quat = np.array([0., 0., 0., 0.]) # Start with 0 as requested for missing
    prev_right_quat = np.array([0., 0., 0., 0.])
    
    # Can also use Identity quaternion [0, 0, 0, 1] (x,y,z,w) or [1, 0, 0, 0] (w, x, y, z)
    # User said: "R is None 时：quat 全 0"
    
    for i in range(sequence.shape[0]):
        # Extract original hands
        left_hand = sequence[i, :63].reshape(21, 3)
        right_hand = sequence[i, 63:126].reshape(21, 3)
        
        # --- LEFT HAND ---
        left_R, left_valid = get_canonical_transform(left_hand)
        if left_valid:
            # Shift wrist to origin
            left_hand_centered = left_hand - left_hand[0]
            # Rotate: P_new = P_old @ R (if row vectors and R contains col basis)
            # Actually: v_new = R^T * v_old
            # For row vec: v_new = v_old @ R
            left_hand_aligned = left_hand_centered @ left_R
            
            # Quat
            try:
                r_obj = R.from_matrix(left_R)
                left_quat = r_obj.as_quat() # x, y, z, w
            except Exception:
                left_quat = np.zeros(4)
                left_hand_aligned = np.zeros_like(left_hand)

            prev_left_quat = left_quat
        else:
            left_hand_aligned = np.zeros_like(left_hand)
            left_quat = np.zeros(4) # As requested: "quat 全 0"

        # --- RIGHT HAND ---
        right_R, right_valid = get_canonical_transform(right_hand)
        if right_valid:
            right_hand_centered = right_hand - right_hand[0]
            right_hand_aligned = right_hand_centered @ right_R
            
            try:
                r_obj = R.from_matrix(right_R)
                right_quat = r_obj.as_quat()
            except Exception:
                right_quat = np.zeros(4)
                right_hand_aligned = np.zeros_like( right_hand)
                
            prev_right_quat = right_quat
        else:
            right_hand_aligned = np.zeros_like(right_hand)
            right_quat = np.zeros(4)
            
        # Reconstruct frame
        # User said "位置：用 aligned_seq"
        aligned_frame = np.concatenate([left_hand_aligned.flatten(), right_hand_aligned.flatten()])
        aligned_sequence_list.append(aligned_frame)
        
        # Quats
        quat_frame = np.concatenate([left_quat, right_quat])
        quaternion_list.append(quat_frame)
        
    aligned_sequence = np.array(aligned_sequence_list)
    quaternions = np.array(quaternion_list)

    # === 尺度归一化 (Scale) ===
    # Even if aligned, scale might vary. Keep scale normalization?
    # User didn't say to remove it. "坐标用“每帧对齐”" implies local frame.
    # Usually in canonical frame we might want to normalize size too, or keep it to preserve hand size info.
    # Let's keep the user's previous logic of global scale normalization if possible, 
    # OR since we align every frame to wrist, maybe we just normalize by hand size per frame?
    # Original code calculated one global scale from the first valid frame.
    # Let's stick to the Original Logic for Scale to be safe, but applied to the *aligned* sequence?
    # Or should we normalize each frame to unit hand size?
    # "canonical frame" often implies unit size too, but user only mentioned rotation.
    # Let's apply the SAME scale normalization logic as before (using reference valid frame).
    
    # Use first valid frame for Reference Scale (as in original code)
    # But now using aligned sequence? Or original?
    # Distance is rotation invariant. So original or aligned is fine.
    
    # Copy-paste Scale Logic
    scale_ref_frame = sequence[0]
    for i in range(sequence.shape[0]):
        if np.any(sequence[i] != 0):
            scale_ref_frame = sequence[i]
            break
            
    left_hand_ref = scale_ref_frame[:63].reshape(21, 3)
    right_hand_ref = scale_ref_frame[63:126].reshape(21, 3)
    
    def get_hand_scale(hand_coords):
        if np.all(hand_coords == 0): return 0.0
        wrist = hand_coords[0]
        middle_mcp = hand_coords[9]
        dist = np.linalg.norm(middle_mcp - wrist)
        return dist

    scale_left = get_hand_scale(left_hand_ref)
    scale_right = get_hand_scale(right_hand_ref)
    
    scale = 1.0
    if scale_left > 0 and scale_right > 0:
        scale = (scale_left + scale_right) / 2.0
    elif scale_left > 0:
        scale = scale_left
    elif scale_right > 0:
        scale = scale_right
        
    if scale < 1e-6: scale = 1.0
        
    # Apply scale to ALIGNED sequence
    normalized_seq = aligned_sequence / scale
    
    # 2. 计算速度特征（一阶差分）
    velocity = np.zeros_like(normalized_seq)
    velocity[1:] = normalized_seq[1:] - normalized_seq[:-1]

    # 3. 计算加速度特征（二阶差分）
    acceleration = np.zeros_like(velocity)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    # Clean up NaNs in vel/acc
    velocity = np.nan_to_num(velocity)
    acceleration = np.nan_to_num(acceleration)

    # 4. 拼接特征：位置(126) + 速度(126) + 加速度(126) = 378
    extended_seq = np.zeros((sequence.shape[0], 378))
    extended_seq[:, :126] = normalized_seq
    extended_seq[:, 126:252] = velocity
    extended_seq[:, 252:] = acceleration

    # === 双手相对位置特征 (18) ===
    # Use ALIGNED coords? 
    # If both hands are in their OWN canonical frame (Wrist at 0), relative position of Right Wrist to Left Wrist is 0 - 0 = 0?
    # NO! If we align each hand to *its own* wrist, we lose the relative position between hands (Wrist-to-Wrist vector).
    # User said: "位置：用 aligned_seq"
    # User did NOT say to keep global relative info. But "Rotational Alignment" usually destroys global translation relative to world.
    # However, for two hands, relative position is important.
    # If we map both to (0,0,0) wrist, we lose `right_wrist - left_wrist`.
    # Maybe we should add `right_wrist_global - left_wrist_global` as a feature?
    # In original code: `relative_features` tracks `right_hand[joint] - left_hand[joint]`.
    # If `normalized_seq` has both wrists at 0, then `relative_features` will be calculated based on local frames.
    # BUT, the user prompt implies we want to preserve rotation info via quaternions and use aligned points for local shape.
    # If we strictly follow "Use aligned_seq" for everything, we might lose cross-hand distance.
    # Let's keep the `relative_features` logic based on `normalized_seq` (which is aligned).
    # Note: If wrists are 0, then relative wrist is 0.
    
    relative_features = np.zeros((sequence.shape[0], 18))
    key_joints = [0, 4, 8, 12, 16, 20]

    for i in range(sequence.shape[0]):
        left_hand = normalized_seq[i, :63].reshape(21, 3)
        right_hand = normalized_seq[i, 63:126].reshape(21, 3)

        # Right - Left (Wrists will be 0-0=0)
        relative_features[i, :3] = right_hand[0] - left_hand[0]
        for j, joint_idx in enumerate(key_joints[1:]):
            idx = 3 + j * 3
            # This will represent "Right Finger in Right Frame - Left Finger in Left Frame"
            # Which might be meaningless for cross-hand interaction if they are far apart in reality?
            # BUT the user asked for this specific alignment. I will stick to it.
            # If the user wants global relative info they might need to add it explicitly.
            # *wait*, user said: "位置：用 aligned_seq（126）"
            # Feature total: 422. (Original was 414).
            # 414 = 126+126+126 + 18(rel) + 18(angle).
            # New = 422. 414 + 8(quat) = 422.
            # So we KEEP all original features, just constructed from aligned data + Quats.
            relative_features[i, idx:idx + 3] = right_hand[joint_idx] - left_hand[joint_idx]

    # === 关节角度特征 (18) ===
    # Calculate on aligned data (Angle is rotation invariant anyway, so should be same)
    angle_features = np.zeros((sequence.shape[0], 18))
    for i in range(sequence.shape[0]):
        left_hand = normalized_seq[i, :63].reshape(21, 3)
        right_hand = normalized_seq[i, 63:126].reshape(21, 3)

        # Left Hand Angles
        for j in range(6):
            if j == 0:
                p1, p2, p3 = left_hand[0], left_hand[1], left_hand[17]
                angle_features[i, j] = calculate_angle(p2 - p1, p3 - p1)
            else:
                p1, p2, p3 = left_hand[0], left_hand[j], left_hand[j * 4]
                angle_features[i, j] = calculate_angle(p2 - p1, p3 - p2)

        # Right Hand Angles
        for j in range(6):
            if j == 0:
                p1, p2, p3 = right_hand[0], right_hand[1], right_hand[17]
                angle_features[i, 6 + j] = calculate_angle(p2 - p1, p3 - p1)
            else:
                p1, p2, p3 = right_hand[0], right_hand[j], right_hand[j * 4]
                angle_features[i, 6 + j] = calculate_angle(p2 - p1, p3 - p2)

        # Cross Hand Angles (Directional)
        # Wrist vector (will be 0 if aligned to 0) -> this feature becomes 0 or undefined (NaN).
        # IF wrists are both at 0, wrist_vector = 0.
        # normalize(0) -> NaN or warnings.
        # User might have overlooked this or intends to use global wrist dist?
        # "在没有人体/相机方向时..."
        # Given strict instruction to align to canonical frame, Wrist-to-Wrist vector is lost.
        # I will set these undefined angles to 0 to avoid NaNs.
        
        # NOTE: If we want to keep cross-hand info, we usually align RELATIVE to a root (e.g. torso), not independent hands.
        # But instructions say: "为每帧每只手计算旋转矩阵...把点对齐到 canonical"
        # This implies independent alignment.
        
        # 1. Wrist Vector (LOST) -> Set to 0
        
        # 2. Finger Directions (Preserved in local frame)
        # relative angle between "Left Index in Left Frame" and "Right Index in Right Frame"
        # This tells us if fingers are pointing in same relative direction *w.r.t their own palms*.
        # This is actually a valid "coordination" feature.
        
        for k, joint_idx in enumerate([1, 5, 9, 13, 17]):
            idx = 12 + k
            left_dir = left_hand[joint_idx] - left_hand[0]
            right_dir = right_hand[joint_idx] - right_hand[0] # Wrists are 0 anyway
            
            # These are just vectors from wrist to finger base in local frame.
            
            l_norm = np.linalg.norm(left_dir)
            r_norm = np.linalg.norm(right_dir)
            
            if l_norm > 1e-6 and r_norm > 1e-6:
                angle_features[i, idx] = calculate_angle(left_dir, right_dir)
            else:
                angle_features[i, idx] = 0.0

    # 5. 合并所有特征
    # extended(378) + relative(18) + angle(18) + quaternions(8) = 422
    final_features = np.concatenate([extended_seq, relative_features, angle_features, quaternions], axis=1)

    # 最终检查
    if np.isnan(final_features).any() or np.isinf(final_features).any():
        print(f"严重警告: 最终特征包含NaN/Inf - {os.path.basename(file_path)}")
        final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)

    return final_features


# 4. 自定义数据集类
class SignLanguageDataset(Dataset):
    def __init__(self, split, mappings, vocab):
        self.split = split
        self.mappings = mappings[split]
        self.vocab = vocab
        self.sample_ids = list(self.mappings.keys())
        
        # Memory Cache
        self.cache = {}
        self.enable_cache = True

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        info = self.mappings[sample_id]

        if self.enable_cache and sample_id in self.cache:
            return self.cache[sample_id]

        data_path = info['data_path']
        sequence = load_hand_data(data_path)

        gloss_ids = [self.vocab['<sos>']]
        for word in info['gloss']:
            gloss_ids.append(self.vocab.get(word, self.vocab['<unk>']))
        gloss_ids.append(self.vocab['<eos>'])
        
        result = {
            'sequence': sequence,
            'gloss': np.array(gloss_ids),
            'sample_id': sample_id
        }
        
        if self.enable_cache:
            self.cache[sample_id] = result
            
        return result


# 5. 数据批处理函数
def collate_fn(batch):
    sequences = [item['sequence'] for item in batch]
    glosses = [item['gloss'] for item in batch]
    sample_ids = [item['sample_id'] for item in batch]

    # 填充序列
    seq_lengths = [seq.shape[0] for seq in sequences]
    max_seq_len = max(seq_lengths)

    # 422 维特征
    padded_seqs = np.zeros((len(sequences), max_seq_len, 422))
    for i, seq in enumerate(sequences):
        padded_seqs[i, :seq.shape[0]] = seq

    gloss_lengths = [len(gloss) for gloss in glosses]
    max_gloss_len = max(gloss_lengths)
    padded_glosses = np.zeros((len(glosses), max_gloss_len), dtype=np.int64)
    for i, gloss in enumerate(glosses):
        padded_glosses[i, :len(gloss)] = gloss

    return {
        'sequences': torch.tensor(padded_seqs, dtype=torch.float32),
        'glosses': torch.tensor(padded_glosses, dtype=torch.long),
        'seq_lengths': torch.tensor(seq_lengths, dtype=torch.long),
        'gloss_lengths': torch.tensor(gloss_lengths, dtype=torch.long),
        'sample_ids': sample_ids
    }
