import pickle
import numpy as np
import os

# Define Mappings
# PKL Index -> MediaPipe Index
POSE_MAPPING = {
    0: 1,   # Left Eye (Inner)
    1: 7,   # Left Ear
    2: 8,   # Right Ear
    3: 11,  # Left Shoulder
    4: 12,  # Right Shoulder
    5: 13,  # Left Elbow
    6: 14,  # Right Elbow
    7: 15,  # Left Wrist
    8: 16,  # Right Wrist
    9: 17,  # Left Pinky
    10: 18, # Right Pinky
    11: 19, # Left Index
    12: 20, # Right Index
    13: 21, # Left Thumb
    14: 22, # Right Thumb
    15: 23, # Left Hip
    16: 24, # Right Hip
}

FACE_MAPPING = {
    0: 1,   # Nose Tip
    1: 33,  # Left Eye Outer
    2: 152, # Chin Bottom
    3: 263, # Right Eye Outer
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

def load_and_test(file_path):
    print(f"Loading {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("-" * 20)
    
    # 1. Pose
    pose = np.array(data['pose']) # (T, 17, 4)
    print(f"Raw Pose Shape: {pose.shape}")
    mapped_pose = {}
    for pkl_idx, mp_idx in POSE_MAPPING.items():
        # Taking the first frame for visualization
        mapped_pose[mp_idx] = pose[0, pkl_idx]
    
    print("Mapped Pose (Frame 0 samples):")
    print(f"  MP ID 11 (Left Shoulder): {mapped_pose[11]}")
    print(f"  MP ID 12 (Right Shoulder): {mapped_pose[12]}")
    
    # 2. Face
    # Check if face exists
    if 'face' in data:
        face = np.array(data['face']) # (T, 20, 3)
        print(f"\nRaw Face Shape: {face.shape}")
        
        # Standardize to 4 channels
        T, V, C = face.shape
        if C == 3:
            print("  Face has 3 channels. Appending confidence 1.0.")
            conf = np.ones((T, V, 1))
            face_4ch = np.concatenate([face, conf], axis=-1)
        else:
            face_4ch = face
            
        print(f"  Final Face Shape: {face_4ch.shape}")
        
        mapped_face = {}
        for pkl_idx, mp_idx in FACE_MAPPING.items():
            mapped_face[mp_idx] = face_4ch[0, pkl_idx]

        print("Mapped Face (Frame 0 samples):")
        print(f"  MP ID 1 (Nose): {mapped_face[1]}")
        print(f"  MP ID 152 (Chin): {mapped_face[152]}")
    
    # 3. Hands
    left_hand = np.array(data['left_hand'])
    print(f"\nRaw Left Hand Shape: {left_hand.shape}")
    # Slice to 4 channels
    lh_4ch = left_hand[..., :4]
    print(f"  Sliced Left Hand Shape: {lh_4ch.shape}")
    
    return True

if __name__ == "__main__":
    test_path = r"D:\Datasets\Label_Corrected\processed\S000048_P0000_T00\data\keypoints.pkl"
    load_and_test(test_path)
