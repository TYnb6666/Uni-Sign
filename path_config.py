import os

class DataConfig:
    # Project Root
    # Default to the environment variable if set, otherwise use the hardcoded path
    PROJECT_ROOT = os.environ.get('PROJECT_ROOT', r"D:\3dHandGeature")
    
    # ---------------------------------------------------------
    # Data Data Roots (Can be Absolute Paths)
    # ---------------------------------------------------------
    # If your data is on another disk or outside the project, set these directly.
    # We use os.environ.get to allow server configuration without changing code.
    
    # Hand Data
    HAND_DATA_ROOT = os.environ.get('HAND_DATA_ROOT', r"D:\Datasets\Label_Corrected\processed")

    # Face Data
    FACE_DATA_ROOT = os.environ.get('FACE_DATA_ROOT', r"D:\Datasets\Label_Corrected\processed")
    
    # Pose Data
    POSE_DATA_ROOT = os.environ.get('POSE_DATA_ROOT', r"D:\Datasets\Label_Corrected\processed")
    
    # Label Data
    LABEL_ROOT = os.environ.get('LABEL_ROOT', r"D:\Datasets\sentence_label")
    # ---------------------------------------------------------
    
    @classmethod
    def get_label_file(cls, split):
        # Using split_1.txt for train/dev/test split info, but strict label file usage depends on loader
        return os.path.join(cls.LABEL_ROOT, f'{split}.pkl' if split != 'split_1' else 'split_1.txt')

    @classmethod
    def get_hand_data_path(cls, split, group, sample_id):
        # Structure: D:\Datasets\Label_Corrected\processed\<video_name>\data\keypoints.pkl
        return os.path.join(cls.HAND_DATA_ROOT, sample_id, "data", "keypoints.pkl")

    @classmethod
    def get_face_data_path(cls, split, group, sample_id):
        return os.path.join(cls.FACE_DATA_ROOT, sample_id, "data", "keypoints.pkl")

    @classmethod
    def get_pose_data_path(cls, split, group, sample_id):
        return os.path.join(cls.POSE_DATA_ROOT, sample_id, "data", "keypoints.pkl")
