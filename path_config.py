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
    HAND_DATA_ROOT = os.environ.get('HAND_DATA_ROOT', os.path.join(PROJECT_ROOT, "Hand_Processed"))
    # HAND_DATA_ROOT = r"E:\ServerData\Hands"

    # Face Data
    FACE_DATA_ROOT = os.environ.get('FACE_DATA_ROOT', r"D:\Datasets\Face_Processed")
    
    # Pose Data
    POSE_DATA_ROOT = os.environ.get('POSE_DATA_ROOT', r"D:\Datasets\Pose_Processed")
    
    # Label Data
    LABEL_ROOT = os.environ.get('LABEL_ROOT', os.path.join(PROJECT_ROOT, "CE-CSL", "label"))
    # ---------------------------------------------------------
    
    @classmethod
    def get_label_file(cls, split):
        return os.path.join(cls.LABEL_ROOT, f'{split}.csv')

    @classmethod
    def get_hand_data_path(cls, split, group, sample_id):
        # Matches structure: Hand_Processed/train/A/train-00001/data/hand_data.csv
        return os.path.join(cls.HAND_DATA_ROOT, split, group, sample_id, "data", "hand_data.csv")

    @classmethod
    def get_face_data_path(cls, split, group, sample_id):
        # Assuming typical structure for Face data if it follows the pattern
        # Adjust as needed based on actual Face data structure
        return os.path.join(cls.FACE_DATA_ROOT, split, group, sample_id, "data", "face_data.csv")

    @classmethod
    def get_pose_data_path(cls, split, group, sample_id):
        return os.path.join(cls.POSE_DATA_ROOT, split, group, sample_id, "data", "pose_data.csv")
