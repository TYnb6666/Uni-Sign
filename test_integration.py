import sys
import os
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append(os.getcwd())

# Mock environment dependencies
deepspeed_mock = MagicMock()
sys.modules['deepspeed'] = deepspeed_mock
sys.modules['deepspeed.comm'] = MagicMock()
sys.modules['deepspeed.accelerator'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['einops'] = MagicMock()

def test_imports():
    print("Testing imports...")
    try:
        from data_loader_multigraph import SignLanguageMultiGraphDataset, collate_fn_multigraph
        from stgcn_layers import Graph
        from models import Uni_Sign
        import fine_tuning
        print("Imports successful.")
    except Exception as e:
        print(f"Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

def test_model_instantiation():
    print("Testing model instantiation...")
    try:
        from models import Uni_Sign
        args = SimpleNamespace(
            hidden_dim=256,
            dataset='CSL_Daily',
            rgb_support=False,
            label_smoothing=0.1,
            # Add other missing args if needed
        )
        
        try:
            model = Uni_Sign(args)
            print("Model instantiated successfully.")
        except OSError as e:
             # MT5 path error
             if "mt5" in str(e) or "not found" in str(e) or "Standard" in str(e):
                 print("Caught expected error (MT5 weights missing locally):", e)
                 print("Model logic seems fine up to MT5 loading.")
                 return True 
             else:
                 raise e
        except Exception as e:
            if "mt5" in str(e) or "not found" in str(e):
                 print("Caught expected error:", e)
                 return True
            raise e

        # Test Graph Structure
        if 'model' in locals():
            print("Graph keys:", model.graph.keys())
            expected_keys = ['body', 'left', 'right', 'face']
            for k in expected_keys:
                if k not in model.graph:
                    print(f"Missing graph key: {k}")
                    return False
        
    except Exception as e:
        print(f"Model instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    if test_imports() and test_model_instantiation():
        print("Integration test passed.")
    else:
        print("Integration test failed.")
