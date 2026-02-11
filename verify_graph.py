import sys
import os

# Ensure we can import from the project root
sys.path.append(os.getcwd())

from stgcn_layers.MultiGraphs import Graph
import numpy as np

def verify_face_graph():
    print("Verifying Face Graph...")
    graph = Graph(layout='face', strategy='distance') # strategy might need 'distance' or 'spatial' or 'uniform'
    # Default strategy in models.py is 'distance'
    
    A = graph.A
    print(f"Adjacency Matrix Shape: {A.shape}")
    
    # strategy='distance' with max_hop=1 generates 2 matrices (0-hop, 1-hop)
    expected_shape = (2, 18, 18) # (K, V, V)
    if A.shape == expected_shape:
        print("PASS: Shape is correct.")
    else:
        print(f"FAIL: Expected {expected_shape}, got {A.shape}")
        
    # Check center (Nose)
    # Nose is MP 1. In sorted list [1, 13, ...], it is at index 0.
    center_idx = 0
    print(f"Center Node Index: {graph.center}")
    if graph.center == center_idx:
        print("PASS: Center index is correct (0).")
    else:
        print(f"FAIL: Expected 0, got {graph.center}")
        
    print("Verification Complete.")

if __name__ == "__main__":
    verify_face_graph()
