# model/ST_GCN/MultiGraphs.py
"""
Modular Graph Definitions for Multi-Graph ST-GCN.

Layouts:
- 'left' / 'right': 21 nodes (MediaPipe Hand)
- 'body': 9 nodes (Pose subset)
- 'face': 18 nodes (Face subset)
"""

import numpy as np


class Graph:
    """
    Graph structure for ST-GCN.

    Args:
        layout (str): One of 'left', 'right', 'body', 'face'
        strategy (str): 'uniform', 'distance', or 'spatial'
        max_hop (int): Maximum hop distance
        dilation (int): Dilation for hop distance
    """

    def __init__(self, layout='left', strategy='uniform', max_hop=1, dilation=1):
        self.layout = layout
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return f"Graph({self.layout}, nodes={self.num_node}, A.shape={self.A.shape})"

    def get_edge(self, layout):
        """Define edges based on layout."""
        if layout in ('left', 'right'):
            # MediaPipe Hand: 21 nodes
            self.num_node = 21
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                # Wrist to finger bases
                (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
                # Thumb
                (1, 2), (2, 3), (3, 4),
                # Index
                (5, 6), (6, 7), (7, 8),
                # Middle
                (9, 10), (10, 11), (11, 12),
                # Ring
                (13, 14), (14, 15), (15, 16),
                # Pinky
                (17, 18), (18, 19), (19, 20),
            ]
            self.edge = self_link + neighbor_link
            self.center = 0  # Wrist

        elif layout == 'body':
            # Pose subset: 9 nodes
            # Indices: 0=Nose(1), 1=LEar(7), 2=REar(8), 3=LSh(11), 4=RSh(12),
            #          5=LElb(13), 6=RElb(14), 7=LWrist(15), 8=RWrist(16)
            self.num_node = 9
            self_link = [(i, i) for i in range(self.num_node)]
            # Connections: Nose-LEar, Nose-REar, Nose-LSh, Nose-RSh,
            #              LSh-LElb, LElb-LWrist, RSh-RElb, RElb-RWrist
            neighbor_link = [
                (0, 1),  # Nose - LEar
                (0, 2),  # Nose - REar
                (0, 3),  # Nose - LSh
                (0, 4),  # Nose - RSh
                (3, 5),  # LSh - LElb
                (5, 7),  # LElb - LWrist
                (4, 6),  # RSh - RElb
                (6, 8),  # RElb - RWrist
            ]
            self.edge = self_link + neighbor_link
            self.center = 0  # Nose

        elif layout == 'face':
            # Face subset: 18 nodes
            # 9 Jaw + 8 Inner Mouth + 1 Nose (connected to all)
            # Jaw: nodes 0-8 (sequential)
            # Inner Mouth: nodes 9-16 (closed loop)
            # Nose: node 17 (center, connects to all)
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]

            # Jaw chain: 0-1-2-3-4-5-6-7-8
            jaw_links = [(i, i + 1) for i in range(8)]

            # Inner mouth closed loop: 9-10-11-12-13-14-15-16-9
            mouth_links = [(i, i + 1) for i in range(9, 16)] + [(16, 9)]

            # Nose (17) connects to all other nodes
            nose_links = [(17, i) for i in range(17)]

            neighbor_link = jaw_links + mouth_links + nose_links
            self.edge = self_link + neighbor_link
            self.center = 17  # Nose

        else:
            raise ValueError(f"Unknown layout: {layout}")

    def get_adjacency(self, strategy):
        """Build adjacency matrix based on strategy."""
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A

        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A

        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A

        else:
            raise ValueError(f"Unknown strategy: {strategy}")


def get_hop_distance(num_node, edge, max_hop=1):
    """Compute hop distance matrix."""
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0

    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d

    return hop_dis


def normalize_digraph(A):
    """Normalize adjacency matrix."""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


# Landmark index mappings for data loader
POSE_INDICES = [1, 7, 8, 11, 12, 13, 14, 15, 16]  # 9 nodes
FACE_INDICES = [
    # Jaw (9 points): 454, 288, 365, 378, 152, 149, 136, 58, 234
    454, 288, 365, 378, 152, 149, 136, 58, 234,
    # Inner Mouth (8 points): 13, 14, 78, 308, 81, 178, 311, 402
    13, 14, 78, 308, 81, 178, 311, 402,
    # Nose (1 point): 1
    1
]
