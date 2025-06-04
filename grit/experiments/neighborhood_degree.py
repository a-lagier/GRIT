import numpy as np
from grit.transform.utils import get_node_neighbor_dict_sparse
from sklearn.preprocessing import PolynomialFeatures
import torch
from torch_sparse import SparseTensor


def compute_coefficient(adj, deg, a, k=2):
    # TODO : batch the process do all nodes at the same time
    """
        adj: adjacency matrix   N x N
        deg: degree array       N x 1
        a: node                 int
    """
    a_index = []
    a_value = []

    n = adj.size(0)

    neighbors_array = torch.zeros_like(deg, dtype=torch.int)
    neighbors_array[a] = 1

    for k_ in range(k):
        neighbors_array = adj @ neighbors_array
        neighbors_k = torch.nonzero(neighbors_array) # add enumeration (here 2 and 1 in the neighbors_array gives the same computation)?

        degrees_k = deg[neighbors_k].cpu().numpy()

        d, c = np.unique(degrees_k, axis=0, return_counts=True)
        for d_, c_ in zip(d,c):
            a_index.append([k_, d_[0]])
            a_value.append(c_)

    a_ = SparseTensor.from_edge_index(edge_index=torch.tensor(a_index), edge_attr=torch.tensor(a_value))

    return a_

def compute_coefficient2(adj, deg, nodes, k=2):
    """
        adj: adjacency matrix   N x N
        deg: degree array       N x 1
        a: node                 int
    """

    m = nodes.size(0)
    n = adj.size(0)

    neighbors_array = torch.zeros((n,m), dtype=torch.int)
    for (i,a) in enumerate(nodes):
        neighbors_array[a,i] = 1

    deg_mat = deg.view(-1,1).repeat_interleave(m, dim=1)

    for k_ in range(k):
        neighbors_array = adj @ neighbors_array
        pass
    # TODO : finish implementing batched process of compute_coefficient

    return

n = 100
adj = torch.ones((n,n), dtype=torch.int)
deg = adj.sum(dim=1)
import torch

# Example tensors
tensor_n = torch.tensor([2.0, 3.0, 4.0])  # n elements
vector_k = torch.tensor([1.0, 2.0, 3.0, 4.0])  # k elements

# Reshape for broadcasting: vector_k as column vector, tensor_n as row vector
result = torch.pow(tensor_n[None, :], vector_k[:, None])
print(result)
# print(compute_coefficient(adj, deg, 2))