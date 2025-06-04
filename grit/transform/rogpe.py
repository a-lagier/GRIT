# ------------------------ : new rwpse ----------------
from typing import Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_scatter import scatter, scatter_add, scatter_max

from torch_geometric.graphgym.config import cfg

from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)
import torch_sparse
from torch_sparse import SparseTensor


def add_node_attr(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data


def compute_coefficient(adj, deg, a, output_shape, k=2):
    # TODO : batch the process do all nodes at the same time
    """
        adj: adjacency matrix   N x N
        deg: degree array       N x 1
        a: node                 int
    """
    a_index = []
    a_value = []

    neighbors_array = torch.zeros_like(deg)
    neighbors_array[a] = 1

    for k_ in range(k):
        neighbors_array = adj @ neighbors_array
        neighbors_k = torch.nonzero(neighbors_array) # add enumeration (here 2 and 1 in the neighbors_array gives the same computation)?

        degrees_k = deg[neighbors_k].cpu().numpy()

        d, c = np.unique(degrees_k, axis=0, return_counts=True)
        for d_, c_ in zip(d,c):
            a_index.append([k_, d_[0]])
            a_value.append(c_)

    a_index = torch.tensor(a_index, dtype=torch.long).T
    a_value = torch.tensor(a_value, dtype=torch.long)

    # print("a_index", a_index.shape)
    # print(a_index[0,:].max())
    # print(a_index[1,:].max())
    # print(output_shape)
    a_ = SparseTensor.from_edge_index(edge_index=a_index, edge_attr=a_value, sparse_sizes=output_shape)

    return a_


def add_rogpe(data, d_max, k_hop=3, attr_name_coeffs="coeffs"):
    n = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(n, n)
                                       )
    deg = adj.sum(dim=1)
    adj = adj.to_dense()

    n_coeffs = d_max * k_hop
    coeffs = torch.zeros((n,n_coeffs))

    # TODO : to do so complete `compute_coefficient2` in experiments/neighborhood_degree.py
    for i in range(n):
        coeffs[i] = compute_coefficient(adj, deg, i, output_shape=(k_hop, d_max), k=k_hop).to_dense().float().view(-1)

    # print("Number of nan coeffs is :",coeffs.isnan().sum())
    data = add_node_attr(data, coeffs, attr_name=attr_name_coeffs)
    data.deg = deg.type(torch.long)
    data.log_deg = torch.log(deg + 1)

    return data


