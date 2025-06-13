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


def add_deg(data):
    n = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(n, n)
                                       )
    deg = adj.sum(dim=1)

    data.deg = deg.type(torch.long)
    data.log_deg = torch.log(deg + 1)

    return data


def compute_structure_coefficients(data, cfg):
    """
        Structure coefficients : count the number of nodes of degree d at distance k
    """


    # fetch number of coefficient to compute
    output_shape = (cfg.posenc_ROGPE.coeffs.k_hop, cfg.max_degree)
    n_coeffs = output_shape[0] * output_shape[1]
    cfg.posenc_ROGPE.coeffs.n_coeffs = n_coeffs

    n = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(n, n)
                                       )
    deg = adj.sum(dim=1)
    adj = adj.to_dense()

    # compute coefficients
    coeffs_ = torch.zeros((n, *output_shape))
    neighbors_array = torch.eye(n)
    for k_ in range(k):
        neighbors_array = adj @ neighbors_array
        neighbors_k = torch.nonzero(neighbors_array)
        neighbors_k[:,1] = deg[neighbors_k[:,1]]

        neighbors_k = neighbors_k.cpu().numpy()

        d, c = np.unique(neighbors_k, axis=0, return_counts=True)
        # TODO vectorize the loop
        for d_, c_ in zip(d, c):
            i,degree = d_
            coeffs_[i, k_, int(degree)] = int(c_)

    return coeffs_


def compute_distance_coefficients(data, cfg):
    """
        Distance coefficient : compute the number of nodes at distance k_ <= k for every node
    """

    # fetch number of coefficient to compute
    n = data.num_nodes
    k = cfg.posenc_ROGPE.coeffs.k_hop

    output_shape = (k,)
    cfg.posenc_ROGPE.coeffs.n_coeffs = k

    edge_index, edge_weight = data.edge_index, data.edge_weight
    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(n, n)
                                       ).to_dense().bool()

    # compute the coefficients
    coeffs_ = torch.zeros(n,*output_shape)
    for k_ in range(k):
        coeffs_[:,k_] = adj.sum(dim=1)
        adj = adj.float() @ adj.float()
        adj = adj.bool()

    return coeffs_

def compute_random_walk_coefficients(data, cfg):
    """
        Random walk coefficient : compute the random walk coefficient based on RRWP positional encoding
    """
   # fetch number of coefficient to compute
    n = data.num_nodes
    walk_length = cfg.posenc_ROGPE.coeffs.k_hop

    output_shape = (walk_length,)
    cfg.posenc_ROGPE.coeffs.n_coeffs = walk_length

    edge_index, edge_weight = data.edge_index, data.edge_weight
    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(n, n)
                                       )
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float('inf')] = 0

    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()

    # compute the coefficients
    pe_list = [torch.eye(n), adj]
    out = adj
    for j in range(2, walk_length):
        out = out @ adj
        pe_list.append(out)

    coeffs_ = torch.stack(pe_list, dim=-1).diagonal().transpose(0,1)

    return coeffs_
    

def add_rogpe(data, coeff_function, cfg, attr_name_coeffs="coeffs"):
    """
        coeff_function takes as input data and cfg and return a (n x M) tensor
    """

    coeffs = coeff_function(data, cfg)

    data = add_node_attr(data, coeffs, attr_name=attr_name_coeffs)
    data = add_deg(data)

    return data