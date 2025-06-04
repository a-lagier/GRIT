# ------------------------ : new rwpse ----------------
from typing import Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter, scatter_add, scatter_max
from grit.transform.mmsbm import MMSBM_SGMCMC, flags

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


@torch.no_grad()
def get_mmsbm_enc(data, n_communities=8):
    FLAGS = flags()
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    model = MMSBM_SGMCMC(flags=FLAGS, n=num_nodes, k=n_communities, edges=edge_index, step_size_scalar=1)
    model.model_training()
    pi, beta = torch.from_numpy(model.pi), torch.from_numpy(model.beta)

    del model

    return pi, beta

@torch.no_grad()
def edge_mmsbm_coeff(pi_a, pi_b, beta):
    return (pi_a * beta).dot(pi_b)

@torch.no_grad()
def add_full_rrwp_mmsbm(data,
                  walk_length=8,
                  n_communities=8,
                  attr_name_abs="rrwp", # name: 'rrwp'
                  attr_name_rel="rrwp", # name: ('rrwp_idx', 'rrwp_val')
                  add_identity=True,
                  spd=False,
                  **kwargs
                  ):
    device=data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    if edge_weight is None:
        edge_weight = torch.zeros(edge_index.size(1), dtype=torch.float)

    pi, beta = get_mmsbm_enc(data, n_communities=n_communities)

    # TODO : parallelize the loop
    for i in range(edge_index.size(1)):
        a,b = edge_index[:,i]
        edge_weight[i] = edge_mmsbm_coeff(pi[a], pi[b], beta)
    
    del pi
    del beta

    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes),
                                       )
    del edge_weight

    # Compute D^{-1} A:
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1) # view(-1,1)
    adj = adj.to_dense()

    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1

    out = adj
    pe_list.append(adj)

    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)

    pe = torch.stack(pe_list, dim=-1) # n x n x k

    abs_pe = pe.diagonal().transpose(0, 1) # n x k

    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    # rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
    rel_pe_idx = torch.stack([rel_pe_col, rel_pe_row], dim=0)
    # the framework of GRIT performing right-mul while adj is row-normalized, 
    #                 need to switch the order or row and col.
    #    note: both can work but the current version is more reasonable.


    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)

    return data

