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
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix

from torch_geometric.graphgym.config import cfg
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


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=0.)
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=0.).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs

def compute_structure_coefficients(data, cfg):
    """
        Structure coefficients : count the number of nodes of degree d at distance k

        Invalid : the number of coefficient vary depending on the graph
    """


    # fetch number of coefficient to compute
    k = cfg.posenc_ROGPE.coeffs.k_hop
    output_shape = (k, cfg.max_degree)
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

    coeffs_ = coeffs_.view(-1, n_coeffs)

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

def compute_eigen_coefficients(data, cfg):
    """
        Eigen coefficients : compute eigenvectors/values
    """
    # fetch number of coefficient to compute
    n = data.num_nodes
    d = cfg.posenc_ROGPE.coeffs.k_hop # to be changed

    output_shape = (2*d,)
    cfg.posenc_ROGPE.coeffs.n_coeffs = 2*d

    if data.is_undirected():
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization="sym", num_nodes=n)
    )
    evals_sn, evects_sn = np.linalg.eigh(L.toarray())
    eigvals, eigvecs = get_lap_decomp_stats(
        evals=evals_sn, evects=evects_sn,
        max_freqs=d)

    eigvals = eigvals.reshape(eigvecs.shape)

    coeffs_ = torch.cat([eigvecs, eigvals], dim=1)

    coeffs_[coeffs_.isnan()] = 0.

    return coeffs_

    

def add_rogpe(data, coeff_function, cfg, attr_name_coeffs="coeffs"):
    """
        coeff_function takes as input data and cfg and return a (n x M) tensor
    """

    coeffs = coeff_function(data, cfg)

    data = add_node_attr(data, coeffs, attr_name=attr_name_coeffs)
    data = add_deg(data)

    return data