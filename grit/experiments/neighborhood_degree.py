import numpy as np
from grit.loader.master_loader import preformat_StochasticBlockModel, preformat_ZINC, preformat_OGB_Graph
from grit.transform.utils import get_node_neighbor_dict_sparse
from torch_geometric.utils import k_hop_subgraph
from sklearn.preprocessing import PolynomialFeatures
import torch
from torch_sparse import SparseTensor


def compute_coefficient(adj, deg, a, output_shape, k=2):
    # TODO : batch the process do all nodes at the same time
    """
        adj: adjacency matrix   N x N
        deg: degree array       N x 1
        a: node                 int
    """
    a_coeffs = torch.zeros(output_shape)

    n = adj.size(0)

    neighbors_array = torch.zeros_like(deg)
    neighbors_array[a] = 1

    for k_ in range(k):
        neighbors_array = adj @ neighbors_array
        neighbors_k = torch.nonzero(neighbors_array) # add enumeration (here 2 and 1 in the neighbors_array gives the same computation)?

        degrees_k = deg[neighbors_k].cpu().numpy()

        d, c = np.unique(degrees_k, axis=0, return_counts=True)
        for d_, c_ in zip(d,c):
            a_coeffs[k_, int(d_)] = c_

    # a_ = SparseTensor.from_edge_index(edge_index=torch.tensor(a_index), edge_attr=torch.tensor(a_value))

    return a_coeffs

def compute_coefficient2(adj, deg, output_shape, k=2):
    """
        adj: adjacency matrix   N x N
        deg: degree array       N x 1
    """

    n = adj.size(0)

    coeffs_ = torch.zeros((n, *output_shape))

    neighbors_array = torch.eye(n)
    for k_ in range(k):
        neighbors_array = adj @ neighbors_array
        neighbors_k = torch.nonzero(neighbors_array)
        neighbors_k[:,1] = deg[neighbors_k[:,1]]

        neighbors_k = neighbors_k.cpu().numpy()

        d, c = np.unique(neighbors_k, axis=0, return_counts=True) # see pandas unique seems to be faster that numpy unique
        # TODO vectorize the loop
        for d_, c_ in zip(d, c):
            i,degree = d_
            coeffs_[i, k_, int(degree)] = int(c_)

    return coeffs_


def k_hop_subgraph(
    node_idx,
    num_hops,
    edge_index,
    relabel_nodes= False,
    num_nodes = None,
    flow = 'source_to_target',
    directed = False,
):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]
    distances = [torch.zeros_like(node_idx)]

    for k_ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])
        distances.append(k_ * torch.ones_like(col[edge_mask]))

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    if not directed:
        edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes, ), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask


def aggregate_k_hop(edge_index, angles, k):
    enhanced_angles = angles
    
    k_paths = edge_index
    row, col = k_paths
    enhanced_angles[row] += angles[col] * (1 / 2.)
    for k_ in range(k - 1):
        edge1_idx, edge2_idx = torch.where(k_paths[1][:, None] == edge_index[0][None, :])

        path_sources = k_paths[0][edge1_idx]
        path_targets = edge_index[1][edge2_idx]
        
        # Stack to create the result tensor
        k_paths = torch.stack([path_sources, path_targets], dim=0)

        # Delete duplicates
        # TODO : take into account the number of duplicates in the aggregation
        k_paths = torch.tensor(np.unique(k_paths.cpu().numpy(), axis=1))

        row, col = k_paths
        enhanced_angles[row] += angles[col] * (1/(k_ + 2))
        print(k_ + 2)
        print(k_paths)
        print(k_paths.shape)
    
    print(enhanced_angles)

n = 10000
k_hop = 3
output_shape = (k_hop, n+1)
# adj = torch.ones((n,n))
edge_index = torch.tensor([[range(n), range(n)], [range(1,n+1), range(2, n+2)]]).reshape(2, 2*n)
print(edge_index)
d = 3
angles = torch.ones((n+2, d))
aggregate_k_hop(edge_index, angles, 4)
# print(np.unique(adj, axis=1))
# adj = torch.tensor([[0,1,1], [1,0,0],[0,1,0]])
# print(adj.nonzero())
# deg = adj.sum(dim=1)


# k = 10
# dataset = preformat_OGB_Graph("../../datasets/", 'ogbg-molbace')

# data = dataset[0]
# n = data.num_nodes
# nodes_idx = torch.arange(n)

# _,edge_index,_,_ = k_hop_subgraph(nodes_idx, k, data.edge_index)

# print(edge_index)

# from time import time
# from tqdm import tqdm

# n_test = 200
# st1 = time()
# for _ in tqdm(range(n_test)):
#     for i in range(n):
#         compute_coefficient(adj, deg, i, output_shape, k=k_hop)
# et1 = time() - st1

# st2 = time()
# for _ in tqdm(range(n_test)):
#     compute_coefficient2(adj, deg, output_shape, k=k_hop)
# et2 = time() - st2

# print(f"Sequential time is {et1 / n_test} sec and parallel time is {et2 / n_test} sec")