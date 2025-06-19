import numpy as np
import torch
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter
from grit.loader.master_loader import preformat_StochasticBlockModel, preformat_ZINC, preformat_OGB_Graph
import matplotlib.pyplot as plt


def aggregate_k_hop(edge_index, angles, k, alpha=2.):
    enhanced_angles = angles.clone()

    f = lambda k : np.log(k + 1.)
    
    k_paths = edge_index
    row, col = k_paths
    enhanced_angles += scatter(angles[col] * f(1), row, dim=0)

    for k_ in range(2, k + 1):
        edge1_idx, edge2_idx = torch.where(k_paths[1][:, None] == edge_index[0][None, :])

        path_sources = k_paths[0][edge1_idx]
        path_targets = edge_index[1][edge2_idx]
        
        k_paths = torch.stack([path_sources, path_targets], dim=0)

        # Delete duplicates
        k_paths = torch.tensor(np.unique(k_paths.cpu().numpy(), axis=1))
        
        if torch.cuda.is_available():
            k_paths = k_paths.cuda()

        row, col = k_paths
        enhanced_angles += scatter(angles[col] * f(k), row, dim=0)

    return enhanced_angles

dataset = preformat_OGB_Graph("../../datasets/", 'ogbg-molbace')

data = dataset[1]


def display_grids(batch):
    n = data.num_nodes
    d_angles = 10
    k = 5

    adj = to_dense_adj(data.edge_index).view(n,n)

    adj_k = adj.clone()
    for _  in range(k-1):
        adj_k += adj_k @ adj

    std = 0.5
    angles = torch.normal(0., std, size=(n, d_angles))


    def similarity_func(X, sim_func="dot"):
        # X of dimension (number of nodes, dim_angles)
        if sim_func == "dot":
            return X @ X.transpose(0,1)
        elif sim_func == "norm":
            norm = X.pow(2).sum(dim=1)
            norm_1 = norm.view(-1, 1).repeat_interleave(n, dim=-1)
            return norm_1 + norm_1.transpose(0,1) -2 * (X @ X.transpose(0,1))


    angles_similarity = similarity_func(angles, sim_func="norm")

    alpha = 1/10.
    enhanced_angles_10 = aggregate_k_hop(data.edge_index, angles, k=k, alpha=alpha)
    enhanced_angles_10_similarity = similarity_func(enhanced_angles_10, sim_func="norm")

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].imshow(adj)
    ax[0,0].set_title("Adjacency matrix")

    ax[1,0].imshow(angles_similarity)
    ax[1,0].set_title("Angles similarity")

    ax[0,1].imshow(adj_k)
    ax[0,1].set_title(f"Adjacency matrix with k={k}")

    ax[1,1].imshow(enhanced_angles_10_similarity)
    ax[1,1].set_title(f"Enhanced angles similarity k={k}")
    plt.show()
