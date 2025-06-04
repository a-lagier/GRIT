import numpy as np
from grit.transform.mmsbm import MMSBM_SGMCMC, flags
from grit.loader.master_loader import preformat_StochasticBlockModel, preformat_ZINC, preformat_OGB_Graph
from grit.transform.rrwp_mmsbm import add_full_rrwp_mmsbm, get_mmsbm_enc
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch

@torch.no_grad()
def edge_mmsbm_coeff(pi_a, pi_b, beta):
    return (pi_a * beta).dot(pi_b)

def display_graph_from_edge_index(edge_index, edge_weight=None, pi=None, title="Graph Visualization"):
    # Create a NetworkX graph
    G = nx.Graph()  # Use nx.DiGraph() for directed graphs
    
    # Add edges from the edge_index array
    # edge_index should be shape (2, num_edges)
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    # Choose a layout algorithm (same for both graphs if weights are provided)
    pos = nx.spring_layout(G, k=1, iterations=50)
    # Display two graphs side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Left plot: Graph without weights
    plt.sca(ax1)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2, ax=ax1)
    
    ax1.set_title(f"{title} - Unweighted")
    ax1.axis('off')
    

    # Right plot: Graph with weights
    plt.sca(ax2)
    if edge_weight is not None:
        # Add weights to the graph
        G_weighted = G.copy()
        for i, (u, v) in enumerate(edges):
            G_weighted[u][v]['weight'] = edge_weight[i]
        
        # Draw nodes
        nx.draw_networkx_nodes(G_weighted, pos, node_color=pi.argmax(axis=1), 
                            node_size=500, alpha=0.8, ax=ax2)
        
        # Draw edges with varying thickness based on weights
        # Normalize weights for edge thickness
        min_weight = np.min(edge_weight)
        max_weight = np.max(edge_weight)
        normalized_weights = (edge_weight - min_weight) / (max_weight - min_weight) if max_weight != min_weight else np.ones_like(edge_weight)
        edge_widths = 1 + normalized_weights * 4  # Width between 1 and 5
        
        for i, (u, v) in enumerate(edges):
            nx.draw_networkx_edges(G_weighted, pos, edgelist=[(u, v)], 
                                width=edge_widths[i], alpha=0.6, ax=ax2)
        
        # Add edge weight labels
        edge_labels = {}
        for i, (u, v) in enumerate(edges):
            edge_labels[(u, v)] = f'{edge_weight[i] / max_weight:.2f}'
        nx.draw_networkx_edge_labels(G_weighted, pos, edge_labels, font_size=8, ax=ax2)
        
        ax2.set_title(f"{title} - Weighted")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return G_weighted

dataset = preformat_OGB_Graph("../../datasets/", 'ogbg-molbace')

data = dataset[0]
edge_index = data.edge_index

# print(edge_index)
# exit(-1)

# pi, beta = get_mmsbm_enc(data, n_communities=3)

# edge_weight = data.edge_weight
# if edge_weight is None:
#     edge_weight = torch.zeros(edge_index.size(1), dtype=torch.float)

# # TODO : parallelize the loop
# for i in range(edge_index.size(1)):
#     a,b = edge_index[:,i]
#     edge_weight[i] = edge_mmsbm_coeff(pi[a], pi[b], beta)
# print(pi)
# for i in range(pi.shape[0]):
#     plt.plot(pi[i])
display_graph_from_edge_index(edge_index.numpy(), edge_weight=None, pi=None)
