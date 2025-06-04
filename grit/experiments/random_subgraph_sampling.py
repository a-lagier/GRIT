import torch
from numpy.random import choice
import networkx as nx
import matplotlib.pyplot as plt
from grit.transform.utils import get_node_neighbor_dict_sparse

def normalize(p):
    cum_sum = 0
    for v in p.keys():
        cum_sum += p[v]
    
    for v in p.keys():
        p[v] /= cum_sum

    return p

def compute_distribution(distances, exponent=1):
    return normalize({p: 1/(distances[p] + 1)**exponent for p in distances.keys()})

def random_subgraph_sampling(edge_dict, root, n_iter=20, undirected=True):
    """
    Random Subgraph Sampling algorithm computes a subgraph from a single root node `root`

    Iterative algorithm that augments the subgraph by successively sampling node of the subgraph and edge

    Returns random subgraph S = (V, E)
    """


    # Initialize distance and probability distribution
    distances = {root: 0}
    p = {root: 1.}

    S_edges = set()

    for _ in range(n_iter):
        v = choice(list(p.keys()), p=list(p.values()))

        if edge_dict[v]:
            w = choice(edge_dict[v])

            if w in p.keys():
                distances[w] = min(distances[w], distances[v] + 1)
            else:
                distances[w] = distances[v] + 1

            S_edges.add((v,w))
            if undirected:
                S_edges.add((w,v))
    
        p = compute_distribution(distances)
    
    return p.keys(), S_edges


def display_graph_from_edge_index(N, edge_index, subgraph, title="Graph Visualization"):
    G = nx.Graph() 

    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G, k=1, iterations=50)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # First graph display
    plt.sca(ax1)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2, ax=ax1)
    nx.draw_networkx_labels(G, pos)
    
    ax1.set_title(f"{title}")
    ax1.axis('off')
    

    # Second graph display
    V,E = subgraph
    
    plt.sca(ax2)
    
    G_subgraph = G.copy()
    
    node_color = [p in V for p in pos.keys()]

    nx.draw_networkx_nodes(G_subgraph, pos, node_color=node_color,
                          node_size=500, alpha=0.8, ax=ax2)
    nx.draw_networkx_labels(G_subgraph, pos)
    
    for i, (u, v) in enumerate(edges):
        nx.draw_networkx_edges(G_subgraph, pos, edgelist=[(u, v)], 
                              edge_color="r" if (u,v) in E else "b", alpha=0.6, ax=ax2)
    
    ax2.set_title(f"{title} - Subgraph")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# hard coded 1st graph of ZINC-12K dataset
N = 29
edge_index = torch.tensor([[ 0,  1,  1,  2,  2,  2,  3,  3,  4,  4,  5,  5,  5,  6,  6,  7,  7,  8,
          8,  8,  9, 10, 10, 10, 11, 11, 12, 12, 12, 13, 13, 14, 14, 15, 15, 15,
         16, 16, 16, 16, 17, 18, 19, 19, 19, 20, 20, 21, 21, 21, 22, 23, 23, 24,
         24, 25, 25, 26, 26, 27, 27, 27, 28, 28],
        [ 1,  0,  2,  1,  3, 28,  2,  4,  3,  5,  4,  6, 27,  5,  7,  6,  8,  7,
          9, 10,  8,  8, 11, 27, 10, 12, 11, 13, 26, 12, 14, 13, 15, 14, 16, 25,
         15, 17, 18, 19, 16, 16, 16, 20, 24, 19, 21, 20, 22, 23, 21, 21, 24, 19,
         23, 15, 26, 12, 25,  5, 10, 28,  2, 27]]).numpy()
edge_dict = get_node_neighbor_dict_sparse(edge_index, N)

root = 0
S = random_subgraph_sampling(edge_dict, root)
display_graph_from_edge_index(N, edge_index, S)