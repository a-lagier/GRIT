import numpy as np
from grit.transform.mmsbm import MMSBM_SGMCMC, flags
from grit.loader.master_loader import preformat_StochasticBlockModel
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

def display_graph_from_edge_index(edge_index, pi=None, node_labels=None, title="Graph Visualization"):
    G = nx.Graph()
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)

    if pi is None:
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8)
    else:
        cluster_number = np.argmax(pi, axis=1)
        nx.draw_networkx_nodes(G, pos, node_color=cluster_number, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, alpha=0.6, width=2)

    if node_labels:
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
    else:
        nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    return G

def get_node_neighbor_dict(edges, N):
    node_neighbors_dict = {i:[] for i in range(N)}
    for i in range(edges.shape[1]):
        a,b = edges[:,i]
        node_neighbors_dict[a].append(b)
    return node_neighbors_dict


def load_sbm_dataset(block_sizes=[20,20], edge_probs=[[0.49,1e-2],[1e-2,0.49]]):
    dataset_dir = './datasets'
    dataset = preformat_StochasticBlockModel(dataset_dir, block_sizes, edge_probs, no_split=True)

    n = sum(block_sizes)
    k = len(block_sizes)
    edges = dataset.edge_index.numpy()
    neighbors_dict = get_node_neighbor_dict(edges, n)

    mu = 100
    FLAGS = flags()
    FLAGS.max_itr = 1
    model = MMSBM_SGMCMC(FLAGS, n, k, edges, step_size_scalar=1, node_neighbors_dict=neighbors_dict, mu=mu)
    model.model_training()
    pi, beta = model.pi, model.beta

    print(pi, beta)

    prior_clusters = dataset.y.to(int)
    print(f"The accuracy of SBM clustering is {cluster_accuracy(prior_clusters, pi)}")
    display_graph_from_edge_index(edges, pi)
    return

def cluster_accuracy(prior_clusters, pi):
    k = pi.shape[1]

    predicted_clusters = np.argmax(pi, axis=1)
    distrib_clusters = np.zeros((k,k))
    for i in range(k):
        prior_cluster_nodes = prior_clusters == i

        nel = prior_cluster_nodes.sum(axis=0)
        c = predicted_clusters[prior_cluster_nodes]
        for j in range(k):
            distrib_clusters[i,j] = (c == j).sum() / nel
    
    m = 0
    for i in range(k):
        for j in range(i+1, k):
            m += np.abs(distrib_clusters[i].dot(distrib_clusters[j]))
    return 1 - m * (2 / (k*(k-1)))


def load_four_triangles_graph():
    n = 13
    k = 4
    edges = np.array([[1,1,2,2,3,3,3,4,4,5,5,6,6,6,7,7,8,8,9,9,9,10,10,11,11,12,12,12,0,0,0,0], [2,3,1,3,1,2,0,5,6,4,6,4,5,0,8,9,7,9,7,8,0,11,12,10,12,10,11,0,3,6,9,12]])
    neighbors_dict = get_node_neighbor_dict(edges, n)
    node_clusters = [[1,2,3], [4,5,6], [7,8,9], [10,11,12]]

    return n, k, edges, neighbors_dict, node_clusters

def load_two_triangles_graph():
    n = 6
    k = 2
    edges = np.array([[0,0,1,1,2,2,5,5,3,3,4,4,5,2], [1,2,0,2,0,1,3,4,5,4,5,3,2,5]])
    neighbors_dict = get_node_neighbor_dict(edges, n)
    node_clusters = [[0, 1, 2], [3, 4, 5]]

    return n, k, edges, neighbors_dict, node_clusters

def evaluate_clusters(pi, k, node_clusters):
    # TODO incorporate beta evaluation

    best_cluster_per_node = np.argmax(pi, axis=1)
    best_cluster_per_evaluated_cluster = np.zeros(k, dtype=np.int32)

    for cluster in range(k):
        if not best_cluster_per_node[cluster]:
            pass

        best_cluster_per_evaluated_cluster[cluster] = best_cluster_per_node[node_clusters[cluster][0]]
        for node in node_clusters[cluster]:
            # check for unicity of best cluster in a single predefined cluster
            if best_cluster_per_evaluated_cluster[cluster] != best_cluster_per_node[node]:
                return False
    
    # check for unicity of best cluster among all best cluster per evaluated cluster
    _, c = np.unique(best_cluster_per_evaluated_cluster, return_counts=True)
    return not np.any(c > 1)

def experiment():
    max_itrs = [50]
    mus = [1]
    n_tests = 500
    FLAGS = flags()

    n, k, edges, neighbors_dict, node_clusters = load_four_triangles_graph()

    for mu in mus:
        for max_itr in max_itrs:
            FLAGS.max_itr = max_itr
            success = 0
            for _ in tqdm(range(n_tests)):
                model = MMSBM_SGMCMC(FLAGS, n, k, edges, step_size_scalar=1, node_neighbors_dict=neighbors_dict, mu=mu)
                model.model_training()
                pi, beta = model.pi, model.beta
                is_successful = evaluate_clusters(pi, k, node_clusters)
                if is_successful:
                    success += 1
            print(f" > With mu = {mu} and max_itr = {FLAGS.max_itr}, the success rate is {success / n_tests}")

k = 2
delta = 1e-4
p = 0.9
block_sizes = [20 for _ in range(k)]
edge_probs = np.ones((k,k)) * delta
for i in range(k):
    edge_probs[i,i] = p - (k - 1)*delta

load_sbm_dataset(block_sizes=block_sizes, edge_probs=edge_probs)