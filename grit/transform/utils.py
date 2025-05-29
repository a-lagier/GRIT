"""
    Copyright (C) 2019. Huawei Technologies Co., Ltd and McGill University. All rights reserved.
    This program is free software; you can redistribute it and/or modify
    it under the terms of the MIT License.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    MIT License for more details.
    
"""
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import accuracy_score
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import os, sys
from time import time


def bernuli_distrbution(y, p):
    bernuli_dist = (p**y)*((1-p)**(1-y))
    return bernuli_dist


def edges_non_edges_index(adj, N, node_neighbors_dict):
    A_coo = coo_matrix(adj)
    A_coo_data = A_coo.data
    diagnol_element = np.array([adj[i,i] for i in range(N)])
    self_loop_index = np.where(diagnol_element == 1)[0]
    self_loop_n = len(self_loop_index)
    links_n = len(np.where(A_coo_data != 0)[0]) - self_loop_n
    links_n = int(links_n/2)
    non_links_n = int((N * N - N)/2 - links_n)

    nonedges_index_a = np.zeros(non_links_n).astype(int)
    nonedges_index_b = np.zeros(non_links_n).astype(int)

    edges_index_a = np.zeros(links_n).astype(int)
    edges_index_b = np.zeros(links_n).astype(int)

    N_list_set = np.array([i for i in range(N)])

    start_edges = 0
    start_non_edges = 0
    for i in range(N):
        # deal with links
        node_i_neighbors = node_neighbors_dict[i]
        node_i_upper_tri_index = np.arange(i+1 , N)
        node_i_neighbors_upper_tri = np.intersect1d(node_i_neighbors, node_i_upper_tri_index)
        end_edges = start_edges + len(node_i_neighbors_upper_tri)
        edges_index_a[start_edges:end_edges] = i
        edges_index_b[start_edges:end_edges] = node_i_neighbors_upper_tri

        start_edges = end_edges

        # deal with non-links
        node_i_non_neighbor = np.setdiff1d(N_list_set, node_i_neighbors)
        node_i_non_neighbor_tri = np.intersect1d(node_i_non_neighbor, node_i_upper_tri_index)
        node_i_non_neighbor_n = len(node_i_non_neighbor_tri)

        end_non_edges = start_non_edges + node_i_non_neighbor_n
        nonedges_index_a[start_non_edges:end_non_edges] = i
        nonedges_index_b[start_non_edges:end_non_edges] = node_i_non_neighbor_tri
        start_non_edges = end_non_edges

    nonedges = (nonedges_index_a, nonedges_index_b)
    edges = (edges_index_a, edges_index_b)
    return edges, nonedges


def sample_non_edges(edges_dict, N, N_edges, ratio):
    # st = time()
    ratio += (N_edges + 1) / N
    N_non_edges = int(ratio*N)
    non_edges_a_indices = np.random.randint(low=0, high=N, size=N_non_edges)
    non_edges_b_indices = np.random.randint(low=0, high=N, size=N_non_edges)
    
    accepted_samples = np.ones(N_non_edges, dtype=bool)

    # TODO : parallelize the edge rejecting sampling (executed O(iter) times)
    for i in range(N_non_edges):
        a,b = non_edges_a_indices[i], non_edges_b_indices[i]
        if a==b or edges_dict.setdefault((a,b), False):
            accepted_samples[i] = False
    
    N_non_edges = int(accepted_samples.sum())
    non_edges_a_indices = non_edges_a_indices[accepted_samples]
    non_edges_b_indices = non_edges_b_indices[accepted_samples]

    non_edges = np.stack([non_edges_a_indices, non_edges_b_indices], axis=0)
    # print(f"Sampling {N_non_edges} edges took {time() - st} sec")
    return non_edges, N_non_edges


def reparameterized_to_beta(theta):  # comunity strength
    theta_constant = np.sum(theta, axis = 1)
    beta = theta[:,1]/theta_constant
    return beta, theta_constant


def initialize_theta_phi_with_better_initialization(beta, pi, theta_constant, phi_constant, K):

    phi = pi*phi_constant
    theta = np.zeros((K, 2))
    theta[:, 1] = theta_constant * beta
    theta[:, 0] = theta_constant - theta[:, 1]
    return theta, phi


def reparameterized_to_pi(phi, N):  #
    row_sum_phi = (np.sum(phi, axis = 1)).reshape(N, 1)
    pi = phi/row_sum_phi
    return pi, row_sum_phi


def step_size_function(itr_index, tao, scaler):
    step_size = (tao + itr_index) ** (-0.5)
    return step_size/scaler


def metric_perp_avg(beta_samples, pi_samples, test_edge_set, y_test, delta):
    sum_edges_perplexity = 0
    for edge in range(len(test_edge_set)):
        a = test_edge_set[edge][0]
        b = test_edge_set[edge][1]
        p_edge = 0
        for i in range(min(20, len(beta_samples))):
            p_edge += z_constant(beta_samples[i], pi_samples[i][a], pi_samples[i][b], y_test[edge], delta)
        sum_edges_perplexity += np.log(1.0/min(20, len(beta_samples)) * p_edge)


    perplexity = np.exp(-sum_edges_perplexity / len(test_edge_set))

    return perplexity


def get_node_neighbor_dict_sparse(edges, N):
    node_neighbors_dict = {}

    # TODO : parallelize the loop (not important because only executed once)
    for i in range(edges.shape[1]):
        a,b = edges[:,i]
        node_neighbors_dict.setdefault(a, []).append(b)
    return node_neighbors_dict

# Method from Zhang et al. uses N^2 memory
def get_node_neighbor_dict_dense(adj, N):
    node_neighbors_dict = {}
    for i in range(N):
        node = adj[i]
        node_neighbors_dict[i] = csr_matrix.nonzero(node)[1]
    return node_neighbors_dict


def accuracy_avg(pi_samples, initial_prediction_labels, true_label, N, K, val_set_index):
    count = np.zeros((N, K))
    for i in range(len(pi_samples)):
        predict_label = pi_samples[i].argmax(axis=1)
        labels_one_hot = np.zeros(pi_samples[i].shape)
        for j in range(len(predict_label)):
            labels_one_hot[j][predict_label[j]] = 1

        if i == 0:
            count = labels_one_hot
        else:
            count += labels_one_hot

    avg_predict_label = count.argmax(axis=1)
    ARI = adjusted_rand_score(true_label[val_set_index], avg_predict_label[val_set_index])
    acc = accuracy_score(true_label[val_set_index], avg_predict_label[val_set_index])
    change_from_the_initial_prediction_labels = len(np.where(initial_prediction_labels == avg_predict_label)[0])/float(N)
    return ARI, acc, change_from_the_initial_prediction_labels, avg_predict_label
