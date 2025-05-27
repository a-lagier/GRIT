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


def sample_non_edges(hash_edges, N, N_edges, ratio):
    # st = time()
    ratio += (N_edges + 1) / N
    N_non_edges = int(ratio*N)
    non_edges_a_indices = np.random.randint(low=0, high=N, size=N_non_edges)
    non_edges_b_indices = np.random.randint(low=0, high=N, size=N_non_edges)
    
    accepted_samples = np.ones(N_non_edges, dtype=bool)

    # TODO : parallelize the edge rejecting sampling (executed O(iter) times)
    for i in range(N_non_edges):
        a,b = non_edges_a_indices[i], non_edges_b_indices[i]
        if a==b or hash_edges.setdefault((a,b), False):
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


def get_node_neighbor_dict(edges, N):
    node_neighbors_dict = {}

    # TODO : parallelize the loop (not important because only executed once)
    for i in range(edges.shape[1]):
        a,b = edges[:,i]
        node_neighbors_dict.setdefault(a, []).append(b)
    return node_neighbors_dict

# Method from Zhang et al. uses N^2 memory
# def get_node_neighbor_dict(adj, N):
#     node_neighbors_dict = {}
#     for i in range(N):
#         node = adj[i]
#         node_neighbors_dict[i] = csr_matrix.nonzero(node)[1]
#     return node_neighbors_dict


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
