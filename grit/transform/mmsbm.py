"""
    Copyright (C) 2019. Huawei Technologies Co., Ltd and McGill University. All rights reserved.
    This program is free software; you can redistribute it and/or modify
    it under the terms of the MIT License.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    MIT License for more details.
    """

import torch
import numpy as np
from grit.transform.utils import reparameterized_to_beta, reparameterized_to_pi, metric_perp_avg, \
    bernuli_distrbution, step_size_function, accuracy_avg, initialize_theta_phi_with_better_initialization, sample_non_edges, \
    get_node_neighbor_dict_dense, get_node_neighbor_dict_sparse, edges_non_edges_index
from scipy.sparse import coo_matrix
import networkx as nx
from time import time
from math import ceil

def add_node_attr(data, value,
                  attr_name):
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value

    return data

# TODO move these parameters into the config file
class flags:
    def __init__(self):
        # Hyper-parameter for the MMSBM model
        self.max_itr = 200
        self.delta = 0.0001
        self.batch_size = 500
        self.sampled_non_edges_ratio = 0.01
        self.gamma_scale = 0.001


class MMSBM_SGMCMC:
    def __init__(self, flags, n, k, edges, step_size_scalar, mu=1, max_iter=100):

        """ follows the notations in the original paper
        :param flags: hyper-parameters for GCN and MMSBM
        :param n: node number
        :param k: class number
        :param edges: edge indices
        :param nonedges: non-edge indices                                                                                                           (unused thus removed)

        :param beta_prior: prior for the community strength                                                                                         (unused thus removed)
        :param membership_prior: prior for the membership                                                                                           (unused thus removed)
        :param theta_constant: re-parameterization constant for community strength beta                                                             (unused thus removed)
        :param phi_constant: re-parameterization constant for membership                                                                            (unused thus removed)
        :param true_labels: ground truth labels                                                                                                     (unused thus removed)
        :param better_initialization_flag: a flag indicate if we train the MMSBM from scratch or we use the better initialization output from gcn   (unused thus removed)
        :param step_size_scalar: step size for the MMSBM
        :param node_neighbors_dict: a dict for query the neighborhood node indices                                                                  (computed inside __init__)
        :param val_set_index: indices for the validation set                                                                                        (unused thus removed)
        """
        self.gamma_scale = flags.gamma_scale
        self.step_size_scalar = step_size_scalar
        self.flags = flags
        self.n = n  # number of nodes
        self.k = k

        # self.dense = self.n <= 100
        self.dense = False
        if self.dense:
            self.adj = np.zeros((n,n), dtype=int)
            self.adj[edges[0,:], edges[1,:]] = 1
            self.adj = nx.adjacency_matrix(nx.from_numpy_array(self.adj))
        else:
            self.adj = None
        self.edges = edges

        self.alpha = 1.0 / k
        self.mu = mu
        self.tao = 1024
        self.n_list_set = np.array([i for i in range(self.n)])

        # Flags hyperparameters
        self.max_iter = flags.max_itr
        self.mini_batch_nodes = min(flags.batch_size, max(min(20, n), n // 10))

        self.sample_n = 20  # sample size for update each local parameters
        self.T = 1  # sample number of pi and beta for each edge during the evaluation process
        self.test_edges_n = 500  # test set edges for the perplexity test
        self.delta = flags.delta
        self.node_neighbors_dict = (get_node_neighbor_dict_dense(self.adj, self.n) if self.dense else get_node_neighbor_dict_sparse(edges, self.n))
        self.avg_predict_label = 0

        # variable initialization (random initialization)
        self.phi = np.random.gamma(self.alpha, 1, size=(self.n, self.k))
        self.theta = np.random.gamma(self.mu, 1, size=(self.k, 2))
        self.beta, self.theta_constant = reparameterized_to_beta(self.theta)
        self.pi, self.phi_constant = reparameterized_to_pi(self.phi, self.n)
        self.initial_prediction_labels = self.pi.argmax(axis=1)

        self.MCMC_MMSBM_prediction_labels = self.initial_prediction_labels
        self.B = np.ones((self.k, self.k)) * flags.delta

        # Info of the given topology, split into the edges and non-edges
        if self.dense:
            self.edges, self.nonedges = edges_non_edges_index(self.adj, self.n, self.node_neighbors_dict)
            self.deg = self.adj.sum(axis=1)
            self.edges_n, self.nonedges_n = int(self.edges.shape[0]), int(self.nonedges.shape[0])
        else:
            self.deg = np.zeros(n, dtype=int)
            self.edges_n, self.nonedges_n = self.edges.shape[1], self.n * self.n - self.edges.shape[1]
            self.edges_dict = {}
            for i in range(self.edges_n): #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! not parallel !!!!!!!!!!!!!!!!!!!!!!
                self.deg[self.edges[0,i]] += 1
                self.edges_dict[tuple(self.edges[:,i])] = True

        self.sampled_non_edges_ratio = self.flags.sampled_non_edges_ratio
        self.sampled_non_edges_n = int(self.sampled_non_edges_ratio * self.nonedges_n)
        self.dir = 'figures/'

    def Z_constant_mini_batch_phi(self, node_a_membership, node_b_membership, link_index_mini_batch):
        bernuli_delta_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.delta)
        bernuli_delta_mini_batch[link_index_mini_batch] = self.delta
        Z_constant_mini_batch = bernuli_delta_mini_batch.copy()
        for k in range(self.k):
            bernuli_beta_k_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.beta[k])
            bernuli_beta_k_mini_batch[link_index_mini_batch] = self.beta[k]
            pi_a_k = (node_a_membership[:, k]).reshape(len(node_a_membership), 1)
            pi_b_k = (node_b_membership[:, k]).reshape(len(node_b_membership), 1)
            Z_constant_mini_batch += (bernuli_beta_k_mini_batch - bernuli_delta_mini_batch) * pi_a_k * pi_b_k
        return Z_constant_mini_batch

    def Z_constant_mini_batch(self, node_a_membership, node_b_membership, links_flag):
        if links_flag:
            bernuli_delta_mini_batch = np.ones((len(node_a_membership), 1)) * self.delta
        else:
            bernuli_delta_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.delta)
        Z_constant_mini_batch = bernuli_delta_mini_batch.copy()
        for k in range(self.k):
            if links_flag:
                bernuli_beta_k_mini_batch = np.ones((len(node_a_membership), 1)) * self.beta[k]
            else:
                bernuli_beta_k_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.beta[k])
            pi_a_k = (node_a_membership[:, k]).reshape(len(node_a_membership), 1)
            pi_b_k = (node_b_membership[:, k]).reshape(len(node_b_membership), 1)
            Z_constant_mini_batch += (bernuli_beta_k_mini_batch - bernuli_delta_mini_batch) * pi_a_k * pi_b_k
        return Z_constant_mini_batch

    def function_f_ab_k_k(self, node_a_membership, node_b_membership, observation_ab, k):
        f_ab_k_k = bernuli_distrbution(observation_ab, self.beta[k]) * node_a_membership[k] * node_b_membership[k]
        return f_ab_k_k

    def function_f_ab_k_mini_batch_pi(self, k, node_a_membership, node_b_membership, links_index):
        bernuli_delta_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.delta)
        bernuli_delta_mini_batch[links_index] = self.delta

        bernuli_beta_k_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.beta[k])
        bernuli_beta_k_mini_batch[links_index] = self.beta[k]

        node_a_k = (node_a_membership[:, k]).reshape(len(node_a_membership), 1)

        node_b_k = (node_b_membership[:, k]).reshape(len(node_b_membership), 1)

        f_ab_k_mini_batch = node_a_k * (
            bernuli_beta_k_mini_batch * node_b_k + bernuli_delta_mini_batch * (1 - node_b_k))
        return f_ab_k_mini_batch

    def function_f_ab_k_k_mini_batch(self, k, node_a_membership, node_b_membership, link_flag):
        if link_flag:
            f_ab_k_k_mini_batch = np.ones((len(node_a_membership), 1)) * self.beta[k]
        else:
            f_ab_k_k_mini_batch = np.ones((len(node_a_membership), 1)) * (1 - self.beta[k])
        f_ab_k_k_mini_batch = f_ab_k_k_mini_batch * (node_a_membership[:, k]).reshape(len(node_b_membership), 1) * (
            node_b_membership[:, k]).reshape(len(node_b_membership), 1)

        return f_ab_k_k_mini_batch

    def update_phi(self, batch_nodes_index, step_size, n_list_set):
        n = self.mini_batch_nodes
        grad_phi = np.zeros((n, self.k))
        # select edges
        node_a = np.zeros(n * n).astype(int)
        node_b = np.zeros(n * n).astype(int)
        y_mini_batch = np.zeros(n * n).astype(int)
        corrections = np.zeros(n * n)
        for i, node in enumerate(batch_nodes_index):
            # deal with links
            node_neighbors = self.node_neighbors_dict.setdefault(node, [])

            links_n = len(node_neighbors)
            node_a[i * n:i * n + n] = node
            node_b[i * n:i * n + links_n] = node_neighbors
            y_mini_batch[i * n:i * n + links_n] = 1

            # deal with non-links
            non_neighbors = np.setdiff1d(n_list_set, node_neighbors) # might cause bottleneck O(N) complexity (leads to O(N^2)) !!
            non_neighbors = np.setdiff1d(non_neighbors, np.array([node]))
            np.random.shuffle(non_neighbors)
            sampled_node_neighbors = non_neighbors[:n - links_n - 1] # -1 because of the [node] deletion
            node_b[(i * n + links_n + 1):(i * n + n)] = sampled_node_neighbors # +1 because of the [node]

            corrections[i * n: i * n + links_n] = 1
            corrections[(i * n + links_n + 1):(i * n + n)] = float(self.n - links_n - 1) / (self.mini_batch_nodes - links_n) # +1 because of the [node]

        corrections = corrections.reshape(n, n)
        pi_a = self.pi[node_a]
        pi_b = self.pi[node_b]
        phi_a = self.phi[node_a]

        links_index = np.where(y_mini_batch == 1)[0]
        Z_ab_mini_batch = self.Z_constant_mini_batch_phi(pi_a, pi_b, links_index)

        for k in range(self.k):
            f_ab_k_mini_batch = self.function_f_ab_k_mini_batch_pi(k, pi_a, pi_b, links_index)
            phi_a_k = (phi_a[:, k]).reshape(n * n, 1)

            temp_denumerator = Z_ab_mini_batch * phi_a_k
            denumerator = temp_denumerator.copy()

            index_zero = np.where(temp_denumerator == 0)[0]
            denumerator[index_zero] = 10 ** (-25)

            temp = f_ab_k_mini_batch / denumerator - np.ones((n * n, 1)) / ((np.sum(phi_a,
                                                                                    axis=1)).reshape(
                n * n, 1))

            temp = temp.reshape(n, n) * corrections
            grad_phi[:, k] = np.sum(temp, axis=1)

        temp_phi = np.abs(self.phi[batch_nodes_index] + step_size / 2 * (
        - self.phi[batch_nodes_index] * self.gamma_scale + grad_phi * self.phi[batch_nodes_index]))
        return temp_phi

    # @do_profile(follow=[])
    def update_theta(self, step_size):
        grad_theta = np.zeros((self.k, 2))

        # Method from Zhang et al. uses N^2 memory
        if self.dense:
            sample_non_edges_index = np.random.randint(self.nonedges_n, size=self.sampled_non_edges_n)
            non_edges_index_a = self.nonedges[0][sample_non_edges_index]
            non_edges_index_b = self.nonedges[1][sample_non_edges_index]

        # Rejection sampling method
        else:
            sampled_non_edges, N_non_edges = sample_non_edges(self.edges_dict, self.n, self.edges_n, self.sampled_non_edges_ratio)
            non_edges_index_a = sampled_non_edges[0]
            non_edges_index_b = sampled_non_edges[1]

        pi_a_non_links = self.pi[non_edges_index_a]
        pi_b_non_links = self.pi[non_edges_index_b]
        z_ab_non_links = self.Z_constant_mini_batch(pi_a_non_links, pi_b_non_links, links_flag=False)

        edges_index_a = self.edges[0]
        edges_index_b = self.edges[1]
        pi_a_links = self.pi[edges_index_a]
        pi_b_links = self.pi[edges_index_b]
        z_ab_links = self.Z_constant_mini_batch(pi_a_links, pi_b_links, links_flag=True)

        # Method from Zhang et al.
        correction = float(self.nonedges_n) / len(non_edges_index_a)


        for k in range(self.k):
            f_ab_kk_mini_batch_links = self.function_f_ab_k_k_mini_batch(k, pi_a_links, pi_b_links, link_flag=True)

            links_term = f_ab_kk_mini_batch_links / z_ab_links

            f_ab_kk_mini_batch_non_links = self.function_f_ab_k_k_mini_batch(k, pi_a_non_links, pi_b_non_links,
                                                                             link_flag=False)
            non_links_term = (f_ab_kk_mini_batch_non_links / z_ab_non_links) * correction

            theta_k = (np.sum(self.theta, axis=1))[k]

            if self.theta[k][0] < 10 ** (-50):
                self.theta[k][0] = 10 ** (-50)

            if self.theta[k][1] < 10 ** (-50):
                self.theta[k][1] = 10 ** (-50)

            # print(self.theta[k][0], self.theta[k][1], theta_k)
            grad_theta[k][0] = np.sum(links_term * (-1.0 / theta_k)) \
                               + np.sum(non_links_term * (1.0 / self.theta[k][0] - 1.0 / theta_k))

            grad_theta[k][1] = np.sum(links_term * (1.0 / self.theta[k][1] - 1.0 / theta_k)) \
                               + np.sum(non_links_term * (-1.0 / theta_k))

        temp_theta = np.abs(
            self.theta + (step_size / 2) * (-self.theta * self.gamma_scale + grad_theta * self.theta))
        return temp_theta

    def train_one_epoch(self, step_size, n_list_set):
        # st = time()

        batch_nodes_index = np.random.choice(self.n, size=self.mini_batch_nodes, replace=False)
        temp_phi = self.update_phi(batch_nodes_index, step_size, n_list_set)

        self.phi[batch_nodes_index] = temp_phi
        self.pi, self.phi_constant = reparameterized_to_pi(self.phi, self.n)

        temp_theta = self.update_theta(step_size)
        self.beta, self.theta_constant = reparameterized_to_beta(temp_theta)
        self.theta = temp_theta

        # print(f"Computing one epoch took {time() - st} sec")
        return

    # Method used in Zhang et al. (useless for our purpose)
    # def evaluation(self, beta_list, pi_list):
    #     perplexity_score = metric_perp_avg(beta_list, pi_list, self.test_set, self.y_test_set, self.delta)
    #     ARI, acc, change_from_the_initial_prediction_labels, avg_predict_label = accuracy_avg(pi_list,
    #                                                                                           self.initial_prediction_labels,
    #                                                                                           self.true_labels,
    #                                                                                           self.sample_n,
    #                                                                                           self.k,
    #                                                                                           self.val_set_index)
    #     return perplexity_score, ARI, acc, change_from_the_initial_prediction_labels, avg_predict_label

    def model_training(self):
        """
        MMSBM model training using stochastic MCMC
        """
        avg_perp = []
        prediction_acc = []
        beta_list = []
        ARI_list = []
        change_of_labels = []
        beta_samples = []
        pi_samples = []

        for itr in range(self.max_iter):
            step_size = step_size_function(itr, self.tao, self.step_size_scalar)
            self.train_one_epoch(step_size, self.n_list_set)

            if len(beta_samples) < self.T:
                beta_samples.append(self.beta)
                pi_samples.append(self.pi)

            if len(beta_samples) == self.T:
                beta_samples.pop(0)
                pi_samples.pop(0)
                beta_samples.append(self.beta)
                pi_samples.append(self.pi)

            for k in range(self.k):
                self.B[k][k] = self.beta[k]


@torch.no_grad()
def add_mmsbm_enc(data, n_communities=8, attr_name_abs='mmsbm', attr_name_rel='mmsbm', spe=False):
    FLAGS = flags()
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight

    model = MMSBM_SGMCMC(flags=FLAGS, n=num_nodes, k=n_communities, edges=edge_index, step_size_scalar=1, node_neighbors_dict=get_node_neighbor_dict(edge_index, num_nodes))
    model.model_training()
    pi, beta = torch.from_numpy(model.pi), torch.from_numpy(model.beta)

    abs_pe = pi

    rel_pe_idx = edge_index
    rel_pe_val = torch.zeros((edge_index.shape[1], abs_pe.shape[1]))
    
    # TODO : parallelize the relative encoding computation
    for i in range(edge_index.shape[1]):
        index_a, index_b = edge_index[:,i]
        rel_pe_val[i] =  abs_pe[index_b] * (abs_pe[index_a].dot(beta))

    # Convert to float
    abs_pe = abs_pe.float()
    rel_pe_val = rel_pe_val.float()

    data = add_node_attr(data, abs_pe, attr_name=attr_name_abs)
    data = add_node_attr(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = add_node_attr(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")

    data.deg = torch.from_numpy(model.deg)
    data.log_deg = torch.log(data.deg + 1)

    return data