
'''
    The RoGPE encoder for GRIT
'''
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import torch_geometric as pyg
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)
from torch_geometric.graphgym.utils.comp_budget import params_count
from torch_scatter import scatter
import warnings

from time import time

from grit.network.mlp import MLP, MultiMLP
from grit.network.deepset import DeepSets



def aggregate_k_hop(edge_index, angles, k, alpha=2.):
    # TODO : add values to edges
    if k < 1:
        return angles

    # f = lambda k : np.log(k + 1.)
    f = lambda k : np.exp(-alpha*k)

    enhanced_angles = angles
    step_k = torch.zeros_like(enhanced_angles)
    
    k_paths = edge_index
    row, col = k_paths
    # degree_row, degree_col = degree[row], degree[col]
    scatter(angles[col] * f(1), row, out=step_k, dim=0)
    enhanced_angles += step_k

    for k_ in range(2, k + 1):
        src_idx, dest_idx = torch.where(k_paths[1][:, None] == edge_index[0][None, :])

        src = k_paths[0][src_idx]
        dest = edge_index[1][dest_idx]
        
        k_paths = torch.stack([src, dest], dim=0)

        # Delete duplicates
        k_paths = torch.tensor(np.unique(k_paths.cpu().numpy(), axis=1))
        
        if torch.cuda.is_available():
            k_paths = k_paths.cuda()

        row, col = k_paths
        scatter(angles[col] * f(k), row, out=step_k, dim=0)
        enhanced_angles += step_k

    return enhanced_angles

#
#
#
#
# Encoders for coefficients methods in ROGPE
#
#
#
#

@register_node_encoder('rogpe_linear_coeffs')
class RoGPELinearNodeEncoder(torch.nn.Module):
    """
        RoGPE encoder for coeffs : given the degree coefficient, compute the rotation angle for each node
    """
    def __init__(self, in_dim, n_hidden_layers, out_dim=1,
                use_bias=False, dropout=0.1, aggregate_range=3,
                pe_name="rogpe", angle_model="MLP", aggregation="mean",
                use_bn=True):
        super().__init__()

        self.in_dim = in_dim
        self.rotation_dim = in_dim
        self.hidden_dim = in_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = dropout
        self.use_bn = use_bn

        self.aggregate_range = aggregate_range

        self.angle_model = angle_model
        self.aggregation = aggregation


        if self.angle_model == "MLP":
            self.fc = MLP(n_layers=self.n_hidden_layers+2, in_dims=self.in_dim, hidden_dims=self.hidden_dim,
                          out_dims=self.out_dim, use_bn=self.use_bn, activation="relu", dropout_prob=self.dropout)
        elif self.angle_model == "DeepSet":
            self.fc = DeepSets(in_dim=self.in_dim, hidden_dim=self.hidden_dim, out_dim=self.out_dim,
                               n_hidden_layers=self.n_hidden_layers, aggregation=self.aggregation,
                               dropout=self.dropout, use_bn=self.use_bn)
        elif self.angle_model == "MultiMLP":
            self.fc = MultiMLP(n_layers=self.n_hidden_layers+2, in_dims=self.in_dim, hidden_dims=self.hidden_dim,
                          out_dims=self.out_dim, use_bn=self.use_bn, activation="relu", dropout_prob=self.dropout,
                          n_models=5, model_aggregation="mean")


        print(f"RoGPE node encoding model has {params_count(self)} parameters")

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in node angle network model")

        # Compute nodes angles
        X = self.fc(batch.coeffs)

        # Enhance nodes angles by aggregating them with their neighbohrs'
        batch.node_rotation_angles = aggregate_k_hop(batch.edge_index, X, self.aggregate_range)

        return batch


    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"{self.fc.__repr__()})"

@register_edge_encoder('rogpe_linear_coeffs')
class RoGPELinearEdgeEncoder(torch.nn.Module):
    '''
        RoGPE encoder for coeffs
    '''
    def __init__(self, in_dim, n_hidden_layers, out_dim=1, use_bias=False, dropout=0.):
        super().__init__()

        self.in_dim = in_dim
        self.rotation_dim = in_dim // 2
        self.hidden_dim = in_dim
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = nn.Dropout(dropout) # TODO see how to use it

        self.out_dim = out_dim

        layers = [nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias),
                nn.PReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))

        for l in layers:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight)

        self.layers = nn.Sequential(*layers)

        print(f"RoGPE edge encoding model has {params_count(self)} parameters")

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in edge angle network model")

        # src = batch.coeffs[batch.edge_index[0]]
        # dest = batch.coeffs[batch.edge_index[1]]

        # couple = torch.stack([src, dest], dim=2).view(-1, self.in_dim)

        # batch.edge_rotation_angles = self.layers(couple) # + 10000.0

        return batch


###################