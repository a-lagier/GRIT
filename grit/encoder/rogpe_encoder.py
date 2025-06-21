
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



def aggregate_k_hop(edge_index, angles, k, alpha=2.):
    # TODO : add values to edges
    if k < 1:
        return angles

    f = lambda k : np.log(k + 1.)

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
            layers = [nn.Linear(self.rotation_dim, self.hidden_dim), nn.ReLU()]
            for _ in range(n_hidden_layers):
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias),
                    nn.ReLU()
                ])
            layers.append(nn.Linear(self.hidden_dim, self.out_dim))

            for l in layers:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_normal_(l.weight)

            self.layers = nn.Sequential(*layers)
            self.bn = nn.BatchNorm1d(num_features=out_dim) if use_bn else None
            self.dropout = nn.Dropout(p=dropout)
        elif self.angle_model == "DeepSet":

            # Phi network: processes individual coefficients
            # here we enforce that the elements of the set are of dimension 1
            phi_layers = []
            prev_dim = 1
            for _ in range(n_hidden_layers + 2):
                phi_layers.extend([
                    nn.Linear(prev_dim, self.hidden_dim),
                    nn.ReLU()
                ])
                prev_dim = self.hidden_dim

            for l in phi_layers:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_normal_(l.weight)
            
            self.phi = nn.Sequential(*phi_layers)
            
            # Rho network: processes aggregated representation
            rho_layers = []
            for _ in range(n_hidden_layers + 1):
                rho_layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU()
                ])
            rho_layers.append(nn.Linear(self.hidden_dim, self.out_dim))
            
            for l in rho_layers:
                if isinstance(l, nn.Linear):
                    nn.init.xavier_normal_(l.weight)
            
            self.rho = nn.Sequential(*rho_layers)
            self.bn = nn.BatchNorm1d(num_features=out_dim) if use_bn else None
            self.dropout = nn.Dropout(p=dropout)


        print(f"RoGPE node encoding model has {params_count(self)} parameters")

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in node angle network model")

        # Compute nodes angles
        if self.angle_model == "MLP":
            X = self.layers(batch.coeffs) #+ 10000.0 # based on RoPE paper
        elif self.angle_model == "DeepSet":
            # batch_flat of dimension (n_element, n_el_per_set, in_dim)
            batch_flat = batch.coeffs.view(-1, self.in_dim, 1)

            phi_out = self.phi(batch_flat)
            
            if self.aggregation == 'sum':
                aggregated = torch.sum(phi_out, dim=1)
            elif self.aggregation == 'mean':
                aggregated = torch.mean(phi_out, dim=1)
            elif self.aggregation == 'max':
                aggregated = torch.max(phi_out, dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
            X = self.rho(aggregated)
        

        if self.bn is not None:
            shape = X.size()
            X = X.reshape(-1, shape[-1])
            X = self.bn(X)
            X = X.reshape(shape)

        X = self.dropout(X)

        # Enhance nodes angles by aggregating them with their neighbohrs'
        batch.node_rotation_angles = aggregate_k_hop(batch.edge_index, X, self.aggregate_range)

        return batch


class RoGPEMultiNetworkNodeEncoder(torch.nn.Module):
    """
        RoGPE encoder for coeffs : given the degree coefficient, compute the rotation angle for each node

        For each output dimension, we build one network of output dimension 1
    """
    def __init__(self, in_dim, n_hidden_layers, out_dim=1,
                use_bias=False, dropout=0.1, aggregate_range=3,
                pe_name="rogpe", angle_model="MLP", aggregation="mean"):
        super().__init__()

        self.n_networks = out_dim
        self.in_dim = in_dim
        self.hidden_dim = in_dim
        self.out_dim = 1
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.aggregate_range = aggregate_range

        self.angle_model = angle_model
        self.aggregation = aggregation

        self.models = []
        for n in range(self.n_networks):
            if self.angle_model == "MLP":
                layers = [nn.Linear(self.in_dim, self.hidden_dim), nn.ReLU()]
                for _ in range(n_hidden_layers):
                    layers.extend([
                        nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                layers.append(nn.Linear(self.hidden_dim, self.out_dim))

                for l in layers:
                    if isinstance(l, nn.Linear):
                        nn.init.xavier_normal_(l.weight)

                self.models.append(nn.Sequential(*layers))

            elif self.angle_model == "DeepSet":

                # Phi network: processes individual coefficients
                # here we enforce that the elements of the set are of dimension 1
                phi_layers = []
                prev_dim = 1
                for _ in range(n_hidden_layers + 2):
                    phi_layers.extend([
                        nn.Linear(prev_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                    prev_dim = self.hidden_dim

                for l in phi_layers:
                    if isinstance(l, nn.Linear):
                        nn.init.xavier_normal_(l.weight)
                
                phi = nn.Sequential(*phi_layers)
                
                # Rho network: processes aggregated representation
                rho_layers = []
                for _ in range(n_hidden_layers + 1):
                    rho_layers.extend([
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    ])
                rho_layers.append(nn.Linear(self.hidden_dim, self.out_dim))
                
                for l in rho_layers:
                    if isinstance(l, nn.Linear):
                        nn.init.xavier_normal_(l.weight)
                
                rho = nn.Sequential(*rho_layers)

                self.models.append((phi, rho))

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in node angle network model")

        # Compute nodes angles
        if self.angle_model == "MLP":
            batch.node_rotation_angles = self.layers(batch.coeffs)
        elif self.angle_model == "DeepSet":
            # batch_flat of dimension (n_element, n_el_per_set, in_dim)
            batch_flat = batch.coeffs.view(-1, self.in_dim, 1)

            phi_out = self.phi(batch_flat)
            
            if self.aggregation == 'sum':
                aggregated = torch.sum(phi_out, dim=1)
            elif self.aggregation == 'mean':
                aggregated = torch.mean(phi_out, dim=1)
            elif self.aggregation == 'max':
                aggregated = torch.max(phi_out, dim=1)[0]
            else:
                raise ValueError(f"Unknown aggregation: {self.aggregation}")
            batch.node_rotation_angles = self.rho(aggregated)

        # Enhance nodes angles by aggregating them with their neighbohrs'
        batch.node_rotation_angles = aggregate_k_hop(batch.edge_index, batch.node_rotation_angles, self.aggregate_range)

        return batch

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




class DeepSets(nn.Module):
    """
    DeepSets neural network for processing sets of coefficients.
    Architecture: phi(x_i) -> aggregation -> rho(aggregated)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden_layers,
                 aggregation='sum', dropout=0.1):
        super().__init__()
        
        warnings.warn("Be careful when using Deep Sets network, the set dimesion must be on the second coordinate")
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        self.aggregation = aggregation
        
        # Phi network: processes individual coefficients
        phi_layers = []
        prev_dim = in_dim
        for _ in range(n_hidden_layers + 2):
            phi_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        self.phi = nn.Sequential(*phi_layers)
        
        # Rho network: processes aggregated representation
        rho_layers = []
        for _ in range(n_hidden_layers + 1):
            phi_layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(dropout)
            ])
        rho_layers.append(nn.Linear(prev_dim, output_dim))
        self.rho = nn.Sequential(*rho_layers)
    
    def forward(self, batch):

        batch_flat = batch.view(-1, self.in_dim)
        phi_out = self.phi(batch_flat)
        
        if self.aggregation == 'sum':
            aggregated = torch.sum(phi_out, dim=1)
        elif self.aggregation == 'mean':
            aggregated = torch.mean(phi_out, dim=1)
        elif self.aggregation == 'max':
            aggregated = torch.max(phi_out, dim=1)[0]
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        output = self.rho(aggregated)
        return output