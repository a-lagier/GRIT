
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

def angle_clamp(x):
    return x % (2*np.pi)


def aggregate_k_hop(edge_index, angles, k, alpha=2.):
    st = time()
    enhanced_angles = angles
    
    k_paths = edge_index
    row, col = k_paths
    enhanced_angles[row] += angles[col] * (1 / 2.) ** alpha

    for k_ in range(k - 1):
        edge1_idx, edge2_idx = torch.where(k_paths[1][:, None] == edge_index[0][None, :])

        path_sources = k_paths[0][edge1_idx]
        path_targets = edge_index[1][edge2_idx]
        
        k_paths = torch.stack([path_sources, path_targets], dim=0)

        # Delete duplicates
        # TODO : take into account the number of duplicates in the aggregation
        k_paths = torch.tensor(np.unique(k_paths.cpu().numpy(), axis=1))

        row, col = k_paths
        enhanced_angles[row] += angles[col] * (1/(k_ + 3)) ** alpha

    # print(f"Aggregating angles took {(time() - st):.4f} sec")    

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
    def __init__(self, in_dim, n_hidden_layers, out_dim=1, use_bias=False, dropout=0.1, aggregate_range=3, pe_name="rogpe"):
        super().__init__()

        self.in_dim = in_dim
        self.rotation_dim = in_dim
        self.hidden_dim = in_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.aggregate_range = aggregate_range

        layers = [nn.Linear(self.rotation_dim, self.hidden_dim), nn.PReLU()]
        for _ in range(n_hidden_layers):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias),
                nn.PReLU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))

        for l in layers[::2]:
            nn.init.xavier_normal_(l.weight)

        self.layers = nn.Sequential(*layers)

        print(f"RoGPE node encoding model has {params_count(self)} parameters")

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in Gamma model")

        # Compute nodes angles
        batch.node_rotation_angles = self.layers(batch.coeffs) #+ 10000.0 # based on RoPE paper

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

        for l in layers[::2]:
            nn.init.xavier_normal_(l.weight)

        self.layers = nn.Sequential(*layers)

        print(f"RoGPE edge encoding model has {params_count(self)} parameters")

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in Gamma model")

        # src = batch.coeffs[batch.edge_index[0]]
        # dest = batch.coeffs[batch.edge_index[1]]

        # couple = torch.stack([src, dest], dim=2).view(-1, self.in_dim)

        # batch.edge_rotation_angles = self.layers(couple) # + 10000.0

        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}"


#
#
#
#
# Encoders for eigenvalues methods in RoGPE
#
#
#
#

@register_node_encoder('rogpe_linear_eigen')
class RoGPEEigenLinearNodeEncoder(torch.nn.Module):
    '''
    
    '''

    def __init__(self, in_dim, out_dim, n_hidden_layers, use_bias=False, dropout=0., angle_clamping=False):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = out_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = nn.Dropout(dropout) # TODO see how to use it
        self.angle_clamping = angle_clamping

        self.theta_0 = 1.0

        # Eigenvector MLP encoder
        layers_eigvects = [nn.Linear(self.in_dim, self.hidden_dim), nn.PReLU()]
        for _ in range(n_hidden_layers):
            layers_eigvects.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers_eigvects.append(nn.PReLU())
        layers_eigvects.append(nn.Linear(self.hidden_dim, self.out_dim))

        for l in layers_eigvects[::2]:
            nn.init.xavier_normal_(l.weight)

        self.layers_eigvects = nn.Sequential(*layers_eigvects)


        # Eigenvalues MLP encoder
        layers_eigvals = [nn.Linear(self.in_dim, self.hidden_dim), nn.PReLU()]
        for _ in range(n_hidden_layers):
            layers_eigvals.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers_eigvals.append(nn.PReLU())
        layers_eigvals.append(nn.Linear(self.hidden_dim, self.out_dim))

        for l in layers_eigvals[::2]:
            nn.init.xavier_normal_(l.weight)

        self.layers_eigvals = nn.Sequential(*layers_eigvals)


        print(f"RoGPE node encoding model has {params_count(self.layers_eigvects)} and {params_count(self.layers_eigvals)} parameters")
        pass

    def forward(self, batch):
        if sum([p.isnan().sum() for p in self.parameters()]):
            warnings.warn("Nan parameters has been found in Encoder model")

        batch.eigvecs = batch.eigvecs.view(-1, self.in_dim)
        batch.eigvals = batch.eigvals.view(-1, self.in_dim)

        # batch.eigvecs = self.layers_eigvects(batch.eigvecs)
        # batch.eigvals = self.layers_eigvals(batch.eigvals)

        batch.node_rotation_angles = (batch.eigvals * batch.eigvecs.pow(2)).sum(dim = -1) * self.theta_0 # disable torch.abs

        if self.angle_clamping: # maybe not a good idea since % operation behaves weirdly with torch autograd
            batch.node_rotation_angles = angle_clamp(batch.node_rotation_angles)

        return batch

@register_edge_encoder('rogpe_linear_eigen')
class RoGPEEigenLinearEdgeEncoder(torch.nn.Module):
    '''
        Dummy encoder
    '''

    def __init__(self):
        super().__init__()

    def forward(self, batch):
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