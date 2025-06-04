
'''
    The RoGPE encoder for GRIT
'''
import torch
from torch import nn
from torch.nn import functional as F
from ogb.utils.features import get_bond_feature_dims
import torch_sparse

import torch_geometric as pyg
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)

from torch_geometric.utils import remove_self_loops, add_remaining_self_loops, add_self_loops
from torch_scatter import scatter
import warnings


class RotationNetwork(nn.Module):
    
    def __init__(self, in_dim, hidden_dim, n_hidden_layers, use_bias, dropout=0.):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = nn.Dropout(dropout)

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=self.use_bias))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))

        for l in layers[::2]:
            nn.init.xavier_normal_(l.weight)

        self.layers = nn.Sequential(*layers)
    
    def forward(self, batch):
        return self.layers(batch)

@register_node_encoder('rogpe_linear')
class RoGPELinearNodeEncoder(torch.nn.Module):
    """
        RoGPE encoder : given the degree coefficient, rotate the node embedding
    """
    def __init__(self, rotation_dim, n_hidden_layers, use_bias=False, dropout=0., pe_name="rogpe"):
        super().__init__()

        self.rotation_dim = rotation_dim
        self.hidden_dim = 2 * self.rotation_dim
        self.n_hidden_layers = n_hidden_layers
        self.use_bias = use_bias
        self.dropout = nn.Dropout(dropout)

        layers = [nn.Linear(self.rotation_dim, self.hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.use_bias))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dim, 1))

        for l in layers[::2]:
            nn.init.xavier_normal_(l.weight)

        self.layers = nn.Sequential(*layers)

    def forward(self, batch):
        # print("Number of nan parameters is:", sum([p.isnan().sum() for p in self.parameters()]))
        # print("Number of nan coeffs is:", batch.coeffs.isnan().sum())
        batch.rotation_angles = self.layers(batch.coeffs)
        # print("Number of nan rotation angles is:", batch.rotation_angles.isnan().sum())
        # print("Rotation angles are :", batch.rotation_angles)

        del batch.coeffs
        return batch


@register_edge_encoder('rogpe_linear')
class RoGPELinearEdgeEncoder(torch.nn.Module):
    '''
        Dummy encoder
    '''
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()

    def forward(self, batch):
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}"