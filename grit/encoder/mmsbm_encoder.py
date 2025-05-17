'''
    The MMSBM encoder
'''

import torch
from torch import nn
from torch.nn import functional as F
import torch_sparse

import torch_geometric as pyg
from torch_geometric.graphgym.register import (
    register_edge_encoder,
    register_node_encoder,
)

from torch_geometric.utils import remove_self_loops, add_remaining_self_loops, add_self_loops
from torch_scatter import scatter
import warnings


@register_node_encoder('mmsbm_linear')
class MMSBMLinearNodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim, out_dim, use_bias=False, batchnorm=False, layernorm=False, pe_name="mmsbm"):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = pe_name

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)
        
    def forward(self, batch):
        # Encode just the first dimension if more exist
        mmsbm = batch[f"{self.name}"].float()
        mmsbm = self.fc(mmsbm)

        if self.batchnorm:
            mmsbm = self.bn(rrwmmsbmp)

        if self.layernorm:
            mmsbm = self.ln(mmsbm)

        if "x" in batch:
            batch.x = batch.x + mmsbm
        else:
            batch.x = mmsbm

        return batch

@register_edge_encoder('mmsbm_linear')
class MMSBMLinearEdgeEncoder(torch.nn.Module):
    '''
        Merge MMSBM with given edge-attr and Zero-padding to all pairs of node
        FC_1(RRWP) + FC_2(edge-attr)
        - FC_2 given by the TypedictEncoder in same cases
        - Zero-padding for non-existing edges in fully-connected graph
        - (optional) add node-attr as the E_{i,i}'s attr
            note: assuming  node-attr and edge-attr is with the same dimension after Encoders
    '''
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=False, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr

        self.batchnorm = batchnorm
        self.layernorm = layernorm
        if self.batchnorm or self.layernorm:
            warnings.warn("batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info ")

        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.

        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        mmsbm_idx = batch.mmsbm_index
        mmsbm_val = batch.mmsbm_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        mmsbm_val = self.fc(mmsbm_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), mmsbm_val.size(1))
            # zero padding for non-existing edges

        if self.overwrite_old_attr:
            out_idx, out_val = mmsbm_idx, mmsbm_val
        else:
            # edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)

            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, mmsbm_idx], dim=1),
                torch.cat([edge_attr, mmsbm_val], dim=0),
                batch.num_nodes, batch.num_nodes,
                op="add"
            )


        # deprecated for now (see definition at .rrwp_encoder.py and analyse complexity)
        if False and self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch

    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"