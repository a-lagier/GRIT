import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

import opt_einsum as oe

from torch_geometric.graphgym.register import *

def pyg_softmax(src, index, num_nodes=None):
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """

    num_nodes = maybe_num_nodes(index, num_nodes)

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()
    out = out / (
            scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)

    return out


class MultiHeadSparseDotProductAttentionLayer(nn.Module):
    """
        Dot Product Attention Layer

        Sparse Implementation
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias=True,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=dict(),
                 num_angles=1,
                 **kwargs):
        super().__init__()


        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)


        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.V.weight)


        # Soft partitioning method
        self.num_angles = num_angles
        self.S = nn.Parameter(torch.zeros(self.num_angles, self.out_dim * self.num_heads // 2), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.S)
        # add normalization term

    def apply_rotation(self, batch, Q, K):
        "Q,K shape is [-1, self.num_heads, self.out_dim]"
        d = self.out_dim * self.num_heads
        d_half = d // 2

        # Node attribute rotation
        # Second method with multiple angles
        node_thetas = batch.node_rotation_angles.view(-1, self.num_angles)
        node_thetas = (node_thetas @ self.softmax(self.S)).repeat_interleave(2, dim=1).reshape(Q.shape)
        
        # First method
        #node_thetas = batch.node_rotation_angles.view(-1)
        #node_thetas_power = torch.tensor([ 1 for i in range(d_half)]).repeat_interleave(2, dim=0) # replace 1 by -2*i/d
        #node_thetas = torch.pow(node_thetas[None, :], node_thetas_power[:, None])
        #node_thetas = node_thetas.reshape(Q.shape)

        cos_pos = torch.cos(node_thetas)
        sin_pos = torch.sin(node_thetas)

        q2 = torch.stack([-Q[..., 1::2], Q[..., ::2]], dim=2)
        k2 = torch.stack([-K[..., 1::2], K[..., ::2]], dim=2)

        q2 = q2.reshape(Q.shape)
        k2 = k2.reshape(K.shape)

        Q = Q * cos_pos + q2 * sin_pos
        K = K * cos_pos + k2 * sin_pos

        return Q, K

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]                           # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]                          # (num relative) x num_heads x out_dim

        score = torch.sum(src * dest, dim=2)                         # dot product attention mechanism

        score /= (self.out_dim)**(1/2)

        score = score.view(-1, self.num_heads, 1)

        # final attn
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        batch.Q_h, batch.K_h = self.apply_rotation(batch, batch.Q_h, batch.K_h)

        self.propagate_attention(batch)
        h_out = batch.wV

        return h_out


class MultiHeadDenseDotProductAttentionLayer(nn.Module):
    """
        Dot Product Attention Layer

        Dense implementation
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias=True,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=dict(),
                 num_angles=1,
                 **kwargs):
        super().__init__()


        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.clamp = np.abs(clamp) if clamp is not None else None
        self.edge_enhance = edge_enhance

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)


        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.V.weight)


        # Soft partitioning method
        self.num_angles = num_angles
        self.S = nn.Parameter(torch.zeros(self.num_angles, self.out_dim * self.num_heads // 2), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_normal_(self.S)
        # add normalization term

    def apply_rotation(self, batch, Q, K):
        "Q,K shape is [-1, self.num_heads, self.out_dim]"
        d = self.out_dim * self.num_heads
        d_half = d // 2

        # Node attribute rotation
        # Second method with multiple angles
        node_thetas = batch.node_rotation_angles.view(-1, self.num_angles)
        node_thetas = (node_thetas @ self.softmax(self.S)).repeat_interleave(2, dim=1).reshape(Q.shape)

        cos_pos = torch.cos(node_thetas)
        sin_pos = torch.sin(node_thetas)

        q2 = torch.stack([-Q[..., 1::2], Q[..., ::2]], dim=2)
        k2 = torch.stack([-K[..., 1::2], K[..., ::2]], dim=2)

        q2 = q2.reshape(Q.shape)
        k2 = k2.reshape(K.shape)

        Q = Q * cos_pos + q2 * sin_pos
        K = K * cos_pos + k2 * sin_pos

        return Q, K

    def propagate_attention(self, batch):
        src = batch.K_h                           # (num relative) x num_heads x out_dim
        dest = batch.Q_h                         # (num relative) x num_heads x out_dim

        src = src.view(self.num_heads, self.out_dim, -1)
        dest = dest.view(self.num_heads, -1, self.out_dim)

        score = dest @ src                         # dot product attention mechanism

        score /= self.out_dim ** 0.5

        # final attn
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        sfmax = nn.Softmax(dim=-1)
        score = sfmax(score)                # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = score @ batch.V_h.transpose(0, 1)  # (num relative) x num_heads x out_dim
        msg = msg.transpose(0, 1)
        batch.wV = msg  # (num nodes in batch) x num_heads x out_dim

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        batch.Q_h, batch.K_h = self.apply_rotation(batch, batch.Q_h, batch.K_h)

        self.propagate_attention(batch)
        h_out = batch.wV

        return h_out



@register_layer("GraphGPS")
class GPSLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, in_dim, out_dim,
                 local_gnn_type, global_model_type, num_heads, act='relu',
                 pna_degrees=None, dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 log_attn_weights=False,
                 num_angles=1,
                 cfg=dict(),
                 **kwargs):
        super().__init__()

        assert in_dim == out_dim
        dim_h = in_dim
        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = register.act_dict[act]

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        # Global attention transformer-style model.
        # TODO : add possibility to change attention layer w.r.t cfg file 
        # TODO : allow out_dim not divisible by num_heads with a encoder layer
        if cfg.attn.get('dotproduct_attn', False):

            if cfg.attn.get('dense', False):
                self.self_attn = MultiHeadDenseDotProductAttentionLayer(
                in_dim=in_dim,
                out_dim=out_dim // num_heads,
                num_heads=num_heads,
                use_bias=cfg.attn.get("use_bias", False),
                dropout=attn_dropout,
                clamp=cfg.attn.get("clamp", 5.),
                act=cfg.attn.get("act", "relu"),
                edge_enhance=True,
                sqrt_relu=cfg.attn.get("sqrt_relu", False),
                signed_sqrt=cfg.attn.get("signed_sqrt", False),
                scaled_attn =cfg.attn.get("scaled_attn", False),
                no_qk=cfg.attn.get("no_qk", False),
                num_angles=num_angles
            )
            else:
                self.self_attn = MultiHeadSparseDotProductAttentionLayer(
                in_dim=in_dim,
                out_dim=out_dim // num_heads,
                num_heads=num_heads,
                use_bias=cfg.attn.get("use_bias", False),
                dropout=attn_dropout,
                clamp=cfg.attn.get("clamp", 5.),
                act=cfg.attn.get("act", "relu"),
                edge_enhance=True,
                sqrt_relu=cfg.attn.get("sqrt_relu", False),
                signed_sqrt=cfg.attn.get("signed_sqrt", False),
                scaled_attn =cfg.attn.get("scaled_attn", False),
                no_qk=cfg.attn.get("no_qk", False),
                num_angles=num_angles
            )
        self.global_model_type = global_model_type

        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            self.norm1_attn = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_attn = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)

        # Feed Forward block.
        self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
        self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        self.act_fn_ff = self.activation()
        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)
        self.ff_dropout1 = nn.Dropout(dropout)
        self.ff_dropout2 = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_with_edge_attr:
                h_local = self.local_model(h,
                                            batch.edge_index,
                                            batch.edge_attr)
            else:
                h_local = self.local_model(h, batch.edge_index)
            h_local = self.dropout_local(h_local)
            h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.self_attn is not None:
            # TODO : uncomment below (maybe)
            h_attn = self.self_attn(batch).reshape((-1, self.dim_h))
            # h_dense, mask = to_dense_batch(h, batch.batch)
            # if self.global_model_type == 'Transformer':
            #     h_attn = self._sa_block(h_dense, None, ~mask)[mask]
            # elif self.global_model_type == 'BiasedTransformer':
            #     # Use Graphormer-like conditioning, requires `batch.attn_bias`.
            #     h_attn = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            # else:
            #     raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_attn = self.dropout_attn(h_attn)
            h_attn = h_in1 + h_attn  # Residual connection.
            if self.layer_norm:
                h_attn = self.norm1_attn(h_attn, batch.batch)
            if self.batch_norm:
                h_attn = self.norm1_attn(h_attn)
            h_out_list.append(h_attn)

        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        # Feed Forward block.
        h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.self_attn(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.self_attn(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
