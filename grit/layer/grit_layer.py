import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter, scatter_max, scatter_add

from torch_geometric.graphgym.register import *
import opt_einsum as oe

from yacs.config import CfgNode as CN

import warnings

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


@register_layer("DotProductTransformer")
class MultiHeadDotProductEdgeAttentionLayer(nn.Module):
    """
        Dot Product Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias=True,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
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
        self.E = nn.Linear(in_dim, 1 * num_heads, bias=True)
        self.Ew = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)


        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.Ew.weight)
        nn.init.xavier_normal_(self.V.weight)

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)


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

        if batch.get("E", None) is not None:
            batch.E = batch.E.reshape(score.shape)
            score = score + batch.E


        score = score.view(-1, self.num_heads, 1)

        e_t = score

        # final attn
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = batch.Ew * score
            batch.wE = batch.wE.view(-1, self.num_heads, self.out_dim)

        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = torch.zeros_like(batch.wV)
            scatter(e_t * score, batch.edge_index[1], dim=0, out=rowV, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)

        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
            batch.Ew = self.Ew(batch.edge_attr).view(-1, self.num_heads, self.out_dim)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        batch.Q_h, batch.K_h = self.apply_rotation(batch, batch.Q_h, batch.K_h)

        batch.Q_h = batch.Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = batch.K_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)
        return h_out, e_out


@register_layer("LinearTransformer")
class MultiHeadLinearAttentionLayer(nn.Module):
    """
        Linear Attention Computation for GRIT

        Only to be used with RoGPE
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias=True,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
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
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.Ew = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.Ew.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)
        
        # Soft partitioning method
        self.num_angles = num_angles
        self.S = nn.Linear(self.num_angles, self.out_dim * self.num_heads // 2, bias=True)
        nn.init.xavier_normal_(self.S.weight)

    def apply_rotation(self, batch, Q, K, E=None):
        "Q,K,E shape is [-1, self.num_heads, self.out_dim]"
        d = self.out_dim * self.num_heads
        d_half = d // 2


        # Node attribute rotation
        # Second method
        node_thetas = batch.node_rotation_angles.view(-1, self.num_angles)
        node_thetas = self.S(node_thetas).repeat_interleave(2, dim=1).reshape(Q.shape)

        # First method
        # node_thetas = batch.node_rotation_angles.view(-1)
        # node_thetas_power = torch.tensor([ -2*i/d for i in range(d_half)]).repeat_interleave(2, dim=0)
        # node_thetas = torch.pow(node_thetas[None, :], node_thetas_power[:, None])
        # node_thetas = node_thetas.reshape(Q.shape)

        cos_pos = torch.cos(node_thetas)
        sin_pos = torch.sin(node_thetas)

        q2 = torch.stack([-Q[..., 1::2], Q[..., ::2]], dim=2)
        k2 = torch.stack([-K[..., 1::2], K[..., ::2]], dim=2)

        q2 = q2.reshape(Q.shape)
        k2 = k2.reshape(K.shape)

        Q = Q * cos_pos + q2 * sin_pos
        K = K * cos_pos + k2 * sin_pos

        # Edge attribute rotation
        if batch.get("E", None) is not None and batch.get("edge_rotation_angles") is not None:
            edge_thetas = batch.edge_rotation_angles.view(-1)
            edge_thetas_power = torch.tensor([-2*i/d for i in range(d_half)]).repeat_interleave(2, dim=0)

            edge_thetas = torch.pow(edge_thetas[None, :], edge_thetas_power[:,None])

            edge_thetas = edge_thetas.reshape(E.shape)

            cos_pos = torch.cos(edge_thetas)
            sin_pos = torch.sin(edge_thetas)

            e2 = torch.stack([-E[..., 1::2], E[..., ::2]], dim=2)

            e2 = e2.reshape(E.shape)

            E = E * cos_pos + e2 * sin_pos

        return Q, K, E

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]                           # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]                          # (num relative) x num_heads x out_dim

        score = src + dest                                             # linear attention ?

        if batch.get("E", None) is not None:
            batch.E = batch.E.reshape(score.shape)
            score = score + batch.E

        score /= 3

        score = score.view(-1, self.num_heads, self.out_dim)

        e_t = score

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = batch.Ew * score
            batch.wE = batch.wE.view(-1, self.num_heads, self.out_dim)

        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)

        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
            batch.Ew = self.Ew(batch.edge_attr).view(-1, self.num_heads, self.out_dim)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)

        batch.Q_h, batch.K_h, batch.E = self.apply_rotation(batch, batch.Q_h, batch.K_h, batch.E)

        batch.Q_h = batch.Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = batch.K_h.view(-1, self.num_heads, self.out_dim)
        batch.E = batch.E.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)
        return h_out, e_out

class MultiHeadAttentionLayerGritSparse(nn.Module):
    """
        Proposed Attention Computation for GRIT
    """

    def __init__(self, in_dim, out_dim, num_heads, use_bias,
                 clamp=5., dropout=0., act=None,
                 edge_enhance=True,
                 sqrt_relu=False,
                 signed_sqrt=True,
                 cfg=CN(),
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
        self.E = nn.Linear(in_dim, out_dim * num_heads * 2, bias=True)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        nn.init.xavier_normal_(self.Q.weight)
        nn.init.xavier_normal_(self.K.weight)
        nn.init.xavier_normal_(self.E.weight)
        nn.init.xavier_normal_(self.V.weight)

        self.Aw = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, 1), requires_grad=True)
        nn.init.xavier_normal_(self.Aw)

        if act is None:
            self.act = nn.Identity()
        else:
            self.act = act_dict[act]()

        if self.edge_enhance:
            self.VeRow = nn.Parameter(torch.zeros(self.out_dim, self.num_heads, self.out_dim), requires_grad=True)
            nn.init.xavier_normal_(self.VeRow)

    def propagate_attention(self, batch):
        src = batch.K_h[batch.edge_index[0]]      # (num relative) x num_heads x out_dim
        dest = batch.Q_h[batch.edge_index[1]]     # (num relative) x num_heads x out_dim
        score = src + dest                        # element-wise multiplication

        if batch.get("E", None) is not None:
            batch.E = batch.E.view(-1, self.num_heads, self.out_dim * 2)
            E_w, E_b = batch.E[:, :, :self.out_dim], batch.E[:, :, self.out_dim:]
            # (num relative) x num_heads x out_dim
            score = score * E_w
            score = torch.sqrt(torch.relu(score)) - torch.sqrt(torch.relu(-score))
            score = score + E_b

        score = self.act(score)
        e_t = score

        # output edge
        if batch.get("E", None) is not None:
            batch.wE = score.flatten(1)

        # final attn
        score = oe.contract("ehd, dhc->ehc", score, self.Aw, backend="torch")
        if self.clamp is not None:
            score = torch.clamp(score, min=-self.clamp, max=self.clamp)

        raw_attn = score
        score = pyg_softmax(score, batch.edge_index[1])  # (num relative) x num_heads x 1
        score = self.dropout(score)
        batch.attn = score

        # Aggregate with Attn-Score
        msg = batch.V_h[batch.edge_index[0]] * score  # (num relative) x num_heads x out_dim
        batch.wV = torch.zeros_like(batch.V_h)  # (num nodes in batch) x num_heads x out_dim
        scatter(msg, batch.edge_index[1], dim=0, out=batch.wV, reduce='add')

        if self.edge_enhance and batch.E is not None:
            rowV = scatter(e_t * score, batch.edge_index[1], dim=0, reduce="add")
            rowV = oe.contract("nhd, dhc -> nhc", rowV, self.VeRow, backend="torch")
            batch.wV = batch.wV + rowV

    def forward(self, batch):
        Q_h = self.Q(batch.x)
        K_h = self.K(batch.x)
        V_h = self.V(batch.x)

        if batch.get("edge_attr", None) is not None:
            batch.E = self.E(batch.edge_attr)
        else:
            batch.E = None

        batch.Q_h = Q_h.view(-1, self.num_heads, self.out_dim)
        batch.K_h = K_h.view(-1, self.num_heads, self.out_dim)
        batch.V_h = V_h.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(batch)
        h_out = batch.wV
        e_out = batch.get('wE', None)

        return h_out, e_out


@register_layer("GritTransformer")
class GritTransformerLayer(nn.Module):
    """
        Proposed Transformer Layer for GRIT
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 dropout=0.0,
                 attn_dropout=0.0,
                 layer_norm=False, batch_norm=True,
                 residual=True,
                 act='relu',
                 norm_e=True,
                 O_e=True,
                 cfg=dict(),
                 num_angles=1,
                 **kwargs):
        super().__init__()

        self.debug = False
        self.in_channels = in_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads

        # ensure that the output dimension is even with RoGPE
        # ensure that the output dimension is divisible by num_heads when using DotProduct attention
        # if cfg.attn.get('dotproduct_attn', False):
        #     if out_dim % self.num_heads != 0:
        #         warnings.warn("The output dimension is not divisible by the number of heads. The output dimension has been modified")
        #         self.out_dim += self.out_dim % self.num_heads
        #         if self.out_dim % 2 != 0:
        #             self.out_dim += self.num_heads
            
        #     elif out_dim % 2 != 0:
        #         warnings.warn("The output dimension of the Transformer layer is odd : to use RoGPE out_dim needs to be even. The output dimension has been modified")
        #         self.out_dim += 1
        #         if self.out_dim % self.num_heads != 0:
        #             self.out_dim += self.out_dim % self.num_heads
        #             if out_dim % 2 != 0:
        #                 self.out_dim += self.num_heads

        # print("Beginning attention")
        # print("in, out, num_heads are :", self.in_dim, self.out_dim, self.num_heads, self.out_dim // self.num_heads)

        self.out_channels = out_dim
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm

        # -------
        self.update_e = cfg.get("update_e", True)
        self.bn_momentum = cfg.bn_momentum
        self.bn_no_runner = cfg.bn_no_runner
        self.rezero = cfg.get("rezero", False)

        self.act = act_dict[act]() if act is not None else nn.Identity()
        if cfg.get("attn", None) is None:
            cfg.attn = dict()
        self.use_attn = cfg.attn.get("use", True)
        # self.sigmoid_deg = cfg.attn.get("sigmoid_deg", False)
        self.deg_scaler = cfg.attn.get("deg_scaler", True)

        self.attention = MultiHeadAttentionLayerGritSparse(
            in_dim=in_dim,
            out_dim=out_dim // num_heads,
            num_heads=num_heads,
            use_bias=cfg.attn.get("use_bias", False),
            dropout=attn_dropout,
            clamp=cfg.attn.get("clamp", 5.),
            act=cfg.attn.get("act", "relu"),
            edge_enhance=cfg.attn.get("edge_enhance", True),
            sqrt_relu=cfg.attn.get("sqrt_relu", False),
            signed_sqrt=cfg.attn.get("signed_sqrt", False),
            scaled_attn =cfg.attn.get("scaled_attn", False),
            no_qk=cfg.attn.get("no_qk", False),
        )

        if cfg.attn.get('graphormer_attn', False):
            self.attention = MultiHeadAttentionLayerGraphormerSparse(
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
            )

        if cfg.attn.get('dotproduct_attn', False):
            self.attention = MultiHeadDotProductEdgeAttentionLayer(
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

        if cfg.attn.get('linear_attn', False):
            self.attention = MultiHeadLinearAttentionLayer(
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


        self.O_h = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        if O_e:
            self.O_e = nn.Linear(out_dim//num_heads * num_heads, out_dim)
        else:
            self.O_e = nn.Identity()

        # -------- Deg Scaler Option ------

        if self.deg_scaler:
            self.deg_coef = nn.Parameter(torch.zeros(1, out_dim//num_heads * num_heads, 2))
            nn.init.xavier_normal_(self.deg_coef)

        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            self.layer_norm1_e = nn.LayerNorm(out_dim) if norm_e else nn.Identity()

        if self.batch_norm:
            # when the batch_size is really small, use smaller momentum to avoid bad mini-batch leading to extremely bad val/test loss (NaN)
            self.batch_norm1_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)
            self.batch_norm1_e = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum) if norm_e else nn.Identity()

        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim * 2)
        self.FFN_h_layer2 = nn.Linear(out_dim * 2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)

        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim, track_running_stats=not self.bn_no_runner, eps=1e-5, momentum=cfg.bn_momentum)

        if self.rezero:
            self.alpha1_h = nn.Parameter(torch.zeros(1,1))
            self.alpha2_h = nn.Parameter(torch.zeros(1,1))
            self.alpha1_e = nn.Parameter(torch.zeros(1,1))

    def forward(self, batch):
        h = batch.x
        num_nodes = batch.num_nodes
        log_deg = get_log_deg(batch)

        h_in1 = h  # for first residual connection
        e_in1 = batch.get("edge_attr", None)
        e = None
        # multi-head attention out

        h_attn_out, e_attn_out = self.attention(batch)

        h = h_attn_out.view(num_nodes, -1)
        h = F.dropout(h, self.dropout, training=self.training)

        # degree scaler
        if self.deg_scaler:
            h = torch.stack([h, h * log_deg], dim=-1)
            h = (h * self.deg_coef).sum(dim=-1)

        h = h.float()
        
        h = self.O_h(h)
        if e_attn_out is not None:
            e = e_attn_out.flatten(1)
            e = F.dropout(e, self.dropout, training=self.training)
            e = self.O_e(e)

        if self.residual:
            if self.rezero: h = h * self.alpha1_h
            h = h_in1 + h  # residual connection
            if e is not None:
                if self.rezero: e = e * self.alpha1_e
                e = e + e_in1

        if self.layer_norm:
            h = self.layer_norm1_h(h)
            if e is not None: e = self.layer_norm1_e(e)

        if self.batch_norm:
            h = self.batch_norm1_h(h)
            if e is not None: e = self.batch_norm1_e(e)

        # FFN for h
        h_in2 = h  # for second residual connection
        h = self.FFN_h_layer1(h)
        h = self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            if self.rezero: h = h * self.alpha2_h
            h = h_in2 + h  # residual connection

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)

        batch.x = h
        if self.update_e:
            batch.edge_attr = e
        else:
            batch.edge_attr = e_in1

        return batch

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})\n[{}]'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels, self.num_heads, self.residual,
            super().__repr__(),
        )


@torch.no_grad()
def get_log_deg(batch):
    if "log_deg" in batch:
        log_deg = batch.log_deg
    elif "deg" in batch:
        deg = batch.deg
        log_deg = torch.log(deg + 1).unsqueeze(-1)
    else:
        warnings.warn("Compute the degree on the fly; Might be problematric if have applied edge-padding to complete graphs")
        deg = pyg.utils.degree(batch.edge_index[1],
                               num_nodes=batch.num_nodes,
                               dtype=torch.float
                               )
        log_deg = torch.log(deg + 1)
    log_deg = log_deg.view(batch.num_nodes, 1)
    return log_deg


