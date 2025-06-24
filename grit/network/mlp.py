from typing import Optional

import torch
from torch import nn


class MLP(nn.Module):
    layers: nn.ModuleList
    fc: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, use_bn: bool, activation: str,
        dropout_prob: float, norm_type: str = "batch"
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 1):
            layer = MLPLayer(in_dims, hidden_dims, use_bn, activation, dropout_prob, norm_type)
            self.layers.append(layer)
            in_dims = hidden_dims

        self.fc = nn.Linear(hidden_dims, out_dims, bias=True)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        for layer in self.layers:
            X = layer(X)      # [***, D_hid]
        X = self.fc(X)        # [***, D_out]
        X = self.dropout(X)   # [***, D_out]
        return X

    @property
    def out_dims(self) -> int:
        return self.fc.out_features


class MultiMLP(nn.Module):
    def __init__(
        self, n_layers: int, in_dims: int, hidden_dims: int, out_dims: int, use_bn: bool, activation: str,
        dropout_prob: float, norm_type: str = "batch", n_models: int = 5, model_aggregation: str = "none"
    ) -> None:
        super().__init__()

        self.n_models = out_dims if model_aggregation == "none" else n_models
        self.model_aggregation = model_aggregation
        model_out_dim = 1 if model_aggregation == "none" else out_dims


        self.models = nn.ModuleList()
        for _ in range(self.n_models):
            model = MLP(n_layers=n_layers, in_dims=in_dims, hidden_dims=hidden_dims, 
                        out_dims=model_out_dim, use_bn=use_bn, activation=activation,
                        dropout_prob=dropout_prob, norm_type=norm_type)
            self.models.append(model)


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        output = []
        for model in self.models:
            output.append(model(X).unsqueeze(-1))      # [***, 1 or D_out]
        
        X = torch.cat(output, dim=-1)

        if self.model_aggregation == "sum":
            X = torch.sum(X, dim=-1)
        elif self.model_aggregation == "mean":
            X = torch.mean(X, dim=-1)
        elif self.model_aggregation == "none":
            pass

        return X


class MLPLayer(nn.Module):
    """
    Based on https://pytorch.org/vision/main/_modules/torchvision/ops/misc.html#MLP
    """
    fc: nn.Linear
    bn: Optional[nn.BatchNorm1d]
    activation: nn.Module
    dropout: nn.Dropout

    def __init__(self, in_dims: int, out_dims: int, use_bn: bool, activation: str,
                 dropout_prob: float, norm_type: str = "batch") -> None:
        super().__init__()
        self.fc = nn.Linear(in_dims, out_dims, bias=True)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_dims) if norm_type == "batch" else nn.LayerNorm(out_dims)
        else:
            self.bn = None

        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'none':
            self.activation = nn.Identity()
        else:
            raise ValueError("Invalid activation!")
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        :param X: Input feature matrix. [***, D_in]
        :return: Output feature matrix. [***, D_out]
        """
        X = self.fc(X)                     # [***, D_out]
        if self.bn is not None:
            shape = X.size()
            X = X.reshape(-1, shape[-1])   # [prod(***), D_out]
            X = self.bn(X)                 # [prod(***), D_out]
            X = X.reshape(shape)           # [***, D_out]

        X = self.activation(X)             # [***, D_out]
        X = self.dropout(X)                # [***, D_out]
        return X