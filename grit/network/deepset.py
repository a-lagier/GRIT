import torch
from torch import nn

class DeepSets(nn.Module):
    """
    DeepSets neural network for processing sets of coefficients.
    Architecture: phi(x_i) -> aggregation -> rho(aggregated)
    """
    def __init__(self, in_dim, hidden_dim, out_dim, n_hidden_layers,
                 aggregation='sum', dropout=0.1,
                 use_bn=True):
        super().__init__()
        
        warnings.warn("Be careful when using Deep Sets network, the set dimesion must be on the second coordinate")
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.n_hidden_layers = n_hidden_layers
        self.aggregation = aggregation
        self.use_bn = use_bn

        if use_bn:
            self.bn = nn.BatchNorm1d(num_features=out_dim) if use_bn else None
        else:
            self.bn = None
        
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
        # batch_flat of dimension (n_element, n_el_per_set, in_dim)
        batch_flat = batch.view(-1, self.in_dim, 1)
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

        if self.bn is not None:
            shape = output.size()
            output = output.reshape(-1, shape[-1])   # [prod(***), D_out]
            output = self.bn(output)                 # [prod(***), D_out]
            output = output.reshape(shape)           # [***, D_out]

        return output