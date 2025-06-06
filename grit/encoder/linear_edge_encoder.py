import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_edge_encoder


@register_edge_encoder('LinearEdge')
class LinearEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        if cfg.dataset.name in ['MNIST', 'CIFAR10'] or cfg.dataset.name.startswith('ogbl-'):
            self.in_dim = 1
        elif cfg.dataset.name.startswith('attributed_triangle-'):
            self.in_dim = 2
        elif cfg.dataset.name.startswith('ogbg-'):
            self.in_dim = 3
        else:
            raise ValueError("Input edge feature dim is required to be hardset "
                             "or refactored to use a cfg option.")
        self.encoder = torch.nn.Linear(self.in_dim, emb_dim)

    def forward(self, batch):
        if batch.x.dtype != float:
            batch.x = batch.x.float()
        batch.edge_attr = self.encoder(batch.edge_attr.view(-1, self.in_dim)) # convert to float (maybe inneficient)
        return batch
