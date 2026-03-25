import torch.nn as nn
from src.models.components import MLP

class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.net = MLP(
            input_dim=cfg.input_dim,
            output_dim=cfg.output_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            activation=cfg.activation
        )

    def forward(self, x):
        return self.net(x)