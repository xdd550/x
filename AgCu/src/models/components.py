import torch.nn as nn

def get_activation(name):
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError("Unknown activation")


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, activation):
        super().__init__()

        layers = []
        act = get_activation(activation)

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act)

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act)

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)