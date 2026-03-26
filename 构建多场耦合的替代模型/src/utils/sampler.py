import torch

def sample_collocation(num_points, device):
    x = torch.rand(num_points, 1, device=device)
    y = torch.rand(num_points, 1, device=device)
    t = torch.rand(num_points, 1, device=device)

    return torch.cat([x, y, t], dim=1)