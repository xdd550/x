import torch

def data_loss(p_pred, p_true):
    return torch.mean((p_pred - p_true) ** 2)


def total_loss(cfg, loss_data, loss_pde):
    return (
        cfg.loss.lambda_data * loss_data +
        cfg.loss.lambda_pde * loss_pde
    )