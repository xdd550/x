"""
物理数学工具
"""
# src/utils/math_utils.py

import torch

def compute_gradients(outputs, inputs):
    """
    计算 outputs 对 inputs 的一阶偏导数。
    利用 PyTorch 的计算图自动微分，支持求高阶导数。
    """
    grads = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,  # 必须开启！为了后续能继续算二阶、四阶导数
        retain_graph=True
    )[0]
    return grads