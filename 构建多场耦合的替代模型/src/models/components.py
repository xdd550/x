"""
定义了所使用的物理信息神经网络（PINN）的底层建筑模块
傅里叶特征映射
"""
import torch
import torch.nn as nn


# 1. 激活函数选择器
def get_activation(name):
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError("Unknown activation")


# ==========================================
# 2. 傅里叶特征映射层 (核心黑科技)
# ==========================================
class FourierFeatureTransform(nn.Module):
    """
    傅里叶特征映射层：解决 PINN 拟合高频相界面时的谱偏差问题
    """

    def __init__(self, input_dim, mapping_size, scale):
        super().__init__()
        self.input_dim = input_dim  # 输入维度（通常是 x, y, t，即 3维）
        self.mapping_size = mapping_size  # 映射后的基底数量（例如 64）
        # 生成一个固定的随机高斯矩阵作为频率基底。注意 requires_grad=False，它是不参与训练的常量
        # scale 参数控制频率跨度，值越大，捕捉陡峭界面的能力越强
        self.B = nn.Parameter(torch.randn(input_dim, mapping_size) * scale, requires_grad=False)

    def forward(self, x):
        # x 的维度: [N, input_dim]
        # 矩阵乘法投影到高频空间
        x_proj = 2 * torch.pi * x @ self.B
        # 拼接 sin 和 cos，输出维度变为 mapping_size * 2
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# ==========================================
# 3. 多层感知机 (MLP) 骨干网络
# ==========================================
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
