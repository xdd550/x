import torch.nn as nn
from src.models.components import MLP, FourierFeatureTransform


class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 1. 设定傅里叶映射参数
        # 默认将 3 维输入 (x,y,t) 映射到 64 个高频基底上
        mapping_size = 64
        self.feature_map = FourierFeatureTransform(
            input_dim=cfg.input_dim,
            mapping_size=mapping_size,
            scale=10.0  # 你可以后续调整这个 scale 寻找最佳的界面锐度
        )

        # 2. 计算映射后的真实输入维度
        # 因为拼接了 sin 和 cos，所以维度是 mapping_size 的两倍
        encoded_dim = mapping_size * 2

        # 3. 初始化 MLP，注意这里的 input_dim 使用的是编码后的 encoded_dim
        self.net = MLP(
            input_dim=encoded_dim,
            output_dim=cfg.output_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            activation=cfg.activation
        )

    def forward(self, x):
        # 数据流向：原始坐标 -> 高频映射 -> 深度神经网络
        x_encoded = self.feature_map(x)
        return self.net(x_encoded)