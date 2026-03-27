import torch.nn as nn
from src.models.components import MLP, FourierFeatureTransform


class PINN(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # 1. 设定傅里叶映射参数
        # 默认将 3 维输入 (x,y,t) 映射到 64 个高频基底上
        mapping_size = 64
        # 实例化傅里叶映射层
        self.feature_map = FourierFeatureTransform(
            input_dim=cfg.input_dim,
            mapping_size=mapping_size,
            scale=cfg.fourier_scale  # 你可以后续调整这个 scale 寻找最佳的界面锐度
        )

        # 2. 计算映射后的真实输入维度
        # 因为拼接了 sin 和 cos，所以维度是 mapping_size 的两倍
        # 所以进入深度神经网络的实际特征维度是 64 * 2 = 128 维
        encoded_dim = mapping_size * 2

        # 3. 初始化 MLP，注意这里的 input_dim 使用的是编码后的 encoded_dim
        self.net = MLP(
            input_dim=encoded_dim,#这里接收的不再是 3 维的 (x,y,t)，而是经过高频放大的 128 维特征！
            output_dim=cfg.output_dim,#2 (预测相场 p 和 浓度场 c)
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            activation=cfg.activation
        )

    def forward(self, x):
        # 数据流向：原始坐标 -> 高频映射 -> 深度神经网络

        # 把原始的 [Batch_size, 3] 维坐标 (x, y, t) 送入傅里叶映射层
        # 输出 x_encoded 变成了 [Batch_size, 128] 维的高频波动特征
        x_encoded = self.feature_map(x)

        # 把这 128 维的特征送入深度 MLP 网络进行复杂的物理规律学习，
        # 最终吐出 [Batch_size, 2] 的预测结果，也就是每一个时空点对应的相场 p 和浓度场 c
        return self.net(x_encoded)