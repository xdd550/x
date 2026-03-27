"""
“物理常数管理中心
需要让神经网络像猜测盲盒一样，把未知的物理参数（如 D_s, D_l, gamma）当成权重一样去训练、去更新。
这个文件极其优雅地处理了“哪些参数要固定”和“哪些参数要学习”的逻辑。
"""

import torch
import torch.nn as nn


class PhysicalParameters(nn.Module):
    #它就可以像神经网络层一样，被优化器捕捉并更新其内部的 Parameter
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # =========================
        # γ（各向异性参数）
        # =========================

        # 如果开启了：把它包装成 nn.Parameter！
        # 这是一个带梯度的张量，优化器（Adam）在反向传播时会自动更新它的数值
        if cfg.inverse.learn_gamma:
            self.gamma = nn.Parameter(
                torch.tensor(cfg.inverse.gamma_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "gamma",
                torch.tensor(cfg.physics.gamma, dtype=torch.float32)
            )

        # =========================
        # 扩散系数 Ds
        # =========================
        if cfg.inverse.learn_Ds:
            self.Ds = nn.Parameter(
                torch.tensor(cfg.inverse.Ds_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "Ds",
                torch.tensor(cfg.physics.Ds, dtype=torch.float32)
            )

        # =========================
        # 扩散系数 Dl
        # =========================
        if cfg.inverse.learn_Dl:
            self.Dl = nn.Parameter(
                torch.tensor(cfg.inverse.Dl_init, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "Dl",
                torch.tensor(cfg.physics.Dl, dtype=torch.float32)
            )

        # 常数（不训练）
        self.register_buffer(
            "R",
            torch.tensor(cfg.physics.R, dtype=torch.float32)
        )

    # =========================
    # 参数约束（防止发散）
    # =========================
    def clamp_parameters(self):
        # 检查是否开启了安全锁
        if self.cfg.inverse.clamp:

            if hasattr(self, "gamma"):
                # 检查对象里目前有没有 gamma 这个属性
                self.gamma.data = torch.clamp(
                    self.gamma.data,
                    self.cfg.inverse.gamma_min,
                    self.cfg.inverse.gamma_max
                )

                """
                # torch.clamp 的作用：把 self.gamma 的数值强行限制在 [gamma_min, gamma_max] 之间
                # 比如 gamma_min 是 0，如果某一步优化器用力过猛，把 gamma 减成了 -0.01，
                # clamp 会立刻把它掰回 0，防止物理逻辑崩溃。
                """
                # 2. 约束固相扩散系数 Ds
                # 读取 config.yaml 中的 Ds_min 和 Ds_max 进行截断
                if hasattr(self, "Ds"):
                    self.Ds.data = torch.clamp(
                        self.Ds.data,
                        self.cfg.inverse.Ds_min,
                        self.cfg.inverse.Ds_max
                    )

                # 3. 约束液相扩散系数 Dl
                # 读取 config.yaml 中的 Dl_min 和 Dl_max 进行截断
                if hasattr(self, "Dl"):
                    self.Dl.data = torch.clamp(
                        self.Dl.data,
                        self.cfg.inverse.Dl_min,
                        self.cfg.inverse.Dl_max
                    )