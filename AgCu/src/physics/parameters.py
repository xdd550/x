import torch
import torch.nn as nn


class PhysicalParameters(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # =========================
        # γ（各向异性参数）
        # =========================
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
        if self.cfg.inverse.clamp:

            if hasattr(self, "gamma"):
                self.gamma.data = torch.clamp(
                    self.gamma.data,
                    self.cfg.inverse.gamma_min,
                    self.cfg.inverse.gamma_max
                )