import torch
import pytorch_lightning as pl

from src.models.pinn import PINN
from src.physics.pde import compute_pde_residual
from src.losses.loss_fn import data_loss, total_loss
from src.utils.sampler import sample_collocation
from src.physics.parameters import PhysicalParameters
class PINNModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = PINN(cfg.model)

        # 可学习参数
        self.params = PhysicalParameters(cfg)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, t, p_true, c_true = batch

        inputs = torch.cat([x, y, t], dim=1)
        inputs.requires_grad_(True)

        pred = self(inputs)
        p_pred = pred[:, 0:1]

        # =========================
        # 数据损失
        # =========================
        loss_data_val = data_loss(p_pred, p_true)

        # =========================
        # PDE损失（用额外采样点）
        # =========================
        collocation = sample_collocation(
            self.cfg.sampling.num_collocation,
            self.device
        )
        collocation.requires_grad_(True)

        pred_col = self(collocation)

        loss_pde_val = compute_pde_residual(
            collocation, pred_col, self.params
        )

        # =========================
        # 总损失
        # =========================
        loss = total_loss(self.cfg, loss_data_val, loss_pde_val)

        # logging
        self.log("loss", loss)
        self.log("loss_data", loss_data_val)
        self.log("loss_pde", loss_pde_val)

        if self.cfg.inverse.learn_gamma:
            self.log("gamma", self.params.gamma)

        return loss

    def on_after_backward(self):
        self.params.clamp()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)