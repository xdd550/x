import pytorch_lightning as pl
import torch

from src.losses.loss_fn import data_loss
from src.models.pinn import PINN
from src.physics.parameters import PhysicalParameters
from src.physics.pde import compute_pde_residual
from src.utils.sampler import sample_collocation


class PINNModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = PINN(cfg.model)

        # 可学习参数
        self.params = PhysicalParameters(cfg)

        # 新增：自适应损失权重的可学习参数
        # 初始化 4 个标量（对应 Data, PDE, IC, BC）
        # 这里存储的是方差的对数 (log_sigma^2)，以确保训练的数值稳定性
        # =========================
        self.log_vars = torch.nn.Parameter(torch.zeros(4))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, t, p_true, c_true = batch

        inputs = torch.cat([x, y, t], dim=1)
        inputs.requires_grad_(True)

        pred = self(inputs)
        p_pred = pred[:, 0:1]
        c_pred = pred[:, 1:2]

        # =========================
        # 1. 数据损失 (Data Loss)
        # =========================
        loss_data_p = data_loss(p_pred, p_true)
        loss_data_c = data_loss(c_pred, c_true)
        loss_data_val = loss_data_p + loss_data_c

        # =========================
        # 2. 初始条件损失 (IC Loss) - 解决初始点过少的问题
        # =========================
        # 自动从当前 batch 中筛选出初始时刻 (t=0) 的点
        # 因为数据可能被归一化，我们用一个极小的阈值来判断 t=0
        ic_mask = (t < 1e-5).squeeze()
        if ic_mask.sum() > 0:
            # 如果当前 batch 里有初始点，单独给它们计算损失并加倍惩罚
            loss_ic = data_loss(p_pred[ic_mask], p_true[ic_mask]) + \
                      data_loss(c_pred[ic_mask], c_true[ic_mask])
        else:
            loss_ic = torch.tensor(0.0, device=self.device)

        # =========================
        # 3. 物理残差损失 (PDE Loss)
        # =========================
        collocation = sample_collocation(
            self.cfg.sampling.num_collocation,
            self.device
        )
        collocation.requires_grad_(True)
        pred_col = self(collocation)
        loss_pde_val = compute_pde_residual(collocation, pred_col, self.params)

        # =========================
        # 4. 边界条件损失 (BC Loss) - 对齐 C++ 的零通量边界
        # =========================
        # 随机在四个边界(x=0, x=1, y=0, y=1)各采样 500 个点
        num_bc = 500
        t_bc = torch.rand(num_bc, 1, device=self.device)

        # x 边界 (左 x=0, 右 x=1)
        y_rand = torch.rand(num_bc, 1, device=self.device)
        x_left = torch.zeros(num_bc, 1, device=self.device, requires_grad=True)
        x_right = torch.ones(num_bc, 1, device=self.device, requires_grad=True)

        # y 边界 (下 y=0, 上 y=1)
        x_rand = torch.rand(num_bc, 1, device=self.device)
        y_bottom = torch.zeros(num_bc, 1, device=self.device, requires_grad=True)
        y_top = torch.ones(num_bc, 1, device=self.device, requires_grad=True)

        # 组装边界坐标
        pts_left = torch.cat([x_left, y_rand, t_bc], dim=1)
        pts_right = torch.cat([x_right, y_rand, t_bc], dim=1)
        pts_bottom = torch.cat([x_rand, y_bottom, t_bc], dim=1)
        pts_top = torch.cat([x_rand, y_top, t_bc], dim=1)

        # 计算边界上的网络预测
        pred_left = self(pts_left)
        pred_right = self(pts_right)
        pred_bottom = self(pts_bottom)
        pred_top = self(pts_top)

        # 计算对法线方向的导数 (要求等于 0)
        dp_dx_left = torch.autograd.grad(pred_left[:, 0:1], x_left, grad_outputs=torch.ones_like(pred_left[:, 0:1]),
                                         create_graph=True)[0]
        dc_dx_left = torch.autograd.grad(pred_left[:, 1:2], x_left, grad_outputs=torch.ones_like(pred_left[:, 1:2]),
                                         create_graph=True)[0]

        dp_dx_right = torch.autograd.grad(pred_right[:, 0:1], x_right, grad_outputs=torch.ones_like(pred_right[:, 0:1]),
                                          create_graph=True)[0]
        dc_dx_right = torch.autograd.grad(pred_right[:, 1:2], x_right, grad_outputs=torch.ones_like(pred_right[:, 1:2]),
                                          create_graph=True)[0]

        dp_dy_bottom = \
            torch.autograd.grad(pred_bottom[:, 0:1], y_bottom, grad_outputs=torch.ones_like(pred_bottom[:, 0:1]),
                                create_graph=True)[0]
        dc_dy_bottom = \
            torch.autograd.grad(pred_bottom[:, 1:2], y_bottom, grad_outputs=torch.ones_like(pred_bottom[:, 1:2]),
                                create_graph=True)[0]

        dp_dy_top = \
            torch.autograd.grad(pred_top[:, 0:1], y_top, grad_outputs=torch.ones_like(pred_top[:, 0:1]),
                                create_graph=True)[
                0]
        dc_dy_top = \
            torch.autograd.grad(pred_top[:, 1:2], y_top, grad_outputs=torch.ones_like(pred_top[:, 1:2]),
                                create_graph=True)[
                0]

        # 整合所有边界的梯度平方和
        loss_bc = torch.mean(dp_dx_left ** 2 + dc_dx_left ** 2) + \
                  torch.mean(dp_dx_right ** 2 + dc_dx_right ** 2) + \
                  torch.mean(dp_dy_bottom ** 2 + dc_dy_bottom ** 2) + \
                  torch.mean(dp_dy_top ** 2 + dc_dy_top ** 2)

        # =========================
        # 5. 总损失组合 (自适应不确定性加权)
        # =========================
        # 提取各个损失项的动态方差 (相当于动态除以权重)
        precision_data = torch.exp(-self.log_vars[0])
        precision_pde = torch.exp(-self.log_vars[1])
        precision_ic = torch.exp(-self.log_vars[2])
        precision_bc = torch.exp(-self.log_vars[3])

        # 动态组装 Loss：L_total = sum( L_i * exp(-log_var_i) + log_var_i )
        loss = (
                (precision_data * loss_data_val + self.log_vars[0]) +
                (precision_pde * loss_pde_val + self.log_vars[1]) +
                (precision_ic * loss_ic + self.log_vars[2]) +
                (precision_bc * loss_bc + self.log_vars[3])
        )

        # logging (现在不用手动调 yaml 里的 lambda 了)
        self.log("loss", loss)
        self.log("loss_data", loss_data_val)
        self.log("loss_pde", loss_pde_val)
        self.log("loss_ic", loss_ic)
        self.log("loss_bc", loss_bc)

        # 记录自适应权重的变化轨迹 (发论图表必备)
        self.log("weight_data", precision_data)
        self.log("weight_pde", precision_pde)
        self.log("weight_ic", precision_ic)
        self.log("weight_bc", precision_bc)

        if self.cfg.inverse.learn_gamma:
            self.log("gamma", self.params.gamma)

        return loss

    def on_after_backward(self):
        self.params.clamp_parameters()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.train.lr)
