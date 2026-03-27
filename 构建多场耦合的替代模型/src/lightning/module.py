
import pytorch_lightning as pl
import torch

# 导入各个组件：损失函数、网络结构、物理参数、偏微分方程残差计算、采样器
from src.losses.loss_fn import data_loss
from src.models.pinn import PINN
from src.physics.parameters import PhysicalParameters
from src.physics.pde import compute_pde_residual
from src.utils.sampler import sample_collocation

class PINNModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = PINN(cfg.model) #实例化 PINN 神经网络

        # 可学习的物理参数 (gamma, Ds, Dl 等)
        self.params = PhysicalParameters(cfg) #实例化了物理参数模块，包含了待反演的 D_s、D_l 和 gamma。

        # 自适应损失权重的可学习参数 (Data, PDE, IC, BC)
        # 它们的量级可能相差几万倍，手动调参不可能调得好。
        # 初始值为0，表示初始权重为1
        self.log_vars = torch.nn.Parameter(torch.zeros(4))

    # 前向传播：把时空坐标扔进网络，吐出预测的相场 p 和浓度场 c
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # 1. 从 datamodule 送来的包裹里拆出数据
        x, y, t, p_true, c_true = batch

        # 将 x, y, t 拼接成 [N, 3] 的张量作为输入
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
        # 2. 初始条件损失 (IC Loss)
        # =========================
        ic_mask = (t < 1e-5).squeeze()#通过判定 t 约为 0（小于 1e-5）筛选出位于初始时刻的点。
        if ic_mask.sum() > 0:
            # 如果当前这批数据里恰好有初始点，我们就对这些点单独施加一次强约束
            loss_ic = data_loss(p_pred[ic_mask], p_true[ic_mask]) + \
                      data_loss(c_pred[ic_mask], c_true[ic_mask])
        else:
            loss_ic = torch.tensor(0.0, device=self.device)

        # =========================
        # 3. 物理残差损失 (PDE Loss)
        # =========================
        # 在整个时空域内随机“盲目”地撒一批配点 (Collocation Points)
        collocation = sample_collocation(
            self.cfg.sampling.num_collocation,
            self.device,
            model=self.model,
            cfg=self.cfg,
            current_epoch=self.current_epoch
        )
        collocation.requires_grad_(True)
        pred_col = self(collocation)# 让网络预测这些盲点的值
        loss_pde_val = compute_pde_residual(collocation, pred_col, self.params)
        # 把盲点的坐标、预测值、以及待反演的参数，一起送进 pde.py 里计算物理残差！

        # =========================
        # 4. 边界条件损失 (BC Loss)
        # =========================
        num_bc = 500
        t_bc = torch.rand(num_bc, 1, device=self.device)# 随机生成时间 t

        # 随机在 x=0 (左), x=1 (右), y=0 (下), y=1 (上) 四个边界上撒点
        y_rand = torch.rand(num_bc, 1, device=self.device)
        x_left = torch.zeros(num_bc, 1, device=self.device, requires_grad=True)
        x_right = torch.ones(num_bc, 1, device=self.device, requires_grad=True)

        x_rand = torch.rand(num_bc, 1, device=self.device)
        y_bottom = torch.zeros(num_bc, 1, device=self.device, requires_grad=True)
        y_top = torch.ones(num_bc, 1, device=self.device, requires_grad=True)

        pts_left = torch.cat([x_left, y_rand, t_bc], dim=1)
        pts_right = torch.cat([x_right, y_rand, t_bc], dim=1)
        pts_bottom = torch.cat([x_rand, y_bottom, t_bc], dim=1)
        pts_top = torch.cat([x_rand, y_top, t_bc], dim=1)

        pred_left = self(pts_left)
        pred_right = self(pts_right)
        pred_bottom = self(pts_bottom)
        pred_top = self(pts_top)

        # 【核心物理约束】：计算预测场相对于边界法线方向的导数
        # 在真实物理实验或C++模拟中，边界通常是封闭的，物质不能流出，这就是 Neumann 边界（偏导数为0）。
        dp_dx_left = torch.autograd.grad(pred_left[:, 0:1], x_left, grad_outputs=torch.ones_like(pred_left[:, 0:1]), create_graph=True)[0]
        dc_dx_left = torch.autograd.grad(pred_left[:, 1:2], x_left, grad_outputs=torch.ones_like(pred_left[:, 1:2]), create_graph=True)[0]

        dp_dx_right = torch.autograd.grad(pred_right[:, 0:1], x_right, grad_outputs=torch.ones_like(pred_right[:, 0:1]), create_graph=True)[0]
        dc_dx_right = torch.autograd.grad(pred_right[:, 1:2], x_right, grad_outputs=torch.ones_like(pred_right[:, 1:2]), create_graph=True)[0]

        dp_dy_bottom = torch.autograd.grad(pred_bottom[:, 0:1], y_bottom, grad_outputs=torch.ones_like(pred_bottom[:, 0:1]), create_graph=True)[0]
        dc_dy_bottom = torch.autograd.grad(pred_bottom[:, 1:2], y_bottom, grad_outputs=torch.ones_like(pred_bottom[:, 1:2]), create_graph=True)[0]

        dp_dy_top = torch.autograd.grad(pred_top[:, 0:1], y_top, grad_outputs=torch.ones_like(pred_top[:, 0:1]), create_graph=True)[0]
        dc_dy_top = torch.autograd.grad(pred_top[:, 1:2], y_top, grad_outputs=torch.ones_like(pred_top[:, 1:2]), create_graph=True)[0]

        loss_bc = torch.mean(dp_dx_left ** 2 + dc_dx_left ** 2) + \
                  torch.mean(dp_dx_right ** 2 + dc_dx_right ** 2) + \
                  torch.mean(dp_dy_bottom ** 2 + dc_dy_bottom ** 2) + \
                  torch.mean(dp_dy_top ** 2 + dc_dy_top ** 2)

        # =========================
        # 5. 总损失组合 (自适应权重)
        # =========================
        precision_data = torch.exp(-self.log_vars[0])
        precision_pde = torch.exp(-self.log_vars[1])
        precision_ic = torch.exp(-self.log_vars[2])
        precision_bc = torch.exp(-self.log_vars[3])

        loss = (
                (precision_data * loss_data_val + self.log_vars[0]) +
                (precision_pde * loss_pde_val + self.log_vars[1]) +
                (precision_ic * loss_ic + self.log_vars[2]) +
                (precision_bc * loss_bc + self.log_vars[3])
        )
        """
        # 【神奇的自适应公式】：如果某个 loss 太大降不下去，网络会增大对应的 log_vars，
        # 从而减小 precision，让这个 loss 不至于产生爆炸的梯度毁掉整个网络；
         # 同时，公式尾部加上的 log_vars 又会防止网络无脑把 precision 缩到 0。
        """

        # 记录日志，on_epoch=True 保证在每个 Epoch 结束时计算平均值并打印
        # 必须加上 on_step=False, on_epoch=True
        # 这样 Lightning 才会帮我们把整个 Epoch 的值平均下来，回调函数才能拿到数据！
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_data", loss_data_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_pde", loss_pde_val, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_ic", loss_ic, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_bc", loss_bc, on_step=False, on_epoch=True, prog_bar=True)

        # 物理参数也是一样，需要 on_epoch=True
        if self.cfg.inverse.learn_gamma:
            self.log("params/gamma", self.params.gamma, on_step=False, on_epoch=True)
        if self.cfg.inverse.learn_Ds:
            self.log("params/Ds", self.params.Ds, on_step=False, on_epoch=True)
        if self.cfg.inverse.learn_Dl:
            self.log("params/Dl", self.params.Dl, on_step=False, on_epoch=True)

        return loss

    # 【修复1】：必须在 optimizer 更新参数之"后"进行截断，所以用 on_train_batch_end
    # 我们立刻介入，检查反演的 Ds, Dl 和 gamma 有没有被优化器改成负数。
    # 如果有，立刻用 clamp_parameters() 把它拉回到合法区间。
    def on_train_batch_end(self, outputs, batch, batch_idx):
        with torch.no_grad():
            self.params.clamp_parameters()

    # 【修复2】：为物理参数分配单独的（更大的）学习率
    def configure_optimizers(self):
        # 1. 配置优化器参数组 (分离神经网络和物理反演参数)
        base_params = list(self.model.parameters()) + [self.log_vars]
        physics_params = list(self.params.parameters())

        optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': self.cfg.train.lr},
            {'params': physics_params, 'lr': self.cfg.train.lr * 10.0}  # 物理参数保持 10 倍学习率
        ])

        # 2. 检查 YAML 配置中是否开启了学习率衰减
        if self.cfg.train.lr_scheduler:
            # 引入 PyTorch 的 StepLR
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.cfg.train.step_size,  # 5000
                gamma=self.cfg.train.gamma  # 0.5
            )

            # 在 Lightning 中，必须明确指出是按 "step" 更新，否则默认按 "epoch" 算！
            # 如果按 epoch 算，5000 个 epoch 才衰减一次，而你的总 epoch 只有 1000，那就永远不会衰减了。
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # 关键点：告诉 Lightning 每跑完一个 batch 就计一个 step
                    "frequency": 1
                }
            }

        # 如果 yaml 里关掉了 lr_scheduler，就只返回优化器
        return optimizer