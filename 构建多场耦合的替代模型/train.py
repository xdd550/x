import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch

from src.lightning.module import PINNModule
from src.lightning.datamodule import PhaseDataModule
from src.utils.callbacks import VisualizerCallback

# ==========================================
# 新增：自定义回调，在每个 Epoch 结束时打印各项监控指标
# ==========================================
class PrintMetricsCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 获取 module.py 中 self.log() 记录的所有指标字典
        metrics = trainer.callback_metrics
        epoch = trainer.current_epoch

        # 提取 Loss (注意这里的键名必须和 module.py 中 self.log 的名字一致)
        # 如果字典里还没记录到，则默认给 0.0
        loss_tot = metrics.get('train/loss', 0.0)
        loss_data = metrics.get('train/loss_data', 0.0)
        loss_pde = metrics.get('train/loss_pde', 0.0)
        loss_bc = metrics.get('train/loss_bc', 0.0)

        # 提取反演出来的物理参数
        gamma = metrics.get('params/gamma', 0.0)
        Ds = metrics.get('params/Ds', 0.0)
        Dl = metrics.get('params/Dl', 0.0)

        # 在终端打印格式化的输出 (使用 .4e 科学计数法，方便观察极小数值的变化)
        print(f"\n[Epoch {epoch:04d}] "
              f"Tot: {loss_tot:.4e} | "
              f"Data: {loss_data:.4e} | "
              f"PDE: {loss_pde:.4e} | "
              f"BC: {loss_bc:.4e} || "
              f"Gamma: {gamma:.4e} | "
              f"Ds: {Ds:.4e} | "
              f"Dl: {Dl:.4e}")


def main():
    # ⬇️ 加上这一行，牺牲极微小的精度，换取最高 2-3 倍的训练提速！
    torch.set_float32_matmul_precision('high')

    cfg = OmegaConf.load("configs/config.yaml")

    # 初始化数据和模型

    datamodule = PhaseDataModule(cfg)
    model = PINNModule(cfg)

    # 实例化回调
    vis_callback = VisualizerCallback(resolution=256)
    #  实例化打印工具
    print_callback = PrintMetricsCallback()

    trainer = pl.Trainer(
        max_epochs=cfg.train.max_epochs,

        accelerator="gpu" if cfg.device == "gpu" else "cpu",
        devices=1,
        # ==========================================
        # 新增：将刚才写的可视化打印回调函数挂载到 Trainer 上
        # ==========================================
        callbacks=[print_callback, vis_callback],
        log_every_n_steps=1  # 确保每步都产生日志，防止因为步数太少而不打印
    )

    trainer.fit(model, datamodule)


if __name__ == "__main__":
    main()