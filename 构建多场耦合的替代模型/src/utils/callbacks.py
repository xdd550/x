"""
自动将网络的预测结果（相场 p 和浓度场 c）渲染成图像并保存到磁盘上。
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.callbacks import Callback
import os

class VisualizerCallback(Callback):
    def __init__(self, resolution=256):
        super().__init__()
        self.res = resolution  # 图像分辨率，256x256 足够看清枝晶了

    @torch.no_grad()
    def on_train_epoch_end(self, trainer, pl_module):
        # 每 100 轮保存一次图像，避免产生太多垃圾文件
        if trainer.current_epoch % 100 != 0:
            return

        # 1. 在 [0, 1] 范围内建立 2D 坐标网格
        x = torch.linspace(0, 1, self.res)
        y = torch.linspace(0, 1, self.res)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # 2. 选一个固定的时间点 t（例如 t=0.5，即演化到中期）
        t = torch.ones_like(grid_x) * 0.5

        # 拼接成模型需要的输入格式 [Batch, 3] -> (x, y, t)
        flat_x = grid_x.reshape(-1, 1)
        flat_y = grid_y.reshape(-1, 1)
        flat_t = t.reshape(-1, 1)
        inputs = torch.cat([flat_x, flat_y, flat_t], dim=1).to(pl_module.device)

        # 3. 模型推理
        pred = pl_module(inputs)
        p_pred = pred[:, 0].reshape(self.res, self.res).cpu().numpy()  # 预测相场
        c_pred = pred[:, 1].reshape(self.res, self.res).cpu().numpy()  # 预测浓度场

        # 4. 使用 Matplotlib 画图
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # 画相场图 (Dendrite Morphology)
        im1 = ax[0].imshow(p_pred, extent=[0, 1, 0, 1], origin='lower', cmap='viridis')
        ax[0].set_title(f"Phase Field (Epoch {trainer.current_epoch})")
        fig.colorbar(im1, ax=ax[0])

        # 画浓度场图 (Solute Distribution)
        im2 = ax[1].imshow(c_pred, extent=[0, 1, 0, 1], origin='lower', cmap='plasma')
        ax[1].set_title("Concentration Field")
        fig.colorbar(im2, ax=ax[1])

        # 5. 保存到当前版本专属的日志文件夹中
        # 获取当前实验的专属路径 (例如 lightning_logs/version_7)
        save_dir = trainer.log_dir if trainer.log_dir else "."
        save_path = os.path.join(save_dir, f"vis_epoch_{trainer.current_epoch}.png")

        plt.savefig(save_path)
        plt.close()