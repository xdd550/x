"""
把杂乱的数据加载、预处理、以及打包成 Batch 的逻辑全部封装在一起，
让你的训练主代码（train.py）保持极致的干净。
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataset import PhaseDataset


class PhaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg  # 把百宝箱 cfg 存下来备用

    def setup(self, stage=None):
        # 实例化我们之前写的那个带有“重点采样”机制的 Dataset
        self.dataset = PhaseDataset(self.cfg.data)

    def train_dataloader(self):
        # 把 Dataset 包装进 DataLoader，变成可以按批次输出的流水线
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.data.shuffle  # 是否打乱顺序
        )
