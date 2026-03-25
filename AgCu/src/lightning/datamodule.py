import pytorch_lightning as pl
from torch.utils.data import DataLoader
from data.dataset import PhaseDataset

class PhaseDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        self.dataset = PhaseDataset(self.cfg.data)

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=self.cfg.data.shuffle
        )