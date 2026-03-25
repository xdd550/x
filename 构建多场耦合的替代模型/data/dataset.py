import numpy as np
import torch
from torch.utils.data import Dataset

class PhaseDataset(Dataset):
    def __init__(self, cfg):
        data = np.load(cfg.path)

        x = data['x']
        y = data['y']
        t = data['t']
        p = data['p']
        c = data['c']

        # =========================
        # 采样
        # =========================
        if cfg.num_samples < len(x):
            idx = np.random.choice(len(x), cfg.num_samples, replace=False)
            x, y, t, p, c = x[idx], y[idx], t[idx], p[idx], c[idx]

        # =========================
        # 归一化
        # =========================
        if cfg.normalize:
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            t = (t - t.min()) / (t.max() - t.min())

        # =========================
        # 加噪声（可选）
        # =========================
        if cfg.add_noise:
            noise = cfg.noise_level * np.std(p) * np.random.randn(*p.shape)
            p = p + noise

        # 转tensor
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
            self.t[idx],
            self.p[idx],
            self.c[idx],
        )