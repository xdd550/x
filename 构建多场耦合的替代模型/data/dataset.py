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
        # 改进：界面感知重采样 (Interface-Aware Resampling)
        # =========================
        if cfg.num_samples < len(x):
            # 1. 找出界面区域 (0 < p < 1) 和纯相/无用区域 (p 趋近于 0 或 1)
            # 考虑到数值截断误差，使用 0.01 和 0.99 作为界限
            interface_mask = (p[:, 0] > 0.01) & (p[:, 0] < 0.99)
            bulk_mask = ~interface_mask

            interface_indices = np.where(interface_mask)[0]
            bulk_indices = np.where(bulk_mask)[0]

            # 2. 设定采样配比：强烈向界面倾斜 (例如 70% 界面点，30% 背景点)
            # 论文写作点：这被称为 "Hard Negative Mining" 或重点采样机制
            num_interface = int(cfg.num_samples * 0.70)
            num_bulk = cfg.num_samples - num_interface

            # 如果数据中真实的界面点总数不够配额，则全取，剩下的名额补给背景点
            if len(interface_indices) < num_interface:
                num_interface = len(interface_indices)
                num_bulk = cfg.num_samples - num_interface

            # 3. 分别进行无放回随机采样
            sampled_interface = np.random.choice(interface_indices, num_interface, replace=False)
            sampled_bulk = np.random.choice(bulk_indices, num_bulk, replace=False)

            # 4. 合并并打乱最终索引
            idx = np.concatenate([sampled_interface, sampled_bulk])
            np.random.shuffle(idx)

            x, y, t, p, c = x[idx], y[idx], t[idx], p[idx], c[idx]

        # =========================
        # 归一化 (保持你原来的逻辑)
        # =========================
        if cfg.normalize:
            x = (x - x.min()) / (x.max() - x.min())
            y = (y - y.min()) / (y.max() - y.min())
            t = (t - t.min()) / (t.max() - t.min())

        # ... (后续保留你原来的加噪声和转 tensor 的代码)
        if cfg.add_noise:
            noise = cfg.noise_level * np.std(p) * np.random.randn(*p.shape)
            p = p + noise

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