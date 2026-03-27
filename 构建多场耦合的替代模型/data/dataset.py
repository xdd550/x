import numpy as np
import torch
from torch.utils.data import Dataset

class PhaseDataset(Dataset):
    def __init__(self, cfg):
        #  加载磁盘上的 npz 数据文件
        data = np.load(cfg.path)

        x = data['x']
        y = data['y']
        t = data['t']
        p = data['p']  # 相场标签值 (0~1之间)
        c = data['c']

        # =========================
        # 改进：界面感知重采样 (Interface-Aware Resampling)
        # 这也是论文中极具创新性的一点：针对相场问题的重点采样机制
        # =========================
        if cfg.num_samples < len(x):
            # 1. 找出界面区域 (0 < p < 1) 和纯相/无用区域 (p 趋近于 0 或 1)
            # 考虑到数值截断误差，使用 0.01 和 0.99 作为界限
            interface_mask = (p[:, 0] > 0.01) & (p[:, 0] < 0.99)
            bulk_mask = ~interface_mask # 剩下的就是纯固相和纯液相（背景区域）

            # 获取这两类区域在数组中的具体行索引
            interface_indices = np.where(interface_mask)[0]
            bulk_indices = np.where(bulk_mask)[0]

            # 2. 设定采样配比：强烈向界面倾斜 (例如 70% 界面点，30% 背景点)
            # 论文写作点：这被称为 "Hard Negative Mining" 或重点采样机制
            # 强制要求每次采样的点中，有 70% 必须落在又薄又锐利的界面上！
            # 剩下 30% 落在背景区域，以维持全局宏观场的稳定
            num_interface = int(cfg.num_samples * 0.70)
            num_bulk = cfg.num_samples - num_interface
            """
            强制规定每一轮喂给网络的 cfg.num_samples（比如 50000 个）点中，必须有 70% 是从那层极薄的界面上抠下来的。
            剩下的 30% 分给庞大的纯相背景，用来维持模型对宏观大环境的认知。
            """
            # 如果数据中真实的界面点总数不够配额，则全取，剩下的名额补给背景点
            if len(interface_indices) < num_interface:
                num_interface = len(interface_indices)
                num_bulk = cfg.num_samples - num_interface

            # 3. 分别进行无放回随机采样
            sampled_interface = np.random.choice(interface_indices, num_interface, replace=False)
            sampled_bulk = np.random.choice(bulk_indices, num_bulk, replace=False)
            """
            replace=False：表示无放回抽样。抽过这道题，这次就不会再抽到，保证喂给模型的数据尽可能多样化。
            np.random.shuffle(idx)：将抽出来的界面点和背景点混在一起，然后像洗牌一样彻底打乱。
            
            """
            # 4. 合并并打乱最终索引    合并这两拨抽出来的索引，并随机打乱它们的顺序
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


        if cfg.add_noise:
            # 根据相场数据的标准差和设定的 noise_level 生成随机高斯噪声
            noise = cfg.noise_level * np.std(p) * np.random.randn(*p.shape)
            p = p + noise

        # 将所有的 numpy 数组转换为 PyTorch 的张量 (Tensor)，数据类型指定为 float32
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
        self.p = torch.tensor(p, dtype=torch.float32)
        self.c = torch.tensor(c, dtype=torch.float32)

    def __len__(self):
        # 告诉 DataLoader 这个数据集一共有多少条数据
        return len(self.x)

    def __getitem__(self, idx):
        # 定义按索引获取单条样本的方式
        # 返回格式为: (x, y, t, p_true, c_true)
        return (
            self.x[idx],
            self.y[idx],
            self.t[idx],
            self.p[idx],
            self.c[idx],
        )