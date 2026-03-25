pinn-phasefield/
│
├── configs/                    # YAML配置（核心）
│   ├── default.yaml
│   ├── data.yaml
│   ├── model.yaml
│   ├── train.yaml
│   └── inverse.yaml
│
├── data/
│   ├── raw/                   # 原始.plt
│   ├── processed/             # npz数据
│   └── dataset.py             # 数据集类（PyTorch）
│
├── src/
│   ├── models/
│   │   ├── pinn.py            # PINN网络
│   │   └── components.py      # MLP等模块
│   │
│   ├── physics/
│   │   ├── pde.py             # PDE残差（核心）
│   │   └── parameters.py      # 可学习参数（γ等）
│   │
│   ├── losses/
│   │   └── loss_fn.py         # loss组合
│   │
│   ├── utils/
│   │   ├── convert.py         # plt → npz
│   │   ├── sampler.py         # 随机采样
│   │   └── normalization.py   # 归一化
│   │
│   ├── lightning/
│   │   ├── module.py          # LightningModule（核心）
│   │   └── datamodule.py      # LightningDataModule
│   │
│   └── train.py               # 训练入口
│
├── experiments/               # 输出（日志、模型）
│
├── requirements.txt
└── README.md


| C代码        | PINN          |
| ---------- | ------------- |
| difpx      | p_x           |
| difpxx     | p_xx          |
| ang        | atan2         |
| cos(4*ang) | torch.cos(4θ) |
| newp       | 不存在（直接拟合）     |
| 时间循环       | p_t           |




