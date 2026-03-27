"""

dataset.py (静态雷达)：它始终盯着实验数据里的界面（Truth Interface），保证网络不跑偏。

sampler.py (动态巡逻)：它盯着网络自己预测出来的界面（Predicted Interface）。
如果网络觉得这里有界面，它就在这里加强物理审查（PDE Loss），强迫这个预测界面必须符合扩散方程。
"""

import torch


def sample_collocation(num_points, device, model=None, cfg=None, current_epoch=0):
    """
    智能自适应采样器：前期全局巡逻，后期界面精准打击
    """
    # 获取配置中的阈值，例如设为 50
    start_focus_epoch = cfg.sampling.get('start_focus_epoch', 50)

    # 判断是否开启界面追踪：必须满足 (1)配置文件开启 (2)模型已传入 (3)达到指定Epoch
    should_focus = (cfg.sampling.focus_interface and
                    model is not None and
                    current_epoch >= start_focus_epoch)

    if not should_focus:
        # --- 策略 A: 全局均匀采样 (前期探索) ---
        x = torch.rand(num_points, 1, device=device)
        y = torch.rand(num_points, 1, device=device)
        t = torch.rand(num_points, 1, device=device)
        return torch.cat([x, y, t], dim=1)

    # --- 策略 B: 预测导向的界面追踪 (后期加固) ---
    # 先生成 3 倍的候选点，从中筛选最像界面的位置
    num_candidates = num_points * 3
    candidates = torch.rand(num_candidates, 3, device=device)

    with torch.no_grad():
        # 让网络对候选点进行一次快速“预判”
        pred = model(candidates)
        p_pred = pred[:, 0:1]  # 提取预测相场 p

    # 计算权重：离 p=0.5 (界面中心) 越近的点，权重呈指数级增长
    # 这里的 20.0 是陡峭系数，越大越集中在界面边缘
    weights = torch.exp(-20.0 * (p_pred - 0.5) ** 2).squeeze()

    # 按权重比例进行多项式采样，选出最终的配点
    idx = torch.multinomial(weights, num_points, replacement=False)

    return candidates[idx]