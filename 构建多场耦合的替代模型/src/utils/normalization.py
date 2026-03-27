import numpy as np

def minmax_norm(x):
    """
        最小-最大归一化：将数据线性缩放到 [0, 1] 区间。
        计算公式：(x - min) / (max - min)
        这是 PINN 中最推荐的做法，配合 Tanh 激活函数效果最佳。
        """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standard_norm(x):
    """
        标准差归一化 (Z-Score)：将数据转化为均值为 0，标准差为 1 的分布。
        计算公式：(x - mean) / std
        适用于数据分布接近高斯分布的场景。
        """
    return (x - np.mean(x)) / np.std(x)