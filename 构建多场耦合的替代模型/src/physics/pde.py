import torch
import math

# 物理常数
T = 1574.0
Tma, Tmb = 1728.0, 1358.0
La, Lb = 2350.0, 1728.0
Ena, Enb = 3.7e-5, 2.9e-5
h = 4.82e-6
betaa, betab = 0.33, 0.39 # 动力学系数

# 计算相场模型所需的中间物理量 (基于经典的 WBM 模型)
Wa = 3 * Ena / (math.sqrt(2.0) * Tma * h) # 组分 A 的势垒高度
Wb = 3 * Enb / (math.sqrt(2.0) * Tmb * h) # 组分 B 的势垒高度
ea = math.sqrt(6 * math.sqrt(2.0) * Ena * h / Tma) # 梯度能系数
Ma = Tma * Tma * betaa / (6 * math.sqrt(2.0) * h * La) # 组分 A 的迁移率
Mb = Tmb * Tmb * betab / (6 * math.sqrt(2.0) * h * Lb) # 组分 B 的迁移率


def get_gradients(u, inputs):
    """
        计算输出 u 相对于输入 inputs (x, y, t) 的梯度。
        create_graph=True 是必须的，因为我们要计算二阶导数。
        """
    return torch.autograd.grad(
        u, inputs,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]


def compute_pde_residual(inputs, pred, params):
    p = pred[:, 0:1]  # 相场
    c = pred[:, 1:2]  # 浓度场

    # 动态获取反演参数
    gamma = params.gamma
    Ds = params.Ds
    Dl = params.Dl

    # ===========================
    # 1. 计算相场演化残差 (Phase Field)
    # ===========================
    # 1. 计算一阶和二阶导数
    grad_p = get_gradients(p, inputs)
    p_x, p_y, p_t = grad_p[:, 0:1], grad_p[:, 1:2], grad_p[:, 2:3]

    grad_px = get_gradients(p_x, inputs)
    p_xx, p_xy = grad_px[:, 0:1], grad_px[:, 1:2]

    grad_py = get_gradients(p_y, inputs)
    p_yy = grad_py[:, 1:2]

    # 2. 计算热力学驱动力 (Chemical Potential)
    Gpp = 4 * p ** 3 - 6 * p ** 2 + 2 * p
    hpp = 30 * p ** 2 - 60 * p ** 3 + 30 * p ** 4

    Ha = Wa * Gpp + hpp * La * (1 / T - 1 / Tma)
    Hb = Wb * Gpp + hpp * Lb * (1 / T - 1 / Tmb)

    # 3. 各向异性计算 (枝晶生长的核心)
    ang = torch.atan2(p_y, p_x)
    eps = 1e-12
    den = p_x ** 2 + p_y ** 2 + eps

    angx = (p_x * p_xy - p_y * p_xx) / den
    angy = (p_x * p_yy - p_y * p_xy) / den

    # 计算 4 次对称和 8 次对称项 (由反演参数 gamma 决定)
    cos4 = torch.cos(4 * ang)
    cos8 = torch.cos(8 * ang)
    sin4 = torch.sin(4 * ang)
    sin8 = torch.sin(8 * ang)
    # Pa 代表界面张力的各向异性贡献
    Pa = ea ** 2 * (
            16 * gamma * (cos4 + gamma * cos8) * (angx * p_y - angy * p_x)
            - (angx * p_x + angy * p_y) * (8 * gamma * sin4 + 4 * gamma ** 2 * sin8)
            + (p_xx + p_yy) * (1 + 2 * gamma * cos4 + gamma ** 2 * cos4 ** 2)
    )
    # 计算混合迁移率 M 并求出相场残差
    M = (1 - c) * Ma + c * Mb
    residual_p = p_t - M * (Pa - (1 - c) * Ha - c * Hb)

    # ===========================
    # 2. 计算浓度场演化残差 (Concentration Field)
    # ===========================
    # 1. 计算浓度梯度
    grad_c = get_gradients(c, inputs)
    c_x, c_y, c_t = grad_c[:, 0:1], grad_c[:, 1:2], grad_c[:, 2:3]

    # 2. 界面动态扩散系数插值
    h_p = p ** 2 * (3 - 2 * p)# 定义固液混合区的权重
    # 动态等效扩散系数
    D_eff = h_p * Dl + (1 - h_p) * Ds# 根据相场 p 动态决定扩散系数 (固相 Ds vs 液相 Dl)

    #3. 计算扩散通量 J = D * grad(c)
    flux_x = D_eff * c_x
    flux_y = D_eff * c_y

    # 4. 计算通量的散度 (Divergence)
    c_xx = get_gradients(flux_x, inputs)[:, 0:1]
    c_yy = get_gradients(flux_y, inputs)[:, 1:2]

    # 浓度场残差：时间项 - 空间扩散项
    # 残差：dc/dt = div(D * grad c)
    residual_c = c_t - (c_xx + c_yy)

    # ===========================
    # 3. 组合总物理残差
    # ===========================
    loss_pde = torch.mean(residual_p ** 2) + torch.mean(residual_c ** 2)

    return loss_pde