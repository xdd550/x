import torch
import math

# =========================
# 物理常数 (严格对齐 C++ IninPhaseAndTemp 等设定)
# =========================
T = 1574.0
Tma, Tmb = 1728.0, 1358.0
La, Lb = 2350.0, 1728.0
Ena, Enb = 3.7e-5, 2.9e-5
h = 4.82e-6
betaa, betab = 0.33, 0.39

# 提前计算复合常数以节省算力
Wa = 3 * Ena / (math.sqrt(2.0) * Tma * h)
Wb = 3 * Enb / (math.sqrt(2.0) * Tmb * h)
ea = math.sqrt(6 * math.sqrt(2.0) * Ena * h / Tma)
Ma = Tma * Tma * betaa / (6 * math.sqrt(2.0) * h * La)
Mb = Tmb * Tmb * betab / (6 * math.sqrt(2.0) * h * Lb)


# =========================
# 自动求导工具函数
# =========================
def get_gradients(u, inputs):
    """
    计算 u 对 inputs (x, y, t) 的偏导数
    返回 shape: [N, 3]
    """
    return torch.autograd.grad(
        u, inputs,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]


# =========================
# PDE残差计算（核心物理逻辑）
# =========================
def compute_pde_residual(inputs, pred, params):
    # 分离预测场: p(相场), c(浓度场)
    p = pred[:, 0:1]
    c = pred[:, 1:2]
    gamma = params.gamma

    # ---------------------------
    # 1. 计算一阶导数
    # ---------------------------
    grad_p = get_gradients(p, inputs)
    p_x = grad_p[:, 0:1]
    p_y = grad_p[:, 1:2]
    p_t = grad_p[:, 2:3]

    # ---------------------------
    # 2. 计算二阶和交叉导数
    # ---------------------------
    grad_px = get_gradients(p_x, inputs)
    p_xx = grad_px[:, 0:1]
    p_xy = grad_px[:, 1:2]

    grad_py = get_gradients(p_y, inputs)
    p_yy = grad_py[:, 1:2]

    # ---------------------------
    # 3. 热力学项 (Gpp, hpp, Ha, Hb)
    # ---------------------------
    Gpp = 4 * p ** 3 - 6 * p ** 2 + 2 * p
    hpp = 30 * p ** 2 - 60 * p ** 3 + 30 * p ** 4

    Ha = Wa * Gpp + hpp * La * (1 / T - 1 / Tma)
    Hb = Wb * Gpp + hpp * Lb * (1 / T - 1 / Tmb)

    # ---------------------------
    # 4. 界面各向异性项 (核心修正)
    # ---------------------------
    ang = torch.atan2(p_y, p_x)

    # 防止分母为 0 导致 NaN
    eps = 1e-12
    den = p_x ** 2 + p_y ** 2 + eps

    # 角度的空间导数 (对应 C++ 中的 angx, angy)
    angx = (p_x * p_xy - p_y * p_xx) / den
    angy = (p_x * p_yy - p_y * p_xy) / den

    cos4 = torch.cos(4 * ang)
    cos8 = torch.cos(8 * ang)
    sin4 = torch.sin(4 * ang)
    sin8 = torch.sin(8 * ang)

    # 严格对齐 C++ 中的强各向异性表达 Pa
    Pa = ea ** 2 * (
            16 * gamma * (cos4 + gamma * cos8) * (angx * p_y - angy * p_x)
            - (angx * p_x + angy * p_y) * (8 * gamma * sin4 + 4 * gamma ** 2 * sin8)
            + (p_xx + p_yy) * (1 + 2 * gamma * cos4 + gamma ** 2 * cos4 ** 2)
    )

    # ---------------------------
    # 5. 组装演化方程 (耦合浓度场 c)
    # ---------------------------
    # 迁移率插值
    M = (1 - c) * Ma + c * Mb

    # 构建残差: p_t = M * (Pa - 驱动力)  (忽略了 C++ 中的随机噪声 Pb)
    residual_p = p_t - M * (Pa - (1 - c) * Ha - c * Hb)

    # PDE 损失，取均方误差
    loss_pde = torch.mean(residual_p ** 2)

    return loss_pde