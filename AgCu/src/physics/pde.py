import torch

# =========================
# 自动求导工具函数
# =========================
def gradients(u, x):
    return torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0]


def laplacian(u, x):
    grad_u = gradients(u, x)  # [N,3]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    u_xx = gradients(u_x, x)[:, 0:1]
    u_yy = gradients(u_y, x)[:, 1:2]

    return u_xx + u_yy


# =========================
# 自由能项（对应C代码 Gpp）
# =========================
def Gpp(p):
    return 4*p**3 - 6*p**2 + 2*p


# =========================
# 各向异性（对应 ang）
# =========================
def compute_angle(px, py):
    return torch.atan2(py, px)


# =========================
# PDE残差（核心）
# =========================
def compute_pde_residual(inputs, pred, params):
    gamma = params.gamma
    x = inputs[:, 0:1]
    y = inputs[:, 1:2]
    t = inputs[:, 2:3]

    p = pred[:, 0:1]

    # =========================
    # 一阶导数
    # =========================
    grads = gradients(p, inputs)

    p_x = grads[:, 0:1]
    p_y = grads[:, 1:2]
    p_t = grads[:, 2:3]

    # =========================
    # 二阶导数（Laplacian）
    # =========================
    lap_p = laplacian(p, inputs)

    # =========================
    # 角度（各向异性）
    # =========================
    theta = compute_angle(p_x, p_y)

    # =========================
    # 各向异性系数（简化版）
    # 对应：cos(4θ)
    # =========================
    anisotropy = 1.0 + gamma * torch.cos(4 * theta)

    # =========================
    # PDE右侧（简化版 Phase-field）
    # =========================
    F = anisotropy * lap_p - Gpp(p)

    # =========================
    # 残差
    # =========================
    residual = p_t - F

    return torch.mean(residual ** 2)