import numpy as np
import glob
import os

# =========================
# 物理参数（必须和C代码一致）
# =========================
deltx = 2.41e-6
delty = 2.41e-6
delt  = 1.1e-8

# =========================
# 读取单个 plt 文件
# =========================
def read_plt(file_path):
    data = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 跳过头部（前三行）
    for line in lines:
        if line.startswith("variables") or line.startswith("zone") or line.startswith("title"):
            continue

        parts = line.strip().split()
        if len(parts) == 4:
            i, j, p, c = parts
            data.append([int(i), int(j), float(p), float(c)])

    data = np.array(data)
    return data  # [N, 4]

# =========================
# 主函数：plt → npz
# =========================
def convert_all_plt_to_npz(folder_path, output_file="phase_data.npz"):
    files = sorted(glob.glob(os.path.join(folder_path, "*.plt")),
                   key=lambda x: int(os.path.basename(x).split('.')[0]))

    X, Y, T, P, C = [], [], [], [], []

    for file in files:
        print(f"Processing: {file}")

        # 提取时间（文件名）
        t_step = int(os.path.basename(file).split('.')[0])
        t_real = t_step * delt

        data = read_plt(file)

        i = data[:, 0]
        j = data[:, 1]
        p = data[:, 2]
        c = data[:, 3]

        # 转换为物理坐标
        x = i * deltx
        y = j * delty
        t = np.full_like(x, t_real)

        X.append(x)
        Y.append(y)
        T.append(t)
        P.append(p)
        C.append(c)

    # 拼接所有时间步
    X = np.concatenate(X)[:, None]
    Y = np.concatenate(Y)[:, None]
    T = np.concatenate(T)[:, None]
    P = np.concatenate(P)[:, None]
    C = np.concatenate(C)[:, None]

    print("Final data shape:", X.shape)

    # 保存
    np.savez(output_file,
             x=X.astype(np.float32),
             y=Y.astype(np.float32),
             t=T.astype(np.float32),
             p=P.astype(np.float32),
             c=C.astype(np.float32))

    print(f"Saved to {output_file}")

# =========================
# 使用示例
# =========================
if __name__ == "__main__":
    folder = "../../data/raw"   # 你的plt文件夹路径
    convert_all_plt_to_npz(folder)