"""
验证论文中弯曲能量是否保证在二面角为pi的情况下为零
paper: https://dl.acm.org/doi/abs/10.5555/1281957.1281987
"""
import numpy as np

def cot(v1, v2):
    """
    计算两个向量夹角的余切值
    v1, v2: 两个numpy数组，代表三维向量
    """
    # 计算点积
    dot_product = np.dot(v1, v2)
    # 计算叉积的模，它等于 |v1||v2|sin(theta)
    cross_product_norm = np.linalg.norm(np.cross(v1, v2))
    
    # 避免除以零的情况（当向量共线时）
    if np.isclose(cross_product_norm, 0):
        # 如果向量共线且同向，角度为0，cot为无穷大
        if dot_product > 0:
            return np.inf
        # 如果向量共线且反向，角度为180度，cot为负无穷大
        else:
            return -np.inf
            
    # cot(theta) = cos(theta) / sin(theta) = (v1.v2) / |v1 x v2|
    return dot_product / cross_product_norm

def triangle_area(p1, p2, p3):
    """
    使用海伦公式或叉积计算三角形面积
    p1, p2, p3: 三角形的三个顶点
    """
    # 使用叉积的一半来计算面积，更直接
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))

def calculate_bending_energy(x0, x1, x2, x3):
    """
    根据论文中的公式计算给定四个顶点的弯曲能量
    x0, x1, x2, x3: 四个顶点的三维坐标 (numpy array)
    """
    print(f"--- 正在处理顶点 ---")
    print(f"x0: {x0}, x1: {x1}, x2: {x2}, x3: {x3}")

    # 1. 定义三角形
    # t0 由 (x0, x1, x2) 构成
    # t1 由 (x0, x1, x3) 构成
    # 公共边是 x0-x1
    
    # 2. 计算面积
    A0 = triangle_area(x0, x1, x2)
    A1 = triangle_area(x0, x1, x3)
    
    if np.isclose(A0, 0) or np.isclose(A1, 0):
        print("警告: 三角形面积为零，无法计算能量。")
        return 0

    print(f"面积 A0={A0:.4f}, A1={A1:.4f}")

    # 3. 计算所需角度的余切值
    # 这是对论文公式最标准的物理解释
    # beta0 : t0中，x1处的角 (边 x1-x0 和 x1-x2 的夹角)
    # gamma0: t0中，x0处的角 (边 x0-x1 和 x0-x2 的夹角)
    # beta1 : t1中，x1处的角 (边 x1-x0 和 x1-x3 的夹角)
    # gamma1: t1中，x0处的角 (边 x0-x1 和 x0-x3 的夹角)
    
    cot_beta0  = cot(x0 - x1, x2 - x1)
    cot_gamma0 = cot(x1 - x0, x2 - x0)
    cot_beta1  = cot(x0 - x1, x3 - x1)
    cot_gamma1 = cot(x1 - x0, x3 - x0)

    print("计算出的余切值:")
    print(f"cot(beta0)={cot_beta0:.4f}, cot(gamma0)={cot_gamma0:.4f}")
    print(f"cot(beta1)={cot_beta1:.4f}, cot(gamma1)={cot_gamma1:.4f}")
    
    # 4. 组装 K0 算子作用于 X 后的向量 v
    # v = k0*x0 + k1*x1 + k2*x2 + k3*x3
    # 根据论文公式 K0 = (c03+c04, c01+c02, -c01-c03, -c02-c04)
    # 我们将 cjk 解释为在相应顶点的角度的余切
    # c03 -> cot_beta0, c04 -> cot_beta1
    # c01 -> cot_gamma0, c02 -> cot_gamma1
    
    k0 = cot_beta0 + cot_beta1
    k1 = cot_gamma0 + cot_gamma1
    k2 = -cot_gamma0 - cot_beta0
    k3 = -cot_gamma1 - cot_beta1

    # 核心计算：应用算子
    v = k0*x0 + k1*x1 + k2*x2 + k3*x3
    
    print(f"计算出的核心向量 v = {v}")
    
    # 5. 计算最终的能量
    # E_b = (3 / (2 * (A0 + A1))) * ||v||^2
    
    norm_v_sq = np.linalg.norm(v)**2
    energy_factor = 3 / (2 * (A0 + A1))
    
    energy = energy_factor * norm_v_sq
    
    return energy

# --- 主验证程序 ---

# === 情况1: 四点共面 (一个正方形) ===
print("===================================")
print("情况1: 四点共面 (平面)")
print("===================================")
x0_p = np.array([0.0, 0.0, 0.0])
x1_p = np.array([1.0, 0.0, 0.0])
x2_p = np.array([1.0, 1.0, 0.0])
x3_p = np.array([0.0, 1.0, 0.0])

# 注意：为了匹配论文图示的“蝴蝶”形状，我们将x2和x3的位置对调
# t0 = (x0, x1, x2), t1 = (x0, x1, x3)
# 在正方形中，x2和x3在公共边的异侧
x0_planar = np.array([0.0, 1.0, 0.0])
x1_planar = np.array([1.0, 1.0, 0.0])
x2_planar = np.array([0.9, 2., 0.0]) # t0 的第三点
x3_planar = np.array([0.1, -0.2, 0.0]) # t1 的第三点

energy_planar = calculate_bending_energy(x0_planar, x1_planar, x2_planar, x3_planar)
print(f"\n>>> 平面情况下的弯曲能量: {energy_planar:.10e}\n")


# === 情况2: 四点不共面 (轻微折叠) ===
print("===================================")
print("情况2: 四点不共面 (非平面)")
print("===================================")
x0_non_planar = np.array([0.0, 1.0, 0.0])
x1_non_planar = np.array([1.0, 1.0, 0.0])
x2_non_planar = np.array([1.0, 2.0, 0.0])
# 将 x3 沿 z 轴轻微抬起
x3_non_planar = np.array([0.0, 0.0, 0.1]) 

energy_non_planar = calculate_bending_energy(x0_non_planar, x1_non_planar, x2_non_planar, x3_non_planar)
print(f"\n>>> 非平面情况下的弯曲能量: {energy_non_planar:.10e}\n")

# === 验证结论 ===
print("===================================")
print("结论")
print("===================================")
if np.isclose(energy_planar, 0):
    print("✅ 验证成功: 在平面情况下，计算出的能量接近于零。")
else:
    print("❌ 验证失败: 在平面情况下，能量不为零。")

if energy_non_planar > 1e-8:
    print("✅ 验证成功: 在非平面情况下，计算出的能量是一个明显的正数。")
else:
    print("❌ 验证失败: 在非平面情况下，能量为零。")

