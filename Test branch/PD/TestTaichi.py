
from collections import defaultdict
import numpy as np
# from numba import jit
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, debug=True)

"""
@ti.func
def quatfromtwovectors(a, b):
    # a -> b的旋转四元数
    v1 = a.normalized()
    v2 = b.normalized()
    cos_theta = v1.dot(v2)

    quat = ti.Vector.zero(ti.f64, 4)
    if cos_theta < -1 + 1e-6:
        pass
        # cos_theta = max(cos_theta, -1)
        # m = ti.Matrix.rows([v1, v2])
        # u, s, v = ti.svd(m, ti.f64)             # 奇异值分解得到垂直的特征向量v3
        # axis = v[:, 2]
        # w2 = (1 + cos_theta) * 0.5              # w2=cos^2(theta/2)
        # w = np.sqrt(w2)
        # vec = axis.normalized() * np.sqrt(1 - w2)
        # quat[0] = w
        # quat[1:] = vec
    else:
        axis = v1.cross(v2)                     # 旋转轴*sin(theta)
        s = ti.sqrt((1 + cos_theta) * 2)        # s=2*cos(theta/2)
        invs = 1 / s
        vec = axis.normalized() * invs
        w = s * 0.5
        quat[0] = w
        quat[1:] = vec
    
    return quat


@ti.kernel
def my_kernel():
    print(quatfromtwovectors(ti.Vector([0, 0, 1]), ti.Vector([0.044096801683,0.000000000000,-0.344229219302])))


def quatfromtwovectors_np(a, b):
    # a -> b的旋转四元数
    v1 = a / np.linalg.norm(a)
    v2 = b / np.linalg.norm(b)
    cos_theta = v1.dot(v2)

    quat = np.zeros(4)
    if cos_theta < -1 + 1e-6:
        print('cos_theta:', cos_theta)
        cos_theta = max(cos_theta, -1)
        m = np.vstack((v1, v2))
        u, s, vh = np.linalg.svd(m, ti.f64)             # 奇异值分解得到垂直的特征向量v3
        axis_tmp = vh[2, :]
        w2 = (1 + cos_theta) * 0.5              # w2=cos^2(theta/2)
        w = np.sqrt(w2)
        vec = axis_tmp * np.sqrt(1 - w2)
        quat[0] = w
        quat[1:] = vec
    else:
        axis_tmp = np.cross(v1, v2)             # 旋转轴*sin(theta)
        s = np.sqrt((1 + cos_theta) * 2)        # s=2*cos(theta/2)
        invs = 1 / s
        vec = axis_tmp * invs
        w = s * 0.5
        quat[0] = w
        quat[1:] = vec
    
    return quat

my_kernel()
print(quatfromtwovectors_np(np.array([0, 0, 1]), np.array([0.044096801683,0.000000000000,-0.344229219302])))
"""

@ti.kernel
def test():
    F = ti.Matrix([[1, 2], [3, 4], [2, 6]])
    FTF = F.transpose() @ F
    FFT = F @ F.transpose()
    print("FTF: ", FTF)
    print("FFT: ", FFT)

    sig, V = ti.sym_eig(FTF)
    _, U = ti.sym_eig(FFT)
    print("Singular values:", ti.sqrt(sig))
    print(V)
    print(ti.sqrt(_))
    print(U)
    U_sort = ti.Matrix.zero(ti.f64, 3, 3)
    U_sort[:, 0] = U[:, 2]
    U_sort[:, 2] = U[:, 0]

    print(U_sort @ ti.Matrix([[sig[0], 0], [0, sig[1]], [0, 0]]) @ V)

test()
print("===")
F_np = np.array([[1, 2], [3, 4], [2, 6]])
FTF_np = F_np.T @ F_np
FFT_np = F_np @ F_np.T

s2, v = np.linalg.eig(FTF_np)
print("Singular values:", np.sqrt(s2))
print(v)

u, s, v = np.linalg.svd(F_np)
print(u, s, v)
print(u @ np.array([[s[0], 0], [0, s[1]], [0, 0]]) @ v)


exit()



# 示例数据
V = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 0.8, 0],
    [1, 1, 0],
    [2, 1, 0],
    [1.9, 0, 0]
])  # 节点位置
T = np.array([
    [0, 1, 2],
    [1, 3, 2],
    [1, 3, 4],
    [1, 4, 5]
])  # 三角形单元


def cotangent(u, v):
    """计算两个向量之间的余切"""
    dot = np.dot(u, v)
    cross = np.linalg.norm(np.cross(u, v))
    return dot / cross if cross != 0 else 0


def compute_cotangent_weights_per_node(V, T):
    """
    计算每个节点的 one-ring 边的余切权重向量
    
    参数:
    V: ndarray of shape (N, 3), 每个节点的位置
    T: ndarray of shape (K, 3), 每个三角形单元的节点索引
    
    返回:
    C: list of lists, 每个节点的 one-ring 边的余切权重
    """
    N = V.shape[0]
    C = [[] for _ in range(N)]  # 每个节点的余切权重向量

    # 边到三角形的映射
    edge_to_triangles = defaultdict(list)

    for t_idx, tri in enumerate(T):
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0]))
        ]
        for edge in edges:
            edge_to_triangles[edge].append(t_idx)

    # 遍历每条边，计算余切权重
    edge_weights = {}
    for edge, triangles in edge_to_triangles.items():
        if len(triangles) == 2:  # 边需要属于两个三角形
            tri1, tri2 = triangles
            idx1, idx2 = T[tri1], T[tri2]
            
            # 找到不属于该边的顶点
            p0, p1 = edge
            p2_1 = list(set(idx1) - set(edge))[0]
            p2_2 = list(set(idx2) - set(edge))[0]

            # 顶点坐标
            v0, v1 = V[p0], V[p1]
            v2_1, v2_2 = V[p2_1], V[p2_2]

            # 向量计算
            cot_alpha = cotangent(v2_1 - v0, v2_1 - v1)
            cot_beta = cotangent(v2_2 - v0, v2_2 - v1)

            # 边的余切权重
            edge_weights[edge] = cot_alpha + cot_beta

    return edge_weights


edge_weights = compute_cotangent_weights_per_node(V, T)


# 将字典转换为双向图
graph = defaultdict(dict)
for (node1, node2), weight in edge_weights.items():
    graph[node1][node2] = weight
    graph[node2][node1] = weight  # 确保双向边

# 初始化用于存储结果的变量
node_neighbors = {}
node_weights = {}

# 填充每个节点的结果
for node, neighbors in graph.items():
    node_neighbors[node] = list(neighbors.keys())  # 一环邻居节点
    node_weights[node] = list(neighbors.values())  # 对应的权重

# 打印结果
print("Node Neighbors and Weights:")
for node in node_neighbors:
    print(f"Node {node}:")
    print(f"  Neighbors: {node_neighbors[node]}")
    print(f"  Weights: {node_weights[node]}")

