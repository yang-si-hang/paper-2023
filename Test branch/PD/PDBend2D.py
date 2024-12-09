"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
created at 2024-12-08 by hsy
"""

from typing import List, Dict
import numpy as np
import numpy.typing as npt
from collections import defaultdict
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

from Utilize.GenMsh import mesh_obj_tri
from GGUI import gui_set


def read_msh(file_path):
    # 考虑从外部导入
    pass


def cotangent(u:npt.NDArray, v:npt.NDArray)->float:
    """计算两个向量之间的余切"""
    dot = np.dot(u, v)
    cross = np.linalg.norm(np.cross(u, v))
    return dot / cross if cross != 0 else 0


def compute_cotangent_weights_per_node(nodes:npt.NDArray, faces:npt.NDArray[np.int32])->npt.NDArray:
    """
    计算每个节点的 one-ring 边的余切权重向量

    参数:
    nodes: ndarray of shape (N, 3), 每个节点的位置
    faces: ndarray of shape (K, 3), 每个三角形单元的节点索引
    
    返回:
    cotangent_weights_vec: list of lists, 每个节点的 one-ring 边的余切权重
    """
    node_num = nodes.shape[0]
    cotangent_weights_vec = [[] for _ in range(node_num)]  # 每个节点的余切权重向量

    # 边到三角形的映射
    edge_to_triangles = defaultdict(list)

    for t_idx, tri in enumerate(faces):
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0]))
        ]
        for edge in edges:
            edge_to_triangles[edge].append(t_idx)

    # 遍历每条边，计算余切权重
    edge_weights = {}
    for edge, triangles in edge_to_triangles.items():       # edge: (p0, p1), triangles: [tri1, tri2]
        if len(triangles) == 2:  # 边需要属于两个三角形
            tri1, tri2 = triangles
            idx1, idx2 = faces[tri1], faces[tri2]
            
            # 找到不属于该边的顶点
            p0, p1 = edge
            p2_1 = list(set(idx1) - set(edge))[0]
            p2_2 = list(set(idx2) - set(edge))[0]

            # 顶点坐标
            v0, v1 = nodes[p0], nodes[p1]
            v2_1, v2_2 = nodes[p2_1], nodes[p2_2]

            # 向量计算
            cot_alpha = cotangent(v2_1 - v0, v2_1 - v1)
            cot_beta = cotangent(v2_2 - v0, v2_2 - v1)

            # 边的余切权重
            edge_weights[edge] = cot_alpha + cot_beta

    edges_np = np.array(list(edge_weights.keys()))  # 提取键，形如 [(1, 2)]
    weights_np = np.array(list(edge_weights.values()))  # 提取值，形如 [0.0]

    edges_cot_weights = np.hstack((edges_np, weights_np[:, np.newaxis]))  # 形如 [[1, 2, 0.0]]

    return edges_cot_weights


def compute_onering_edges(node_num:float, elements:npt.NDArray)->List:
    one_ring_edges = [set() for _ in range(node_num)]

    for e_i in elements:
        idx1, idx2, idx3 = e_i
        one_ring_edges[idx1].update([idx2, idx3])
        one_ring_edges[idx2].update([idx1, idx3])
        one_ring_edges[idx3].update([idx1, idx2])

    return one_ring_edges


@ti.data_oriented
class SoftBend2D:
    def __init__(self, shape:list, E:float, nu:float, dt:float, density:float, g=9.8):
        self.shape = shape
        # 是否传入的是 .msh 文件(已划分网格)
        if isinstance(self.shape, str):
            node_np, edge_np, ele_np = read_msh(self.shape)
        else:
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.01)
            node_np = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))         # di: N*3
        self.E, self.nu, self.dt, self.density, self.g = E, nu, dt, density, g
        self.dim = 3
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.edges_cot_weights = compute_cotangent_weights_per_node(node_np, ele_np)

        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.node_pos = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_vel = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_voronoi = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_onering = 
        self.node_pos_init.from_numpy(node_np.astype(np.float64))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge.from_numpy(edge_np.astype(np.int32))

        self.ele = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)
        self.ele_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele.from_numpy(ele_np.astype(np.int32))

        self.bend_weight = ???
        self.stretch_weight = ???
        self.volume_weight = ???

        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.PARTICLE_N*3))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)

        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")



    @ti.kernel
    def construct_mass(self):
        for e_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[e_i]
            ele_volume_tmp = 0.5 * \
            ti.abs((self.node_pos[idx2] - self.node_pos[idx1]).cross(self.node_pos[idx3] - self.node_pos[idx1]))
            self.node_voronoi[idx1] += ele_volume_tmp / 3
            self.node_voronoi[idx2] += ele_volume_tmp / 3
            self.node_voronoi[idx3] += ele_volume_tmp / 3

            self.ele_volume[e_i] = ele_volume_tmp

        for q_i in range(self.PARTICLE_N):
            self.node_mass[q_i] = self.density * self.node_voronoi[q_i]


    @ti.kernel
    def precomputation(self):
        dim = self.dim

        for q_i in range(self.PARTICLE_N):
            tmp = self.node_mass[q_i] / self.dt**2
            self.lhs[q_i*dim, q_i*dim] += tmp
            self.lhs[q_i*dim+1, q_i*dim+1] += tmp
            self.lhs[q_i*dim+2, q_i*dim+2] += tmp

        for i in range(self.EDGE_N):
            node, neighbor, cot_w = self.edges_cot_weights[i]
            node_idx, neighbor_idx = int(node), int(neighbor)
            
        



    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for q_i in range(self.PARTICLE_N):
            idx1, idx2, idx3 = dim*q_i, dim*q_i+1, dim*q_i+2
            pos = self.node_pos[q_i]
            vel = self.node_vel[q_i]
            self.sn[idx1] = pos[0] + dt * vel[0]
            self.sn[idx2] = pos[1] + dt * vel[1]
            self.sn[idx3] = pos[2] + dt * vel[2]


    @ti.kernel
    def warm_start(self):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = self.node_pos[q_i].x
            self.node_pos_new[q_i].y = self.node_pos[q_i].y
            self.node_pos_new[q_i].z = self.node_pos[q_i].z


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        dim = self.dim
        for q_i in range(self.PARTICLE_N):
            idx1, idx2, idx3 = q_i*dim, q_i*dim+1, q_i*dim+2 
            self.node_pos_new[q_i].x = sol[idx1]
            self.node_pos_new[q_i].y = sol[idx2]
            self.node_pos_new[q_i].z = sol[idx3]


    @ti.kernel
    def update_vel_pos(self):
        for i in range(self.PARTICLE_N):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]


def main():
    class Soft(SoftBend2D):
        def __init__(self, shape:list, E:float, nu:float, dt:float, density:float, g=9.8):
            super().__init__(shape, E, nu, dt, density, g)
    
    soft = 


if __name__ == "__main__":
    main()
