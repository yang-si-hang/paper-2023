"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
cotangent weights: https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
created at 2024-12-08 by hsy
"""

from typing import List, Dict, DefaultDict
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
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


def compute_edge_to_triangles(faces:npt.NDArray[np.int32])->DefaultDict[tuple, List[int]]:
    """计算边对应的三角形"""
    edge_to_triangles = defaultdict(list)

    for t_idx, tri in enumerate(faces):
        edges = [
            (min(tri[0], tri[1]), max(tri[0], tri[1])),
            (min(tri[1], tri[2]), max(tri[1], tri[2])),
            (min(tri[2], tri[0]), max(tri[2], tri[0]))
        ]
        for edge in edges:
            edge_to_triangles[edge].append(t_idx)       # {(1, 2) = [0, 1]}
    
    return edge_to_triangles


def compute_cotangent_weights_per_node(nodes:npt.NDArray, faces:npt.NDArray[np.int32], edge_to_triangles)->npt.NDArray:
    """
    计算每个节点的 one-ring 边的余切权重向量
    参数:
    nodes: ndarray of shape (N, 3), 每个节点的位置
    faces: ndarray of shape (F, 3), 每个三角形单元的节点索引
    返回:
    """
    # 遍历每条边，计算余切权重
    edge_weights = {}
    graph = defaultdict(dict)
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

            graph[p0][p1] = cot_alpha
            graph[p1][p0] = cot_alpha

    node_neighbors = {}
    node_weights = {}

    # 填充每个节点的结果
    for node, neighbors in graph.items():
        node_neighbors[node] = list(neighbors.keys())       # 一环邻居节点, {1: [2, 3]}
        node_weights[node] = list(neighbors.values())       # 对应的权重, {1: [0.5, 0.5]}

    return node_neighbors, node_weights


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

        self.edge_to_triangles = compute_edge_to_triangles(ele_np)
        self.node_neighbors, self.node_cot_w_init = compute_cotangent_weights_per_node(node_np, ele_np, self.edge_to_triangles)

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
        self.positional_weight = ???

        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)

        self.fix_particle_list = []

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
    def construct_Xg_inv(self):
        for i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[i]
            a, b, c = self.node_pos_init[ia][:2], self.node_pos_init[ib][:2], self.node_pos_init[ic][:2]        # 转换为2D
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.Xg_inv[i] = B_i_inv.inverse()


    @ti.kernel
    def lhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            self.lhs[q_i, q_i] += self.node_mass[q_i] / self.dt**2

    
    @ti.kernel
    def lhs_shear(self):
        # https://medium.com/@victorlouisdg/jax-cloth-tutorial-part-1-e7a0e285864f
        for f_i in range(self.ELEMENT_N):
            Xg_inv = self.Xg_inv[f_i]
            a, b, c, d = Xg_inv[0, 0], Xg_inv[0, 1], Xg_inv[1, 0], Xg_inv[1, 1]

            # F's dim=6*9，flatten(F)按照列优先
            self.F_A[f_i][0, 0] = -a - c
            self.F_A[f_i][0, 1] = a
            self.F_A[f_i][0, 2] = c
            self.F_A[f_i][1, 0] = -b - d
            self.F_A[f_i][1, 1] = b
            self.F_A[f_i][1, 2] = d

        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            q_idx_vec = ti.Vector([idx1, idx2, idx3])
            F_A = self.F_A[f_i]
            ATA = F_A.transpose() @ F_A

            shear_weight = self.stretch_weight[f_i]
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += shear_weight * ATA[A_row_idx, A_col_idx]


    def lhs_bend(self):
        """bending constraints"""
        for q_i in self.node_neighbors:
            neighbors = self.node_neighbors[q_i]
            weights = self.node_cot_w_init[q_i]

            weights_np = np.array(weights)
            weights_sum = np.sum(weights_np)

            weights_np_new = np.append(weights_np, -weights_sum)
            neighbors_new = neighbors + [q_i]
            lhs_tmp = np.outer(weights_np_new, weights_np_new)

            for m in neighbors_new:
                for n in neighbors_new:
                    self.lhs[m, n] += self.bend_weight * lhs_tmp[m, n]
            
    
    @ti.kernel
    def lhs_positional(self):
        """Poistional contraints"""
        for q_i in ti.static(self.fix_particle_list):
            weight = self.positional_weight
            self.lhs[q_i, q_i] += weight


    def precomputation(self):
        self.lhs_mass()
        self.lhs_shear()
        self.lhs_bend()
        self.lhs_positional()


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
    def rhs_shear(self):
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos_new[idx1], self.node_pos_new[idx2], self.node_pos_new[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = F_i

            sig, V = ti.sym_eig(F_i.transpose() @ F_i, ti.f64)
            _, U = ti.sym_eig(F_i @ F_i.transpose(), ti.f64)
            self.Bp[i] = U @ V.transpose()


    def local_solve(self):
        self.rhs.fill(0.)




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
    
    soft = Soft([0.1, 0.1], 1.e5, 0.4, 0.01, 6.e2)
    soft.precomputation()

    lhs_np = np.kron(soft.lhs.to_numpy(), np.eye(3))
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)


if __name__ == "__main__":
    main()
