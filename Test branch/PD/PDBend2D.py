"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
cotangent weights: https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
created at 2024-12-08 by hsy
"""

import os
import sys
import time
from typing import List, Dict, DefaultDict
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
import taichi as ti
from taichi.lang import impl
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import mesh_obj_tri
from Utilize.GuiTaichi import gui_set
from Utilize.MathTaichi import svd_3x2, svd_3x2_new


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
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.05)
            node_np = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))         # di: N*3
        np.savetxt("node_np.csv", node_np, fmt='%f', delimiter=",")
        np.savetxt("edge_np.csv", edge_np, fmt='%d', delimiter=",")
        np.savetxt("ele_np.csv", ele_np, fmt='%d', delimiter=",")
        # exit(0)
        self.solve_itr:int = 10
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
        # self.node_onering = 
        self.node_pos_init.from_numpy(node_np.astype(np.float64))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge.from_numpy(edge_np.astype(np.int32))

        self.ele = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)
        self.ele_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele.from_numpy(ele_np.astype(np.int32))

        self.bend_weight = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.stretch_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        # self.volume_weight = ???
        self.positional_weight = 1.e4

        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.Bp_shear = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)       # shear part of the first Piola-Kirchhoff stress tensor
        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.pre_fact_lhs_solve = None

        # self.fix_particle_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.fix_particle_list = [0, 1, 2]
        self.FIX_N = len(self.fix_particle_list)
        self.fix_particle_ti = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))

        self.construct_mass()
        self.construct_Xg_inv()
        
        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")



    @ti.kernel
    def construct_mass(self):
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            ele_volume_tmp = 0.5 * \
            ((self.node_pos_init[idx2] - self.node_pos_init[idx1]).cross(self.node_pos[idx3] - self.node_pos[idx1])).norm()
            self.node_voronoi[idx1] += ele_volume_tmp / 3.
            self.node_voronoi[idx2] += ele_volume_tmp / 3.
            self.node_voronoi[idx3] += ele_volume_tmp / 3.

            self.ele_volume[f_i] = ele_volume_tmp
            self.stretch_weight[f_i] = 2 * self.mu * self.ele_volume[f_i]
            # self.stretch_weight[f_i] = 1.

        for q_i in range(self.PARTICLE_N):
            # self.node_mass[q_i] = self.density * self.node_voronoi[q_i]
            self.node_mass[q_i] = 1.


    @ti.kernel
    def construct_Xg_inv(self):
        for i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[i]
            a = ti.Vector([self.node_pos_init[ia].x, self.node_pos_init[ia].y])
            b = ti.Vector([self.node_pos_init[ib].x, self.node_pos_init[ib].y])
            c = ti.Vector([self.node_pos_init[ic].x, self.node_pos_init[ic].y])
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            # print("B_i_inv:\n", B_i_inv)
            self.Xg_inv[i] = B_i_inv.inverse()
            # print("Xg_inv:\n", self.Xg_inv[i])


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
            # print("F_A:\n", F_A)
            ATA = F_A.transpose() @ F_A
            # print("ATA:\n", ATA)

            shear_weight = self.stretch_weight[f_i]
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += shear_weight * ATA[A_row_idx, A_col_idx]


    def lhs_bend(self):
        """bending constraints"""
        for q_i in self.node_neighbors:
            voronoi = self.node_voronoi[q_i]
            neighbors = self.node_neighbors[q_i]
            weights = self.node_cot_w_init[q_i]

            weights_np = np.array(weights)
            weights_sum = np.sum(weights_np)

            weights_np_new = np.append(weights_np, -weights_sum)
            neighbors_new = neighbors + [q_i]
            lhs_tmp = voronoi * np.outer(weights_np_new, weights_np_new)

            for m in neighbors_new:
                for n in neighbors_new:
                    self.lhs[m, n] += lhs_tmp[m, n]
            
    
    @ti.kernel
    def lhs_positional(self):
        """Poistional contraints"""
        for q_i in range(self.FIX_N):
            idx = self.fix_particle_ti[q_i]
            self.lhs[idx, idx] += self.positional_weight


    def precomputation(self):
        self.lhs_mass()
        self.lhs_shear()
        # self.lhs_bend()
        self.lhs_positional()


    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for q_i in range(self.PARTICLE_N):
            idx1, idx2, idx3 = dim*q_i, dim*q_i+1, dim*q_i+2
            self.sn[idx1] = self.node_pos[q_i].x + self.node_vel[q_i].x * dt
            self.sn[idx2] = self.node_pos[q_i].y + self.node_vel[q_i].y * dt
            self.sn[idx3] = self.node_pos[q_i].z + self.node_vel[q_i].z * dt


    @ti.kernel
    def warm_start(self):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = self.node_pos[q_i].x
            self.node_pos_new[q_i].y = self.node_pos[q_i].y
            self.node_pos_new[q_i].z = self.node_pos[q_i].z


    @ti.kernel
    def rhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            idx1, idx2, idx3 = q_i*3, q_i*3+1, q_i*3+2
            self.rhs[idx1] += self.node_mass[q_i] * self.sn[idx1] / self.dt**2
            self.rhs[idx2] += self.node_mass[q_i] * self.sn[idx2] / self.dt**2
            self.rhs[idx3] += self.node_mass[q_i] * self.sn[idx3] / self.dt**2


    @ti.kernel
    def rhs_shear(self):
        ti.loop_config(serialize=True)
        for f_i in range(self.ELEMENT_N):
            print("==============================================================================================")
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos_new[idx1], self.node_pos_new[idx2], self.node_pos_new[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = F_i
            print(f"F_i:{F_i:e}")

            U, sig, V = svd_3x2_new(F_i)
            print(f"U:{U:e}; sig:{sig:e}; V:{V:e}")
            reon_error = (U @ ti.Matrix([[sig[0], 0], [0., sig[1]], [0, 0]], ti.f64) @ V.transpose() - F_i).norm()
            # if (U @ ti.Matrix([[sig[0], 0], [0., sig[1]], [0, 0]], ti.f64) @ V.transpose() - F_i).norm() > 1e-8:
            #     print(f"SVD failed-----------------------------------------------------------------------------------------")
            #     print(f"F_i: {F_i:e}")
            #     print(f"U: {U:e}")
            #     print(f"sig: {sig:e}")
            #     print(f"V: {V:e}")
            self.Bp_shear[f_i] = U @ ti.Matrix([[1., 0], [0., 1], [0, 0]], ti.f64) @ V.transpose()
            # print(f"Bp_shear:{self.Bp_shear[f_i]:e}")

        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            weight = self.stretch_weight[f_i]
            Bp_shear_i = self.Bp_shear[f_i]
            F_AT = self.F_A[f_i].transpose()
            # print("F_AT:\n", F_AT)

            # Bp_shear_i做transpose，因为AT需要与Bp的x，y，z分别矩阵乘法
            F_ATBp = F_AT @ Bp_shear_i.transpose()
            F_ATBp *= weight
            # print(f"BpT:\n{Bp_shear_i.transpose()}")
            # print("F_ATBp:\n", F_ATBp)

            for q_i, dim_idx in ti.ndrange(3, 3):
                q_idx = self.ele[f_i][q_i]
                self.rhs[q_idx*3+dim_idx] += F_ATBp[q_i, dim_idx]

            # idx1_x, idx1_y, idx1_z = idx1*3, idx1*3+1, idx1*3+2
            # idx2_x, idx2_y, idx2_z = idx2*3, idx2*3+1, idx2*3+2
            # idx3_x, idx3_y, idx3_z = idx3*3, idx3*3+1, idx3*3+2

            # self.rhs[idx1_x] += F_ATBp[0, 0]
            # self.rhs[idx1_y] += F_ATBp[0, 1]
            # self.rhs[idx1_z] += F_ATBp[0, 2]
            # self.rhs[idx2_x] += F_ATBp[1, 0]
            # self.rhs[idx2_y] += F_ATBp[1, 1]
            # self.rhs[idx2_z] += F_ATBp[1, 2]
            # self.rhs[idx3_x] += F_ATBp[2, 0]
            # self.rhs[idx3_y] += F_ATBp[2, 1]
            # self.rhs[idx3_z] += F_ATBp[2, 2]


    @ti.kernel
    def rhs_poistional(self):
        for q_i in range(self.FIX_N):
            weight = self.positional_weight
            q_idx = self.fix_particle_ti[q_i]
            q_i_x, q_i_y, q_i_z = q_idx*3, q_idx*3+1, q_idx*3+2
            self.rhs[q_i_x] += weight * self.node_pos_init[q_idx].x
            self.rhs[q_i_y] += weight * self.node_pos_init[q_idx].y
            self.rhs[q_i_z] += weight * self.node_pos_init[q_idx].z


    def local_solve(self):
        self.rhs.fill(0.)
        self.rhs_mass()
        self.rhs_shear()
        self.rhs_poistional()


    @ti.kernel
    def update_pos_new(self, sol_x:ti.types.ndarray(), sol_y:ti.types.ndarray(), sol_z:ti.types.ndarray()):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = sol_x[q_i]
            self.node_pos_new[q_i].y = sol_y[q_i]
            self.node_pos_new[q_i].z = sol_z[q_i]


    @ti.kernel
    def update_vel_pos(self):
        for i in range(self.PARTICLE_N):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]

    
    def preset_gui(self, pos:List[float], target:List[float], up_orient:List[float]):
        self.window, self.camera, self.scene = gui_set(pos, target, up_orient)
        self.canvas = self.window.get_canvas()
        self.show_preset()

    
    def show_preset(self):
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_N)
        self.node_color = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_N)
        self.edge_show = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge_show.from_numpy(self.edge.to_numpy().astype(np.int32))
        
        for q_i in range(self.PARTICLE_N):
            self.node_color[q_i] = ti.Vector([0., 0., 0.])
        for q_i in self.fix_particle_list:
            self.node_color[q_i] = ti.Vector([1., 0., 0.])


    def gui_show(self, SHOW_FLAG:bool=True, WRITE_FLAG:bool=False, itr_num:int=0):
        if SHOW_FLAG is False:
            return
        self.scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        self.scene.ambient_light((.8, .8, .8))

        self.node_show.from_numpy(self.node_pos.to_numpy().astype(np.float32))
        self.scene.particles(self.node_show, radius=0.001, per_vertex_color=self.node_color)
        self.scene.lines(self.node_show, width=1., indices=self.edge_show, color=(0., 0., 0.), vertex_count=0)

        self.canvas.scene(self.scene)
        self.canvas.set_background_color((1., 1., 1.))

        if WRITE_FLAG is True:
            self.window.save_image(f"FigureWrite/{itr_num:05d}.png")
        self.window.show()


    def substep(self, step_num:int):
        self.construct_sn()
        # np.savetxt(f"sn_{step_num:05d}.csv", self.sn.to_numpy(), fmt='%f', delimiter=",")
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            # print(f"Iteration: {itr}")
            self.local_solve()
            rhs_np = self.rhs.to_numpy()
            # print(f"Rhs:\n{rhs_np}")
            # Split rhs_np into x,y,z components
            rhs_np_x = rhs_np[0::3]
            rhs_np_y = rhs_np[1::3]
            rhs_np_z = rhs_np[2::3]

            node_pos_new_np_x = self.pre_fact_lhs_solve(rhs_np_x)
            node_pos_new_np_y = self.pre_fact_lhs_solve(rhs_np_y)
            node_pos_new_np_z = self.pre_fact_lhs_solve(rhs_np_z)

            self.update_pos_new(node_pos_new_np_x, node_pos_new_np_y, node_pos_new_np_z)
            # print(f"Node pos new:\n", self.node_pos_new.to_numpy())
        
        self.update_vel_pos()

    
    @ti.kernel
    def init_vel(self):
        for q_i in range(self.PARTICLE_N):
            if self.node_pos_init[q_i].y > self.shape[0] - 1.e-3:
                self.node_vel[q_i].y = 1.
            else:
                self.node_vel[q_i].y = 0.


def main():
    class Soft(SoftBend2D):
        def __init__(self, shape:list, E:float, nu:float, dt:float, density:float, g=9.8):
            super().__init__(shape, E, nu, dt, density, g)
    
    soft = Soft([0.1, 0.1], 1.e5, 0.4, 0.01, 6.e2)
    soft.preset_gui([0.05, 0.1, 0.3], [0.05, 0.1, 0.], [0., 1., 0.])

    soft.precomputation()
    # print("Inital lhs:\n", soft.lhs.to_numpy())
    lhs_np = soft.lhs.to_numpy()
    # print("Pos init:\n", soft.node_pos_init.to_numpy().flatten())
    # print("Rhs:\n", lhs_np @ soft.node_pos_init.to_numpy().flatten())
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)
    soft.init_vel()
    np.savetxt("mass.csv", soft.node_mass.to_numpy(), fmt='%f', delimiter=",")
    np.savetxt("lhs.csv", lhs_np, fmt='%f', delimiter=",")

    for itr in range(100):
        soft.substep(itr)
        soft.gui_show(True, False, itr)
        time.sleep(0.1)

    # np.savetxt('rhs.csv', soft.rhs.to_numpy(), fmt='%f', delimiter=',')
        

if __name__ == "__main__":
    main()
