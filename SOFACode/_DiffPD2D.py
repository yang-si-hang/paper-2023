""" 用于SOFA仿真环境(2D场景)控制的DiffPD 2D模型
created at 2025-03-02 by hsy
"""
import os
import sys
import time
from typing import List, Dict, DefaultDict, Tuple
from cv2 import line
from pathlib import Path
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
import taichi as ti
# import meshtaichi_patcher as Patcher
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import mesh_obj_tri, write_obj, write_mshv2_tri, read_mshv2_triangle
from Utilize.GuiTaichi import gui_set


def read_msh(file:str):
    nodes, faces = read_mshv2_triangle(file)

    edges = []
    for face in faces:
        edge1 = sorted([face[0], face[1]])
        edge2 = sorted([face[1], face[2]])
        edge3 = sorted([face[2], face[0]])
        
        edges.append(edge1)
        edges.append(edge2)
        edges.append(edge3)
    
    # Remove duplicate edges by converting to tuples, using a set, then back to list
    unique_edges = []
    edge_set = set()
    for edge in edges:
        edge_tuple = tuple(edge)
        if edge_tuple not in edge_set:
            edge_set.add(edge_tuple)
            unique_edges.append(edge)
    
    edges = np.array(unique_edges, dtype=int)

    return nodes, edges, faces


def line_from_points_2d(p1, p2):
    """
    Given two distinct points p1=(x1, y1) and p2=(x2, y2),
    returns the line parameters (a, b, c) for the line:
        a*x + b*y + c = 0
    with sqrt(a^2 + b^2) = 1.
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    
    if np.allclose(p1, p2):
        raise ValueError("Cannot form a line: the two points are identical.")
    
    # a*x + b*y + c = 0 through points p1, p2
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0]*p2[1] - p2[0]*p1[1]
    
    # Normalize so that sqrt(a^2 + b^2) = 1
    norm = np.sqrt(a*a + b*b)
    a /= norm
    b /= norm
    c /= norm
    
    return a, b, c


def compress_vectors(vectors:npt.NDArray, threshold:float)->npt.NDArray:
    """
    Args:
        vectors (np.ndarray): An N x 2 array of vectors.
        threshold (float): The threshold value for the vector norms.

    Returns:
        np.ndarray: An N x 2 array of vectors after applying the compression.
    """
    # Compute the Euclidean norms for each row vector (shape: N x 1).
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Compute the scaling factor for each vector: if norm > threshold, scale to threshold, else use 1.
    scales = np.where(norms > threshold, threshold / norms, 1.0)
    
    # Apply the scaling factors to compress the vectors.
    compressed_vectors = vectors * scales
    
    return compressed_vectors


@ti.data_oriented
class SoftObject2D:
    def __init__(self, shape, fix:List[int], contact:List[int], 
                 E:float, nu:float, dt:float, density:float, **kwargs):
        self.shape = shape
        # 是否传入的是 .msh 文件(已划分网格)
        if isinstance(self.shape, Path):
            node_np, edge_np, ele_np = read_msh(self.shape)
        else:
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.01)
            node_np_3d = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))         # di: N*3
            np.savetxt("node_np.csv", node_np, fmt='%f', delimiter=",")
            np.savetxt("edge_np.csv", edge_np, fmt='%d', delimiter=",")
            np.savetxt("ele_np.csv", ele_np, fmt='%d', delimiter=",")

            msh_file:str = "Mesh/shape.msh"
            write_mshv2_tri(msh_file, node_np, ele_np)

        self.solve_itr:int = 10
        self.E, self.nu, self.dt, self.density = E, nu, dt, density
        self.dim = 2
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
       
        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.node_pos      = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new  = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_vel      = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_voronoi  = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass     = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass_sum = ti.field(dtype=ti.f64, shape=())
        self.node_pos_init.from_numpy(node_np.astype(np.float64))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge.from_numpy(edge_np.astype(np.int32))

        self.ele = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)
        self.ele_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele.from_numpy(ele_np.astype(np.int32))

        self.stretch_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        # self.stretch_lim_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.positional_weight = 0.         # define later

        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.ele_u = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)          # singular value decomposition
        self.ele_v = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.Bp_shear = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)       # stretch part
        self.stretch_stress = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)

        self.sn  = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.rhs_stretch = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.pre_fact_lhs_solve = None

        self.dBp_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*2, self.PARTICLE_N*2))
        self.dA = None
        self.dx_const = ti.field(ti.f64, shape=self.dim*self.PARTICLE_N)          # dx_dy中的常数部分

        self.fix_particle_list = fix
        self.contact_particle_list = contact
        self.FIX_N = len(self.fix_particle_list)
        self.CON_N = len(self.contact_particle_list)
        self.fix_particle_ti     = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))
        self.contact_particle_ti.from_numpy(np.array(self.contact_particle_list).astype(np.int32))
        self.contact_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.CON_N)
        self.contact_vel.fill(0.)

        # 用于Sofa环境可能有问题
        self.marker_idx:int = 0
        self.marker_pos_desired = ti.Vector.field(2, dtype=ti.f64, shape=(1))
        self.dL_dq_contact = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.dy_contact = None          # 用于控制, numpy array
        self.z = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.marker_pos_desired[0] = self.node_pos_init[self.marker_idx] + ti.Vector([0., 0.01], dt=ti.f64)
        self.dL_dq_contact.fill(0.)
        self.z.fill(0.)

        self.construct_mass()
        self.construct_Xg_inv()
        self.positional_weight = 1.e3 * self.node_mass_sum[None] / self.PARTICLE_N / self.dt**2
        self.construct_dx_const()

        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")
        print(f"Positional weight: {self.positional_weight}")


    @ti.kernel
    def construct_mass(self):
        for f_i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[f_i]
            qa, qb, qc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            ele_volume_tmp = 0.5 * ti.abs(((qb - qa).cross(qc - qa)))
            # print(f"Element {f_i}: {ele_volume_tmp}")

            self.node_voronoi[ia] += ele_volume_tmp / 3.
            self.node_voronoi[ib] += ele_volume_tmp / 3.
            self.node_voronoi[ic] += ele_volume_tmp / 3.
            self.ele_volume[f_i] = ele_volume_tmp
            self.stretch_weight[f_i] = 2 * self.mu * self.ele_volume[f_i]

        for q_i in range(self.PARTICLE_N):
            self.node_mass[q_i] = self.density * self.node_voronoi[q_i]
            # self.node_mass[q_i] = 0.
            self.node_mass_sum[None] += self.node_mass[q_i]


    @ti.kernel
    def construct_Xg_inv(self):
        for i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[i]
            a = ti.Vector([self.node_pos_init[ia].x, self.node_pos_init[ia].y])
            b = ti.Vector([self.node_pos_init[ib].x, self.node_pos_init[ib].y])
            c = ti.Vector([self.node_pos_init[ic].x, self.node_pos_init[ic].y])
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.Xg_inv[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_lhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            self.lhs[q_i, q_i] += self.node_mass[q_i] / self.dt**2


    @ti.kernel
    def construct_lhs_stretch(self):
        # https://medium.com/@victorlouisdg/jax-cloth-tutorial-part-1-e7a0e285864f
        for f_i in range(self.ELEMENT_N):
            Xg_inv = self.Xg_inv[f_i]
            a, b, c, d = Xg_inv[0, 0], Xg_inv[0, 1], Xg_inv[1, 0], Xg_inv[1, 1]

            # F's dim=4*6，flatten(F)按照列优先
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

            stretch_weight = self.stretch_weight[f_i]
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]


    @ti.kernel
    def construct_lhs_positional(self):
        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight

        for i in range(self.CON_N):
            q_i = self.contact_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight


    def precomputation(self):
        self.construct_lhs_mass()
        self.construct_lhs_stretch()
        self.construct_lhs_positional()

    
    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for q_i in range(self.PARTICLE_N):
            idx1, idx2 = dim*q_i, dim*q_i+1
            self.sn[idx1] = self.node_pos[q_i].x + self.node_vel[q_i].x * dt
            self.sn[idx2] = self.node_pos[q_i].y + self.node_vel[q_i].y * dt

        # Contact particles update
        for idx in range(self.CON_N):
            q_i = self.contact_particle_ti[idx]
            self.sn[q_i*dim] = self.node_pos[q_i].x + self.contact_vel[idx].x * dt
            self.sn[q_i*dim + 1] = self.node_pos[q_i].y + self.contact_vel[idx].y * dt


    @ti.kernel
    def warm_start(self):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = self.sn[q_i*2]
            self.node_pos_new[q_i].y = self.sn[q_i*2 + 1]


    @ti.kernel
    def construct_rhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            idx1, idx2 = q_i*self.dim, q_i*self.dim+1
            self.rhs[idx1] += self.node_mass[q_i] * self.sn[idx1] / self.dt**2
            self.rhs[idx2] += self.node_mass[q_i] * self.sn[idx2] / self.dt**2


    @ti.kernel
    def construct_rhs_stretch(self):
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos_new[idx1], self.node_pos_new[idx2], self.node_pos_new[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = F_i
            # print(f"F_i:{F_i:e}")

            U, sig, V = ti.svd(F_i, ti.f64)
            self.ele_u[f_i] = U
            self.ele_v[f_i] = V
            self.stretch_stress[f_i] = ti.Vector([sig[0,0], sig[1,1]], dt=ti.f64)
            # print(f"U:{U:e}; sig:{sig:e}; V:{V:e}")
            self.Bp_shear[f_i] = U @ ti.Matrix([[1., 0], [0., 1]], ti.f64) @ V.transpose()
            self.stretch_energy[f_i] = 0.5 * self.ele_volume[f_i] * ((sig[0,0]-1.)**2 + (sig[1,1]-1.)**2)
            # print(f"Bp_shear:{self.Bp_shear[f_i]:e}")

        for f_i in range(self.ELEMENT_N):
            Bp_shear_i = self.Bp_shear[f_i]
            F_AT = self.F_A[f_i].transpose()

            # Bp_shear_i做transpose，因为AT需要与Bp的x，y，z分别矩阵乘法
            F_ATBp = F_AT @ Bp_shear_i.transpose() * self.stretch_weight[f_i]

            for q_i, dim_idx in ti.ndrange(3, 2):
                q_idx = self.ele[f_i][q_i]
                self.rhs[q_idx*2+dim_idx] += F_ATBp[q_i, dim_idx]
                # self.rhs_stretch[q_idx*3+dim_idx] += F_ATBp_lim[q_i, dim_idx]


    @ti.kernel
    def construct_rhs_poistional(self):
        for q_i in range(self.FIX_N):
            weight = self.positional_weight
            q_idx = self.fix_particle_ti[q_i]
            q_i_x, q_i_y = q_idx*self.dim, q_idx*self.dim+1
            self.rhs[q_i_x] += weight * self.node_pos_init[q_idx].x
            self.rhs[q_i_y] += weight * self.node_pos_init[q_idx].y

        for i in range(self.CON_N):
            q_i = self.contact_particle_ti[i]
            self.rhs[q_i*2] += self.positional_weight * (self.node_pos[q_i].x + self.contact_vel[i].x * self.dt)
            self.rhs[q_i*2+1] += self.positional_weight * (self.node_pos[q_i].y + self.contact_vel[i].y * self.dt)


    def local_solve(self):
        self.rhs.fill(0.)
        self.rhs_stretch.fill(0.)
        self.construct_rhs_mass()
        self.construct_rhs_stretch()
        self.construct_rhs_poistional()


    @ti.kernel
    def update_pos_new(self, sol_x:ti.types.ndarray(), sol_y:ti.types.ndarray()):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = sol_x[q_i]
            self.node_pos_new[q_i].y = sol_y[q_i]


    @ti.kernel
    def update_vel_pos(self):
        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_pos_new[q_idx] = self.node_pos[q_idx] + self.contact_vel[idx] * self.dt

        for i in range(self.PARTICLE_N):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]

        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.node_pos[q_i] = self.node_pos_init[q_i]
            self.node_vel[q_i] = ti.Vector([0., 0.], dt=ti.f64)

        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_vel[q_idx] = ti.Vector([0., 0.], dt=ti.f64)


    def substep(self, step_num:int):
        self.construct_sn()
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            # print(f"Iteration: {itr} ------------------------------------")
            self.local_solve()
            rhs_np = self.rhs.to_numpy()
            # print(f"Rhs:\n{self.rhs_stretch.to_numpy().reshape(-1, 2)}")
            # Split rhs_np into x,y components
            rhs_np_x = rhs_np[0::2]
            rhs_np_y = rhs_np[1::2]

            node_pos_new_np_x = self.pre_fact_lhs_solve(rhs_np_x)
            node_pos_new_np_y = self.pre_fact_lhs_solve(rhs_np_y)

            self.update_pos_new(node_pos_new_np_x, node_pos_new_np_y)
            # print(f"Node pos new:\n", self.node_pos_new.to_numpy().reshape(-1, 2))
        
        self.update_vel_pos()


    @ti.kernel
    def construct_dx_const(self):
        """解决Movement Constraint中dBp/dq为常数的情况
        """
        for q_i in range(self.PARTICLE_N):
            for d in ti.static(range(self.dim)):
                self.dx_const[q_i*self.dim+d] = self.node_mass[q_i] / self.dt**2

        for q_i in ti.static(self.contact_particle_list):
            for d in ti.static(range(self.dim)):
                self.dx_const[q_i*self.dim+d] += self.positional_weight


    @ti.kernel
    def hessian_stretch(self):
        """计算diffpd中stretch constraint的二阶导数矩阵(Hessian)
        """
        self.dBp_stretch.fill(0.)
        dim = self.dim
        for f_i in range(self.ELEMENT_N):
            F_Ai = self.F_A[f_i]
            U, sig, V = self.ele_u[f_i], self.stretch_stress[f_i], self.ele_v[f_i]
            # print(f"F: {self.F[f_i]}")
            # print(f"U: {U}; sig: {sig}; V: {V}")
            # print(f"reconstructed F: {U @ V.transpose()}")

            dBp_dF = ti.Matrix.zero(ti.f64, 4, 4)
            for m in range(2):
                for n in range(2):
                    Omega_uv = ti.Matrix.zero(ti.f64, 2, 2)
                    Omega_uv[0, 1] = (U[m,0]*V[n,1] - U[m,1]*V[n,0]) / (sig[0] + sig[1])
                    Omega_uv[1, 0] = -Omega_uv[0, 1]
                    dBp_df = U @ Omega_uv @ V.transpose()
                    dBp_dF[dim*m + n, :] = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[1, 0], dBp_df[1, 1]])

            idx1, idx2, idx3 = self.ele[f_i]
            for m, n in ti.ndrange(2, 2):                   # 2*2表示Bp*q的维数
                dBp_dF_i = ti.Matrix.zero(ti.f64, 2, 2)     # 2表示参数空间的维数
                for k, l in ti.ndrange(2, 2):
                    dBp_dF_i[k, l] = dBp_dF[2*m+k, 2*n+l]
                AT_dBp_dq_i = F_Ai.transpose() @ dBp_dF_i @ F_Ai    # A.T @ (dBp_x / dF_x) @ A
                AT_dBp_dq_i *= self.stretch_weight[f_i]

                row_idx_vec = ti.Vector([idx1*dim+m, idx2*dim+m, idx3*dim+m])
                col_idx_vec = ti.Vector([idx1*dim+n, idx2*dim+n, idx3*dim+n])
                for k, l in ti.ndrange(3, 3):               # AT_dBp_dq_i's dim
                    row_idx = row_idx_vec[k]
                    col_idx = col_idx_vec[l]
                    self.dBp_stretch[row_idx, col_idx] += AT_dBp_dq_i[k, l]


    def construct_g_hessian(self):
        """论文中g函数的Hessian矩阵
        """
        self.hessian_stretch()
        self.dA = self.dBp_stretch.to_numpy()


    def construct_L(self):
        """marker point的期望位置

        Returns: Tuple[error (ti.Vector), L (ti.f64)]
        """
        self.dL_dq_contact.fill(0.)
        idx = self.marker_idx
        desired_pos = self.marker_pos_desired[0]
        current_pos = self.node_pos[idx]
        error = current_pos - desired_pos
        loss = error.norm() ** 2
        self.dL_dq_contact[idx*2]     = 2*(current_pos.x - desired_pos.x)
        self.dL_dq_contact[idx*2 + 1] = 2*(current_pos.y - desired_pos.y)
        return error, loss
    

    def construct_L_distance(self):
        """将两个标记点拉伸到指定间距
        """
        self.dL_dq_contact.fill(0.)
        idx1, idx2 = 93, 105
        pos1_np, pos2_np = self.node_pos[idx1].to_numpy(), self.node_pos[idx2].to_numpy()
        d_tmp = np.linalg.norm(pos1_np - pos2_np)
        d_desired = 0.018

        a, b, c = line_from_points_2d(self.node_pos_init[idx1].to_numpy(), self.node_pos_init[idx2].to_numpy())
        line_noraml = np.array([a, b], dtype=np.float64)

        # 两个点到直线的距离
        line_distance1 = line_noraml.dot(pos1_np) + c
        line_distance2 = line_noraml.dot(pos2_np) + c

        loss = (d_tmp - d_desired) ** 2 + line_distance1 ** 2 + line_distance2 ** 2
        grad1 = 2*(d_tmp - d_desired) * (pos1_np - pos2_np) / d_tmp + line_distance1 * line_noraml
        grad2 = 2*(d_tmp - d_desired) * (pos2_np - pos1_np) / d_tmp + line_distance2 * line_noraml

        self.dL_dq_contact[idx1*2]     = grad1[0]
        self.dL_dq_contact[idx1*2 + 1] = grad1[1]
        self.dL_dq_contact[idx2*2]     = grad2[0]
        self.dL_dq_contact[idx2*2 + 1] = grad2[1]

        return d_tmp, loss


    def compute_z(self, itr_num:int):
        """diffpd中的z的迭代计算
        """
        dL_dq_contact_np = self.dL_dq_contact.to_numpy()
        z_np_old = self.z.to_numpy()
        z_np = np.zeros_like(z_np_old)
        for itr in range(itr_num):
            rhs_dA = self.dA @ z_np + dL_dq_contact_np
            rhs_dA_x = rhs_dA[0::2]
            rhs_dA_y = rhs_dA[1::2]

            z_new_x = self.pre_fact_lhs_solve(rhs_dA_x)
            z_new_y = self.pre_fact_lhs_solve(rhs_dA_y)
            z_np = np.ravel(np.column_stack((z_new_x, z_new_y)))
        self.z.from_numpy(z_np)
        # print(f"interative error: {np.linalg.norm(z_np - z_np_old)}")


    def compute_dcontact(self):
        """ \partial L / \partial y with contact action, 计算关于contact的导数, 用于控制
        """
        # error_tmp, loss_tmp = self.construct_L()
        d_tmp, loss_tmp = self.construct_L_distance()
        self.construct_g_hessian()
        self.compute_z(10)

        # print(f'Error items: {error_tmp}; Loss sum: {loss_tmp}')
        print(f"Distance: {d_tmp}; Loss: {loss_tmp}")

        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())


    def preset_gui(self, pos:List[float], target:List[float], up_orient:List[float]):
        """Taichi GUI pre-setting
        Args:
            pos (List[float]): Camera position
            target (List[float]): Camera visual target
            up_orient (List[float]): Camera orientation
        """
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
        for q_i in self.contact_particle_list:
            self.node_color[q_i] = ti.Vector([0., 0., 1.])
        for q_i in [93, 105]:
            self.node_color[q_i] = ti.Vector([0., 1., 0.])


    def gui_show(self, SHOW_FLAG:bool=True, WRITE_FLAG:bool=False, itr_num:int=0):
        if SHOW_FLAG is False:
            return
        self.scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        self.scene.ambient_light((.8, .8, .8))

        node_show_np = np.hstack((self.node_pos.to_numpy(), np.zeros((self.PARTICLE_N, 1))), dtype=np.float32)
        self.node_show.from_numpy(node_show_np)
        self.scene.particles(self.node_show, radius=0.001, per_vertex_color=self.node_color)
        self.scene.lines(self.node_show, width=1., indices=self.edge_show, color=(0., 0., 0.), vertex_count=0)

        self.canvas.scene(self.scene)
        self.canvas.set_background_color((1., 1., 1.))

        if WRITE_FLAG is True:
            self.window.save_image(f"FigureWrite/{itr_num:05d}.png")
        self.window.show()


    @ti.kernel
    def init_vel(self):
        for q_i in range(self.PARTICLE_N):
            if self.node_pos_init[q_i].y > self.shape[0] - 1.e-3:
                self.node_vel[q_i].y = 30.
            else:
                self.node_vel[q_i].y = 0.


def main():
    gain = 5.e2
    class Soft(SoftObject2D):
        def __init__(self, shape, fix:List[int], contact:List[int], 
                     E:float, nu:float, dt:float, density:float, **kwargs):
            super().__init__(shape, fix, contact, E, nu, dt, density, **kwargs)

    params = {"E": 1.e4, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = Soft([0.1, 0.1], range(11), [66, 120], **params)
    soft.preset_gui([0.05, 0.05, 0.2], [0.05, 0.05, 0.], [0., 1., 0.])

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    # soft.init_vel()

    for itr in range(100):
        print(f"Time Step: {itr} ======================================")
        time.sleep(0.1)
        soft.substep(itr)

        soft.compute_dcontact()
        dy_dcontact = soft.dy_contact.reshape(-1, 2)        # reshape到与接触点个数相同
        end_speed = -gain * dy_dcontact[soft.contact_particle_list]
        end_speed_compress = compress_vectors(end_speed, 0.04)
        soft.contact_vel.from_numpy(end_speed_compress)
        print(f"End speed: {end_speed.flatten()}")
        print(f"Deformation Energy: {soft.stretch_energy.to_numpy().sum():e}")

        soft.gui_show(True, False, itr)


if __name__ == "__main__":
    main()