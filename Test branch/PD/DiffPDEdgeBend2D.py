"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型, DiffPD 求解材料参数的梯度与动作的梯度
cotangent weights: https://rodolphe-vaillant.fr/entry/33/curvature-of-a-triangle-mesh-definition-and-computation
inextensible cloth with bending: https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=2666
Edge Curvature: https://dl.acm.org/doi/abs/10.5555/1281957.1281987 A quadratic bending model for inextensible surfaces
created at 2024-12-30 by hsy
"""

import os
import sys
import time
from typing import List, Dict, DefaultDict, Tuple
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
import taichi as ti
import meshtaichi_patcher as Patcher
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import mesh_obj_tri, write_obj
from Utilize.GuiTaichi import gui_set
from Utilize.MathTaichi import svd_3x2_new, cotangent_ti


def read_msh(file_path):
    # 考虑从外部导入
    pass


@ti.data_oriented
class SoftBend2D:
    def __init__(self, shape, E:float, nu:float, dt:float, density:float, g=-9.8):
        self.shape = shape
        # 是否传入的是 .msh 文件(已划分网格)
        if isinstance(self.shape, str):
            node_np, edge_np, ele_np = read_msh(self.shape)
        else:
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.01)
            node_np = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))         # di: N*3
            np.savetxt("node_np.csv", node_np, fmt='%f', delimiter=",")
            np.savetxt("edge_np.csv", edge_np, fmt='%d', delimiter=",")
            np.savetxt("ele_np.csv", ele_np, fmt='%d', delimiter=",")

        obj_file:str = "Mesh/shape.obj"
        write_obj(obj_file, node_np, ele_np)

        # exit(0)
        self.solve_itr:int = 10
        self.strain_lim_rate:float = 0.1            # Strain limit rate
        self.E, self.nu, self.dt, self.density, self.g = E, nu, dt, density, g
        self.dim = 3
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
       
        self.mesh = Patcher.load_mesh(obj_file, relations=["VV", "VE", "VF", "EV", "EF", "FV", "FE"])
        self.mesh.verts.place({
            "pos": ti.types.vector(3, ti.f64),
            "pos_init": ti.types.vector(3, ti.f64),
            "neighbor_num": ti.i32                  # 一环邻居数(包括自己)
        }, reorder=False)
        self.mesh.edges.place({
            "v_g": ti.types.vector(3, ti.f64),
            "bend_weight": ti.f64,
            "voronoi": ti.f64,
            "border": bool
        }, reorder=False)
        self.mesh.faces.place({
            "volume": ti.f64
        }, reorder=False)

        self.mesh.verts.pos.from_numpy(self.mesh.get_position_as_numpy().astype(np.float64))
        self.mesh.verts.pos_init.from_numpy(self.mesh.get_position_as_numpy().astype(np.float64))

        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.node_pos = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_vel = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_voronoi = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass_sum = ti.field(dtype=ti.f64, shape=())
        self.node_pos_init.from_numpy(node_np.astype(np.float64))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge.from_numpy(edge_np.astype(np.int32))

        self.ele = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)
        self.ele_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele.from_numpy(ele_np.astype(np.int32))

        self.bend_weight = 0.5e0
        self.stretch_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_lim_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.positional_weight = 0.         # define later

        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.ele_u = ti.Matrix.field(3, 3, dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele_v = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_stress = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_N)    # stretch stress       
        self.Bp_shear = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)       # stretch part
        self.Bp_shear_lim = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)   # strain-limit part

        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_lim_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)

        self.cot_weight = ti.Matrix.field(1, 4, dtype=ti.f64, shape=self.EDGE_N)
        self.Bp_bend = ti.Vector.field(3, dtype=ti.f64, shape=self.EDGE_N)
        self.bend_energy = ti.field(dtype=ti.f64, shape=self.EDGE_N)
        self.v_f = ti.Vector.field(3, dtype=ti.f64, shape=self.EDGE_N)

        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_bend_ti = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.rhs_stretch = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.rhs_bend = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)

        self.dBp_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.ELEMENT_N*3))
        self.dBp_bend = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.EDGE_N*3))
        self.g_hessian = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.PARTICLE_N*3))
        self.dA = None      # dim: 3N*3N, 用于初始化
        self.dL_dq_param = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.z = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.nablaE_dw_stretch = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.nablaE_dw_bend = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)

        self.pre_fact_lhs_solve = None

        self.fix_particle_list = list(range(11))
        # self.contact_particle_list = list(range(420, 441))
        self.contact_particle_list = list(range(110, 121))
        self.FIX_N = len(self.fix_particle_list)
        self.CON_N = len(self.contact_particle_list)
        self.fix_particle_ti = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)
        self.contact_particle_ti.from_numpy(np.array(self.contact_particle_list).astype(np.int32))
        self.contact_vel = ti.Vector.field(3, dtype=ti.f64, shape=self.CON_N)
        self.contact_vel.fill(0.)

        self.construct_mass()
        self.construct_cotangent()
        self.construct_Xg_inv()
        self.positional_weight = 1.e3 * self.node_mass_sum[None] / self.PARTICLE_N / self.dt**2
        
        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")
        print(f"Positional weight: {self.positional_weight}")


    @ti.kernel
    def construct_mass(self):
        for f in self.mesh.faces:
            f_id = f.id
            v1, v2, v3 = f.verts[0], f.verts[1], f.verts[2]
            e1, e2, e3 = f.edges[0], f.edges[1], f.edges[2]
            ele_volume_tmp = 0.5 * ((v2.pos_init - v1.pos_init).cross(v3.pos_init - v1.pos_init)).norm()

            self.node_voronoi[v1.id] += ele_volume_tmp / 3.
            self.node_voronoi[v2.id] += ele_volume_tmp / 3.
            self.node_voronoi[v3.id] += ele_volume_tmp / 3.
            e1.voronoi += ele_volume_tmp / 3.
            e2.voronoi += ele_volume_tmp / 3.
            e3.voronoi += ele_volume_tmp / 3.
            f.volume = ele_volume_tmp

            self.ele_volume[f_id] = ele_volume_tmp
            self.stretch_weight[f_id] = 2 * self.mu * self.ele_volume[f_id]
            self.stretch_lim_weight[f_id] = 1.e3 * self.stretch_weight[f_id]

        for q_i in range(self.PARTICLE_N):
            self.node_mass[q_i] = self.density * self.node_voronoi[q_i]
            # self.node_mass[q_i] = 0.
            self.node_mass_sum[None] += self.node_mass[q_i]


    @ti.kernel
    def construct_cotangent(self):
        ti.mesh_local(self.mesh.edges.bend_weight, self.mesh.edges.voronoi, self.mesh.edges.border)
        for e in self.mesh.edges:
            if e.faces.size > 1:
                e.bend_weight = self.bend_weight * e.voronoi
            else:
                e.bend_weight = 0.
                e.border = True


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
    def construct_lhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            self.lhs[q_i, q_i] += self.node_mass[q_i] / self.dt**2

    
    @ti.kernel
    def construct_lhs_stretch(self):
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

            stretch_weight = self.stretch_weight[f_i]
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]
                
                # Strain-limit
                self.lhs[lhs_row_idx, lhs_col_idx] += self.stretch_lim_weight[f_i] * ATA[A_row_idx, A_col_idx]


    @ti.kernel
    def construct_lhs_bend(self):
        """v_g: corresponding to normal curvature vector of dscrete edges (= curvature H*3)
        """
        ti.mesh_local(self.mesh.verts.pos_init, self.mesh.edges.voronoi, self.mesh.edges.bend_weight)
        for e in self.mesh.edges:
            if e.faces.size > 1:
                v1, v2 = e.verts[0], e.verts[1]
                tri1, tri2 = e.faces[0], e.faces[1]

                m, n = 0, 0
                for i in range(3):
                    if tri1.verts[i].id != v1.id and tri1.verts[i].id != v2.id:
                        m = i
                    if tri2.verts[i].id != v1.id and tri2.verts[i].id != v2.id:
                        n = i
                v3_1, v3_2 = tri1.verts[m], tri2.verts[n]

                cot01 = cotangent_ti(v2.pos_init - v1.pos_init, v3_1.pos_init - v1.pos_init)
                cot02 = cotangent_ti(v2.pos_init - v1.pos_init, v3_2.pos_init - v1.pos_init)
                cot03 = cotangent_ti(v1.pos_init - v2.pos_init, v3_1.pos_init - v2.pos_init)
                cot04 = cotangent_ti(v1.pos_init - v2.pos_init, v3_2.pos_init - v2.pos_init)

                A = ti.Matrix([[cot03+cot04, cot01+cot02, -cot01-cot03, -cot02-cot04]], dt=ti.f64) / e.voronoi
                ATA = A.transpose() @ A
                self.cot_weight[e.id] = A
                v_g_matrix = A @ ti.Matrix.rows([v1.pos_init, v2.pos_init, v3_1.pos_init, v3_2.pos_init])
                e.v_g = ti.Vector([v_g_matrix[0, 0], v_g_matrix[0, 1], v_g_matrix[0, 2]])

                q_idx_vec = ti.Vector([v1.id, v2.id, v3_1.id, v3_2.id])
                for row_idx, col_idx in ti.ndrange(4, 4):
                    lhs_row_idx, lhs_col_idx = q_idx_vec[row_idx], q_idx_vec[col_idx]
                    self.lhs[lhs_row_idx, lhs_col_idx] += e.bend_weight * ATA[row_idx, col_idx]
                    # self.lhs_bend_ti[lhs_row_idx, lhs_col_idx] += e.bend_weight * ATA[row_idx, col_idx]
            
    
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
        self.construct_lhs_bend()
        self.construct_lhs_positional()


    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for q_i in range(self.PARTICLE_N):
            idx1, idx2, idx3 = dim*q_i, dim*q_i+1, dim*q_i+2
            self.sn[idx1] = self.node_pos[q_i].x + self.node_vel[q_i].x * dt
            self.sn[idx2] = self.node_pos[q_i].y + self.node_vel[q_i].y * dt
            self.sn[idx3] = self.node_pos[q_i].z + self.node_vel[q_i].z * dt + self.g * dt**2       # Gravity

        # Contact particles update
        for idx in range(self.CON_N):
            q_i = self.contact_particle_ti[idx]
            self.sn[q_i*3] = self.node_pos[q_i].x + self.contact_vel[idx].x * dt
            self.sn[q_i*3 + 1] = self.node_pos[q_i].y + self.contact_vel[idx].y * dt
            self.sn[q_i*3 + 2] = self.node_pos[q_i].z + self.contact_vel[idx].z * dt + self.g * dt**2


    @ti.kernel
    def warm_start(self):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = self.sn[q_i*3]
            self.node_pos_new[q_i].y = self.sn[q_i*3+1]
            self.node_pos_new[q_i].z = self.sn[q_i*3+2]


    @ti.kernel
    def construct_rhs_mass(self):
        for q_i in range(self.PARTICLE_N):
            idx1, idx2, idx3 = q_i*3, q_i*3+1, q_i*3+2
            self.rhs[idx1] += self.node_mass[q_i] * self.sn[idx1] / self.dt**2
            self.rhs[idx2] += self.node_mass[q_i] * self.sn[idx2] / self.dt**2
            self.rhs[idx3] += self.node_mass[q_i] * self.sn[idx3] / self.dt**2


    @ti.kernel
    def construct_rhs_stretch(self):
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            a, b, c = self.node_pos_new[idx1], self.node_pos_new[idx2], self.node_pos_new[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)
            self.F[f_i] = F_i
            # print(f"F_i:{F_i:e}")

            U, sig, V = svd_3x2_new(F_i)
            self.ele_u[f_i], self.ele_v[f_i], self.stretch_stress[f_i] = U, V, sig
            # print(f"U:{U:e}; sig:{sig:e}; V:{V:e}")
            self.Bp_shear[f_i] = U @ ti.Matrix([[1., 0], [0., 1], [0, 0]], ti.f64) @ V.transpose()
            self.stretch_energy[f_i] = 0.5 * self.stretch_weight[f_i] * ((sig[0]-1.)**2 + (sig[1]-1.)**2)
            # print(f"Bp_shear:{self.Bp_shear[f_i]:e}")

            s_lim = ti.Vector.zero(ti.f64, 2)
            for dim in range(2):
                if sig[dim] > (1 + self.strain_lim_rate):
                    s_lim[dim] = 1 + self.strain_lim_rate
                elif sig[dim] < (1 - self.strain_lim_rate):
                    s_lim[dim] = 1 - self.strain_lim_rate
                else:
                    s_lim[dim] = sig[dim]
            self.Bp_shear_lim[f_i] = U @ ti.Matrix([[s_lim[0], 0], [0., s_lim[1]], [0, 0]], ti.f64) @ V.transpose()
            self.stretch_lim_energy[f_i] = 0.5 * self.stretch_lim_weight[f_i] * ((sig[0]-s_lim[0])**2 + (sig[1]-s_lim[1])**2)

        for f_i in range(self.ELEMENT_N):
            Bp_shear_i = self.Bp_shear[f_i]
            Bp_shear_lim_i = self.Bp_shear_lim[f_i]
            F_AT = self.F_A[f_i].transpose()

            # Bp_shear_i做transpose，因为AT需要与Bp的x，y，z分别矩阵乘法
            F_ATBp = F_AT @ Bp_shear_i.transpose() * self.stretch_weight[f_i]
            F_ATBp_lim = F_AT @ Bp_shear_lim_i.transpose() * self.stretch_lim_weight[f_i]

            for q_i, dim_idx in ti.ndrange(3, 3):
                q_idx = self.ele[f_i][q_i]
                self.rhs[q_idx*3+dim_idx] += F_ATBp[q_i, dim_idx]
                self.rhs[q_idx*3+dim_idx] += F_ATBp_lim[q_i, dim_idx]
                # self.rhs_stretch[q_idx*3+dim_idx] += F_ATBp_lim[q_i, dim_idx]


    @ti.kernel
    def construct_rhs_bend(self):
        ti.mesh_local(self.mesh.edges.v_g, self.mesh.edges.bend_weight)
        for e in self.mesh.edges:
            if e.faces.size > 1:
                v1, v2 = e.verts[0], e.verts[1]
                tri1, tri2 = e.faces[0], e.faces[1]

                m, n = 0, 0
                for i in range(3):
                    if tri1.verts[i].id != v1.id and tri1.verts[i].id != v2.id:
                        m = i
                    if tri2.verts[i].id != v1.id and tri2.verts[i].id != v2.id:
                        n = i
                v3_1, v3_2 = tri1.verts[m], tri2.verts[n]

                v_f_matrix = self.cot_weight[e.id] @ ti.Matrix.rows([v1.pos, v2.pos, v3_1.pos, v3_2.pos])
                v_f = ti.Vector([v_f_matrix[0, 0], v_f_matrix[0, 1], v_f_matrix[0, 2]])
                self.v_f[e.id] = v_f

                if v_f.norm() == 0.:
                    self.Bp_bend[e.id] = e.v_g
                else:
                    self.Bp_bend[e.id] = e.v_g.norm() * v_f / v_f.norm()

                self.bend_energy[e.id] = 0.5 * e.bend_weight * (v_f - self.Bp_bend[e.id]).norm()**2
    
                ATBp = self.cot_weight[e.id].transpose() @ ti.Matrix.rows([self.Bp_bend[e.id]]) * e.bend_weight

                q_idx_vec = ti.Vector([v1.id, v2.id, v3_1.id, v3_2.id])
                for q_i, dim_idx in ti.ndrange(4, 3):
                    q_idx = q_idx_vec[q_i]
                    self.rhs[q_idx*3+dim_idx] += ATBp[q_i, dim_idx]
                    # self.rhs_bend[q_idx*3+dim_idx] += ATBp[q_i, dim_idx]


    @ti.kernel
    def construct_rhs_poistional(self):
        for q_i in range(self.FIX_N):
            weight = self.positional_weight
            q_idx = self.fix_particle_ti[q_i]
            q_i_x, q_i_y, q_i_z = q_idx*3, q_idx*3+1, q_idx*3+2
            self.rhs[q_i_x] += weight * self.node_pos_init[q_idx].x
            self.rhs[q_i_y] += weight * self.node_pos_init[q_idx].y
            self.rhs[q_i_z] += weight * self.node_pos_init[q_idx].z

        for i in range(self.CON_N):
            q_i = self.contact_particle_ti[i]
            self.rhs[q_i*3] += self.positional_weight * (self.node_pos[q_i].x + self.contact_vel[i].x * self.dt)
            self.rhs[q_i*3+1] += self.positional_weight * (self.node_pos[q_i].y + self.contact_vel[i].y * self.dt)
            self.rhs[q_i*3+2] += self.positional_weight * (self.node_pos[q_i].z + self.contact_vel[i].z * self.dt)


    def local_solve(self):
        self.rhs.fill(0.)
        self.rhs_stretch.fill(0.)
        self.rhs_bend.fill(0.)
        self.construct_rhs_mass()
        self.construct_rhs_stretch()
        self.construct_rhs_bend()
        self.construct_rhs_poistional()


    @ti.kernel
    def update_pos_new(self, sol_x:ti.types.ndarray(), sol_y:ti.types.ndarray(), sol_z:ti.types.ndarray()):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = sol_x[q_i]
            self.node_pos_new[q_i].y = sol_y[q_i]
            self.node_pos_new[q_i].z = sol_z[q_i]


    @ti.kernel
    def update_vel_pos(self):
        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_pos_new[q_idx] = self.node_pos[q_idx] + self.contact_vel[idx] * self.dt

        for i in range(self.PARTICLE_N):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]
            self.mesh.verts.pos[i] = self.node_pos[i]

        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.node_pos[q_i] = self.node_pos_init[q_i]
            self.node_vel[q_i] = ti.Vector([0., 0., 0.], dt=ti.f64)
            self.mesh.verts.pos[q_i] = self.node_pos_init[q_i]

        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_vel[q_idx] = ti.Vector([0., 0., 0.], dt=ti.f64)


    @ti.kernel
    def hessian_stretch(self):
        self.dBp_stretch.fill(0.)
        for i in range(self.ELEMENT_N):
            F_Ai = self.F_A[i]
            U, sig, V = self.ele_u[i], self.stretch_stress[i], self.ele_v[i]

            dBp_dF = ti.Matrix.zero(ti.f64, 6, 6)
            for m in range(3):
                for n in range(2):
                    Omega_uv = ti.Matrix.zero(ti.f64, 3, 2)
                    Omega_uv[0, 1] = (U[m,0]*V[n,1] - U[m,1]*V[n,0]) / (sig[0] + sig[1])
                    Omega_uv[1, 0] = -Omega_uv[0, 1]
                    Omega_uv[2, 0] = U[m,2]*V[n,0] / sig[0]
                    Omega_uv[2, 1] = U[m,2]*V[n,1] / sig[1]
                    dBp_df = U @ Omega_uv @ V.transpose()
                    dBp_dF[2*m+n, :] = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[1, 0], dBp_df[1, 1], dBp_df[2, 0], dBp_df[2, 1]])

            idx1, idx2, idx3 = self.ele[i]
            for m, n in ti.ndrange(3, 3):       # 3*3表示维数, 
                dBp_dF_i = ti.Matrix.zero(ti.f64, 2, 2)
                for k, l in ti.ndrange(2, 2):
                    dBp_dF_i[k, l] = dBp_dF[2*m+k, 2*n+l]
                AT_dBp_dq_i = F_Ai.transpose() @ dBp_dF_i @ F_Ai    # A.T @ dBp_x / dF_x @ A

                row_idx_vec = ti.Vector([idx1*3+m, idx2*3+m, idx3*3+m])
                col_idx_vec = ti.Vector([idx1*3+n, idx2*3+n, idx3*3+n])
                for k, l in ti.ndrange(3, 3):       # AT_dBp_dq_i's dim
                    row_idx = row_idx_vec[k]
                    col_idx = col_idx_vec[l]
                    self.dBp_stretch[row_idx, col_idx] += AT_dBp_dq_i[k, l]


    @ti.kernel
    def hessian_bend(self):
        self.dBp_bend.fill(0.)


    def construct_g_hessian(self):
        self.hessian_stretch()
        self.hessian_bend()
        self.dA = self.dBp_stretch.to_numpy() + self.dBp_bend.to_numpy()

    
    def compute_z(self, itr_num:ti.i32):
        dL_dq_param_np = self.dL_dq_param.to_numpy()        # 此处的z是关于param的导数
        z_np = self.z.to_numpy()
        for itr in range(itr_num):
            rhs_dA = self.dA @ z_np + dL_dq_param_np
            z_new_np = self.pre_fact_lhs_solve(rhs_dA)
            z_np = z_new_np
        self.z.from_numpy(z_np)


    def construct_L(self):
        """ construct series type of Loss
        """
        # self.dL_dq_param
        pass


    @ti.kernel
    def construct_energy_grad_params(self):
        """ \partial ΔE / \partial w; w有两个参数: stretch weight & bend weight
        """
        self.nablaE_dw_stretch.fill(0.)
        for f_i in range(self.ELEMENT_N):
            idx1, idx2, idx3 = self.ele[f_i]
            idx_vec = ti.Vector([idx1, idx2, idx3])
            nabla_Ei_s = self.ele_volume[f_i] * self.F_A[f_i].transpose() @ (self.F[f_i] - self.Bp_shear[f_i]).transpose()      # dim: 3*3
            for i, j in ti.ndrange(3, 3):       # X，Y，Z维度按列排序；节点序号按行排序
                q_idx = idx_vec[i]*3 + j
                self.nablaE_dw_stretch[q_idx] += nabla_Ei_s[i, j]

        self.nablaE_dw_bend.fill(0.)
        ti.mesh_local(self.mesh.edges.voronoi)
        for e in self.mesh.edges:
            if e.faces.size > 1:
                v1, v2 = e.verts[0], e.verts[1]
                tri1, tri2 = e.faces[0], e.faces[1]

                m, n = 0, 0
                for i in range(3):
                    if tri1.verts[i].id != v1.id and tri1.verts[i].id != v2.id:
                        m = i
                    if tri2.verts[i].id != v1.id and tri2.verts[i].id != v2.id:
                        n = i
                v3_1, v3_2 = tri1.verts[m], tri2.verts[n]

                nabla_Ei_b = e.voronoi * self.cot_weight[e.id].outer_product(self.v_f[e.id] - self.Bp_bend[e.id])   # dim: 4*3
                q_idx_vec = ti.Vector([v1.id, v2.id, v3_1.id, v3_2.id])
                for i, j in ti.ndrange(4, 3):
                    q_idx = q_idx_vec[i]*3 + j
                    self.nablaE_dw_bend[q_idx] += nabla_Ei_b[i, j]

    
    def compute_dL_dparam(self):
        """ \partial L / \partial w
        """
        self.construct_L()
        self.construct_g_hessian()
        self.compute_z(10)
        self.construct_energy_grad_params()

        z_np = self.z.to_numpy()
        self.dparam_stretch = z_np.dot(self.nablaE_dw_stretch.to_numpy())
        self.dparam_bend = z_np.dot(self.nablaE_dw_bend.to_numpy())


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
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            # print(f"Iteration: {itr} ------------------------------------")
            self.local_solve()
            rhs_np = self.rhs.to_numpy()
            # print(f"Rhs:\n{self.rhs_stretch.to_numpy().reshape(-1, 3)}")
            # Split rhs_np into x,y,z components
            rhs_np_x = rhs_np[0::3]
            rhs_np_y = rhs_np[1::3]
            rhs_np_z = rhs_np[2::3]

            node_pos_new_np_x = self.pre_fact_lhs_solve(rhs_np_x)
            node_pos_new_np_y = self.pre_fact_lhs_solve(rhs_np_y)
            node_pos_new_np_z = self.pre_fact_lhs_solve(rhs_np_z)

            self.update_pos_new(node_pos_new_np_x, node_pos_new_np_y, node_pos_new_np_z)
            # print(f"Node pos new:\n", self.node_pos_new.to_numpy().reshape(-1, 3))
        
        self.update_vel_pos()

    
    @ti.kernel
    def init_vel(self):
        for q_i in range(self.PARTICLE_N):
            if self.node_pos_init[q_i].y > self.shape[0] - 1.e-3:
                self.node_vel[q_i].y = 50.
            else:
                self.node_vel[q_i].y = 0.


    @ti.kernel
    def cal_v_f(self):
        ti.mesh_local(self.mesh.verts.pos)
        for e in self.mesh.edges:
            if e.border == False:
                v1, v2 = e.verts[0], e.verts[1]
                tri1, tri2 = e.faces[0], e.faces[1]

                m, n = 0, 0
                for i in range(3):
                    if tri1.verts[i].id != v1.id and tri1.verts[i].id != v2.id:
                        m = i
                    if tri2.verts[i].id != v1.id and tri2.verts[i].id != v2.id:
                        n = i
                v3_1, v3_2 = tri1.verts[m], tri2.verts[n]

                A = self.cot_weight[e.id]
                v_f = A @ ti.Matrix.rows([v1.pos, v2.pos, v3_1.pos, v3_2.pos])
                print(f"Edge {e.id}: {v1.id}--{v2.id}--{v3_1.id}--{v3_2.id}; pos: {v1.pos}, {v2.pos}, {v3_1.pos}, {v3_2.pos}")
                print(f"cotangent weights: {self.cot_weight[e.id]}; voronoi: {e.voronoi}; v_f: {v_f}; norm: {v_f.norm()}")

def main():
    class Soft(SoftBend2D):
        def __init__(self, shape:list, E:float, nu:float, dt:float, density:float, g=-9.8):
            super().__init__(shape, E, nu, dt, density, g)
    
    soft = Soft([0.1, 0.1], 1.e4, 0.4, 0.01, 10e2)
    # soft.preset_gui([0.05, 0.1, 0.3], [0.05, 0.1, 0.], [0., 1., 0.])
    soft.preset_gui([-0.2, 0.05, 0.15], [0.05, 0.1, 0.], [0., 0., 1.])

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    # soft.init_vel()
    # contact_vel_np = np.array([[0., -0.01, 0.00]] * len(soft.contact_particle_list))
    # soft.contact_vel.from_numpy(contact_vel_np)
    # soft.contact_vel.fill(0.)

    # print(f"vfs: {soft.mesh.edges.v_g.to_numpy()}")
    # np.savetxt("lhs.csv", lhs_np, fmt='%f', delimiter=",")
    # np.savetxt("lhs_bend.csv", soft.lhs_bend_ti.to_numpy(), fmt='%f', delimiter=",")
    # exit(0)

    e_border = soft.mesh.edges.border.to_numpy()
    e_curvature_indices = np.where(e_border == 0)[0].tolist()

    for itr in range(100):
        print(f"Time Step: {itr} ======================================")
        soft.substep(itr)
        # print(f"Stretch stress: \n{soft.stretch_stress.to_numpy()}")
        # print(f"RHS bend: \n{soft.rhs_bend.to_numpy().reshape(-1, 3)}")
        # print(f"Edge normal: \n{np.hstack((soft.v_f.to_numpy()[e_curvature_indices], np.linalg.norm(soft.v_f.to_numpy()[e_curvature_indices], axis=1).reshape(-1,1)))}")
        print(f"Stretch energy: {np.sum(soft.stretch_energy.to_numpy()):e}")
        print(f"Bend energy: {np.sum(soft.bend_energy.to_numpy()):e}")
        print(f"Stretch limit energy: {np.sum(soft.stretch_lim_energy.to_numpy()):e}")

        soft.gui_show(True, False, itr)
        # time.sleep(0.1)
        # print(f"Node 21: {soft.node_pos[21]}")

    # print(f"Stretch stress: \n{soft.stretch_stress.to_numpy()}")

    # np.savetxt('rhs.csv', soft.rhs.to_numpy(), fmt='%f', delimiter=',')
    # np.savetxt('rhs_stretch.csv', soft.rhs_stretch.to_numpy(), fmt='%f', delimiter=',')


if __name__ == "__main__":
    main()
