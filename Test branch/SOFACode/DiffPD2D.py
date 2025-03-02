""" 用于SOFA仿真环境控制的DiffPD 2D模型, 同时也是2D场景
created at 2025-03-02 by hsy
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

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import mesh_obj_tri, write_obj
from Utilize.GuiTaichi import gui_set
from Utilize.MathTaichi import cotangent_ti


def read_msh(file:str): # 预留接口
    pass

@ti.data_oriented
class SoftObject2D:
    def __init__(self, shape, fix:List[int], contact:List[int], 
                 E:float, nu:float, dt:float, density:float, g=-9.8, **kwargs):
        self.shape = shape
        # 是否传入的是 .msh 文件(已划分网格)
        if isinstance(self.shape, str):
            node_np, edge_np, ele_np = read_msh(self.shape)
        else:
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.01)
            # node_np = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))         # di: N*3
            np.savetxt("node_np.csv", node_np, fmt='%f', delimiter=",")
            np.savetxt("edge_np.csv", edge_np, fmt='%d', delimiter=",")
            np.savetxt("ele_np.csv", ele_np, fmt='%d', delimiter=",")

        obj_file:str = "Mesh/shape.obj"
        write_obj(obj_file, node_np, ele_np)

        self.solve_itr:int = 10
        self.strain_lim_rate:float = 0.1            # Strain limit rate
        self.E, self.nu, self.dt, self.density, self.g = E, nu, dt, density, g
        self.dim = 2
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
       
        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
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
        self.Bp_shear = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)       # stretch part
        # self.Bp_shear_lim = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)   # strain-limit part
        self.stretch_stress = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        # self.stretch_lim_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)

        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.rhs_stretch = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*2)
        self.pre_fact_lhs_solve = None

        self.fix_particle_list = fix
        self.contact_particle_list = contact
        self.FIX_N = len(self.fix_particle_list)
        self.CON_N = len(self.contact_particle_list)
        self.fix_particle_ti = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)
        self.contact_particle_ti.from_numpy(np.array(self.contact_particle_list).astype(np.int32))
        self.contact_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.CON_N)
        self.contact_vel.fill(0.)

        self.construct_mass()
        self.construct_cotangent()
        self.construct_Xg_inv()
        self.positional_weight = 1.e3 * self.node_mass_sum[None] / self.PARTICLE_N / self.dt**2
        
        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")
        print(f"Positional weight: {self.positional_weight}")


    @ti.kernel
    def construct_mass(self):
        for f_i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[f_i]
            qa, qb, qc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            ele_volume_tmp = 0.5 * ((qb - qa).cross(qc - qa)).norm()
            self.ele_volume[f_i] = ele_volume_tmp
            self.stretch_weight[f_i] = 2 * self.mu * self.ele_volume[f_id]

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
            self.sn[q_i*2] = self.node_pos[q_i].x + self.contact_vel[idx].x * dt
            self.sn[q_i*2 + 1] = self.node_pos[q_i].y + self.contact_vel[idx].y * dt


    @ti.kernel
    def warm_start(self):
        for q_i in range(self.PARTICLE_N):
            self.node_pos_new[q_i].x = self.sn[q_i*2]
            self.node_pos_new[q_i].y = self.sn[q_i*2+1]


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

            U, sig, V = ti.svd(F_i)
            self.stretch_stress[f_i] = sig
            # print(f"U:{U:e}; sig:{sig:e}; V:{V:e}")
            self.Bp_shear[f_i] = U @ ti.Matrix([[1., 0], [0., 1]], ti.f64) @ V.transpose()
            self.stretch_energy[f_i] = 0.5 * self.stretch_weight[f_i] * ((sig[0]-1.)**2 + (sig[1]-1.)**2)
            # print(f"Bp_shear:{self.Bp_shear[f_i]:e}")

        for f_i in range(self.ELEMENT_N):
            Bp_shear_i = self.Bp_shear[f_i]
            F_AT = self.F_A[f_i].transpose()

            # Bp_shear_i做transpose，因为AT需要与Bp的x，y，z分别矩阵乘法
            F_ATBp = F_AT @ Bp_shear_i.transpose() * self.stretch_weight[f_i]

            for q_i, dim_idx in ti.ndrange(3, 2):
                q_idx = self.ele[f_i][q_i]
                self.rhs[q_idx*3+dim_idx] += F_ATBp[q_i, dim_idx]
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
            self.node_vel[q_i] = ti.Vector([0., 0., 0.], dt=ti.f64)

        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.node_vel[q_idx] = ti.Vector([0., 0., 0.], dt=ti.f64)


    def substep(self, step_num:int):
        self.construct_sn()
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            # print(f"Iteration: {itr} ------------------------------------")
            self.local_solve()
            rhs_np = self.rhs.to_numpy()
            # print(f"Rhs:\n{self.rhs_stretch.to_numpy().reshape(-1, 2)}")
            # Split rhs_np into x,y,z components
            rhs_np_x = rhs_np[0::2]
            rhs_np_y = rhs_np[1::2]

            node_pos_new_np_x = self.pre_fact_lhs_solve(rhs_np_x)
            node_pos_new_np_y = self.pre_fact_lhs_solve(rhs_np_y)

            self.update_pos_new(node_pos_new_np_x, node_pos_new_np_y)
            # print(f"Node pos new:\n", self.node_pos_new.to_numpy().reshape(-1, 2))
        
        self.update_vel_pos()


def main():
    class Soft(SoftObject2D):
        def __init__(self, shape, fix:List[int], contact:List[int], 
                     E:float, nu:float, dt:float, density:float, g=0, **kwargs):
            super().__init__(shape, fix, contact, E, nu, dt, density, g, **kwargs)

    soft = Soft()


if __name__ == "__main__":
    main()