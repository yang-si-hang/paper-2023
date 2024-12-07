"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

from Utilize.GenMsh import mesh_obj_tri


def read_msh(file_path):
    # 考虑从外部导入
    pass


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

        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.node_pos = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_vel = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init.from_numpy(node_np.astype(np.float64))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge.from_numpy(edge_np.astype(np.int32))

        self.ele = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)
        self.ele_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.ele.from_numpy(ele_np.astype(np.int32))

        self.bend_weight = ???

        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")



    @ti.kernel
    def construct_mass(self):
        for i in range(self.ELEMENT_N):
            ia, ib, ic = self.ele[i]
            self.node_mass[ia] += self.ele_volume[i] * self.density / 3
            self.node_mass[ib] += self.ele_volume[i] * self.density / 3
            self.node_mass[ic] += self.ele_volume[i] * self.density / 3


    @ti.kernel
    def precomputation(self):
        dim = self.dim

        for q_i in range(self.PARTICLE_N):