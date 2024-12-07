"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

from Utilize.GenMsh import mesh_obj_triangles


def read_msh(file_path):
    # 考虑从外部导入
    pass

def mesh_obj(obj_shape):
    pass


@ti.data_oriented
class SoftBend2D:
    def __init__(self, shape, E, nu, dt, density, g=9.8):
        self.shape = shape
        # 是否传入的是 .msh 文件(已划分网格)
        if isinstance(self.shape, str):
            node_np, edge_np, element_np = read_msh(self.shape)
        else:
            node_np, edge_np, element_np = mesh_obj_triangles(self.shape)
        self.E, self.nu, self.dt, self.density = E, nu, dt, density
        self.g = g
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = element_np.shape[0]

        self.node_pos = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_new = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)     # local solver
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_vel = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_N)
        self.node_pos_init.from_numpy(.astype(np.float64))
        self.node_pos.from_numpy()

        self.edge = ti.Vector.field(2, dtype=)


    def construct_mass(self):
        for i in range(self.ELEMENT_NUM):
            ia, ib, ic = self.element[i]
            self.node_mass[ia] += self.element_volume[i] * self.rho / 3
            self.node_mass[ib] += self.element_volume[i] * self.rho / 3
            self.node_mass[ic] += self.element_volume[i] * self.rho / 3
