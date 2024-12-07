"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
created at 2024-12-08 by hsy
"""

from typing import List
import numpy as np
import numpy.typing as npt
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

from Utilize.GenMsh import mesh_obj_tri
from GGUI import gui_set


def read_msh(file_path):
    # 考虑从外部导入
    pass


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
            pass


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