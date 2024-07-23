"""
使用PD仿真1D的绳变形
"""

import numpy as np
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, debug=True)


def generate_node(length, num:int):
    node_np = np.zeros((num, 2), dtype=np.float64)
    node_np[:, 0] = np.linspace(0, length, num)
    return node_np


@ti.data_oriented
class PD1D:
    def __init__(self, length, radius, seed_size:float):
        self.length = length
        self.radius = radius
        self.dt = 1./100
        self.rho = 1.e3
        self.E = 1.e5
        self.mu = 0.3
        self.G = self.E / 2 / (1 + self.mu)
        self.positional_weight = 1.e4
        self.section_area = tm.pi * self.radius ** 2

        self.PARTICLE_NUM:int = np.ceil(length / seed_size).astype(int) + 1
        self.l = length / (self.PARTICLE_NUM - 1)

        node_np = generate_node(self.length, self.PARTICLE_NUM)
        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = 0.
        self.node_pos_init.from_numpy(node_np)
        self.node_pos.from_numpy(node_np)

        self.element_frame = ti.Vector.field(4, dtype=ti.f64, shape=self.PARTICLE_NUM-1)            # 四元数
        self.element_inertia = ti.Vector([0., 0., 0., 0.])
        self.stretch_weight = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM-1)

        self.bend_weight = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM-2)

        self.fix_particle_list = [self.PARTICLE_NUM-1]
        self.contact_particle_list = [0]

        self.construct_mass()


    def construct_mass(self):
        self.node_mass = tm.pi * self.radius ** 2 * self.l * self.rho
        
        J1 = J2 = tm.pi * self.radius ** 4 / 4
        J3 = J1 + J2
        self.element_inertia = self.l * self.rho * ti.Vector([0., J1, J2, J3])

    
    def construct_weight(self):
        self.stretch_weight = self.E * self.section_area * self.l
        self.bend_weight = 2 * self.G * tm.pi * self.radius ** 4 / self.l

    
    @ti.kernel
    def precomputation(self):
        dim = 2
        for q_idx in range(self.PARTICLE_NUM-1):
            self.lhs[q_idx*dim, q_idx*dim] = self.node_mass / self.dt ** 2
            self.lhs[q_idx*dim+1, q_idx*dim+1] = self.node_mass / self.dt ** 2
        
        for d in range(dim+dim):
            self.A_stretch[d] = 1. / self.l

        


def main():



if __name__ == '__main__':
    main()