"""
使用PD仿真1D的绳变形
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, debug=True)


@ti.data_oriented
class PD1D:
    def __init__(self, length, radius):
        self.length = length
        self.radius = radius
        self.dt = 1./100
        self.rho = 1.e3
        self.positional_weight = 1.e4

        self.PARTICLE_NUM = 

        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)

        self.element = ti.Vector.field(2, dtype=ti.i32, shape=self.PARTICLE_NUM-1)
        self.element_frame = ti.Vector.field(4, dtype=ti.f64, shape=self.PARTICLE_NUM-1)            # 四元数
        


def main():



if __name__ == '__main__':
    main()