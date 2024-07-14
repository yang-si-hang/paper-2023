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
        