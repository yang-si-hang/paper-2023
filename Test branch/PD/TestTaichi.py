
from collections import defaultdict
import numpy as np
from numba import njit, prange
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, debug=True)


@njit(parallel=True)
def run_numba(x, y):
    for i in prange(16):
        x[i] = i
    for i in prange(4):
        y[i] = i


@ti.data_oriented
class Calc:
    def __init__(self):
        self.x = ti.field(dtype=ti.f32, shape=16)
        self.y = ti.field(dtype=ti.f32, shape=4)
        self.y_list = ti.static([1, 2, 3, 4])

    @ti.kernel
    def run(self):
        for i in range(16):
            self.x[i] = i
        for i in range(4):
            self.y[i] = i

    def run_all(self):
        self.run()
        x_np = self.x.to_numpy(dtype=np.float64)
        y_np = self.y.to_numpy(dtype=np.float64)
        run_numba(x_np, y_np)


a = Calc()
a.run_all()
