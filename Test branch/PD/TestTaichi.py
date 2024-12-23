
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


z_np = np.ones(2, dtype=bool)


@ti.data_oriented
class Calc:
    def __init__(self):
        self.x = ti.field(dtype=ti.f32, shape=16)
        self.y = ti.field(dtype=ti.f32, shape=4)
        self.y_list = ti.static([1, 2, 3, 4])
        self.z = ti.field(dtype=bool, shape=2)
        self.z.fill(1)

    @ti.kernel
    def run(self):
        for i in range(16):
            self.x[i] = i
        for i in range(4):
            self.y[i] = i
        for i in range(2):
            if self.z[i]:
                print(i)

    def run_all(self):
        self.run()
        x_np = self.x.to_numpy(dtype=np.float64)
        y_np = self.y.to_numpy(dtype=np.float64)
        run_numba(x_np, y_np)


a = Calc()
a.run_all()

import meshzoo

points, cells = meshzoo.rectangle_tri(
    np.linspace(0.0, 1.0, 3),
    np.linspace(0.0, 1.0, 3),
    variant="zigzag",  # or "up", "down", "center"
)
# print(f"Points: \n{points}")
# print(f"Cells: \n{cells}")

points, cells = meshzoo.rectangle_quad(
    np.linspace(0.0, 1.0, 5),
    np.linspace(0.0, 1.0, 5),
    cell_type="quad8",  # or "quad8", "quad9"
)

print(f"Points: \n{points}")
print(f"Cells: \n{cells}")

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
# Plot points
plt.scatter(points[:, 0], points[:, 1], color='blue', label='Nodes')

# Plot cell connections - for quad9, we only connect the corner nodes (0,1,2,3)
for cell in cells:
    # Corner nodes are 0,1,2,3 for quad9
    corner_indices = cell[:4]
    # Connect corner points
    for i in range(4):
        j = (i + 1) % 4  # Connect to next corner point
        plt.plot([points[corner_indices[i]][0], points[corner_indices[j]][0]], 
                 [points[corner_indices[i]][1], points[corner_indices[j]][1]], 
                 'r-', alpha=0.5)

    # Plot mid-side and center nodes with different color
    mid_nodes = cell[4:]  # Nodes 4-8 are mid-side and center nodes
    plt.scatter(points[mid_nodes][:, 0], points[mid_nodes][:, 1], 
               color='green', marker='s', label='Mid nodes' if cell is cells[0] else "")

plt.grid(True)
plt.legend()
plt.axis('equal')
plt.show()