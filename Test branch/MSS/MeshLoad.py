"""
This file mesh the object with uniform node
"""
import taichi as ti
ti.init(arch=ti.cpu)
import taichi.math as tm
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


def mesh_object(L, W, seed_size=0.005):
    LN = int(np.ceil(L / seed_size))
    WN = int(np.ceil(W / seed_size))

    # Generate the node
    xx, yy = np.meshgrid(np.linspace(0, L, LN), np.linspace(-W / 2, W / 2, WN))
    xx_pad = xx.flatten('C')
    yy_pad = yy.flatten('C')
    node = np.array([xx_pad, yy_pad]).T

    # Generate the element
    tri = Delaunay(node)

    element = tri.simplices

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_temp = tuple(sorted(simplices[[i, (i+1)%3]]))
            edge_set.add(edge_temp)

    edge = np.array(list(edge_set))

    # show the mesh
    # plt.triplot(xx_pad, yy_pad, element)
    # plt.plot(xx_pad, yy_pad, 'o')
    # plt.show()

    return node, edge, element


node_np, edge_np, element_np = mesh_object(0.1, 0.1, seed_size=0.005)
node_np = np.insert(node_np, 1, 0.*np.ones(node_np.shape[0]), axis=1)

PARTICLE_NUM = node_np.shape[0]
EDGE_NUM = edge_np.shape[0]
ELEMENT_NUM = element_np.shape[0]

particle = ti.Vector.field(3, dtype=ti.f64, shape=PARTICLE_NUM)
particle_mass = ti.field(dtype=ti.f64, shape=PARTICLE_NUM)
vel = ti.Vector.field(3, dtype=ti.f64, shape=PARTICLE_NUM)
force = ti.Vector.field(3, dtype=ti.f64, shape=PARTICLE_NUM)
edge = ti.Vector.field(2, dtype=ti.i32, shape=EDGE_NUM)
rest_edge = ti.field(dtype=ti.f64, shape=EDGE_NUM)
element = ti.Vector.field(3, dtype=ti.i32, shape=ELEMENT_NUM)

particle.from_numpy(node_np)
edge.from_numpy(edge_np)
element.from_numpy(element_np)

particle_show = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
surf_show = ti.field(dtype=ti.i32, shape=int(ELEMENT_NUM*3))
surf_show.from_numpy(element_np.flatten('C'))


@ti.kernel
def cal_rest_len():
    for i in range(EDGE_NUM):
        rest_edge[i] = (particle[edge[i][0]] - particle[edge[i][1]]).norm()


@ti.kernel
def init_mass(density: float=1000.):
    for i in range(ELEMENT_NUM):
        idx = tm.ivec3(0., 0., 0.)
        for j in ti.static(range(3)):
            idx[j] = element[i][j]

        total_mas = density * (particle[idx[0]] - particle[idx[1]]).cross(particle[idx[0]] -
                    particle[idx[2]]).norm() / 2
        for j in ti.static(range(3)):
            particle_mass[idx[j]] += total_mas / 3


init_mass(1137.)




if __name__ == '__main__':
    node, edge, element = mesh_object(0.1, 0.1)
