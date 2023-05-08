"""
This file mesh the object with uniform node
"""
import taichi as ti
# ti.init(arch=ti.cpu)
import taichi.math as tm
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

global_E = 1e5
global_damp = 0.


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

    # # show the mesh
    # plt.axis('equal')
    # plt.triplot(xx_pad, yy_pad, element)
    # plt.plot(xx_pad, yy_pad, 'o')
    # plt.show()
    # exit(0)

    return node, edge, element


# the object geometry size
LL = 0.1
WW = 0.1
global_size = 0.005

node_np, edge_np, element_np = mesh_object(LL, WW, seed_size=0.005)
# print(edge_np.shape)
# np.savetxt('element.csv', element_np, fmt='%d', delimiter=',')
node_np = np.insert(node_np, 1, 0.*np.ones(node_np.shape[0]), axis=1)
np.savetxt('particle_pos.csv', node_np, fmt='%.5f', delimiter=',')

PARTICLE_NUM = node_np.shape[0]
EDGE_NUM = edge_np.shape[0]
ELEMENT_NUM = element_np.shape[0]

particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
particle_init_pos = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
particle_latest_pos = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
particle_mass = ti.field(dtype=ti.f32, shape=PARTICLE_NUM)
vel = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
force = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
vel_1d = ti.ndarray(ti.f32, 3*PARTICLE_NUM)
force_1d = ti.ndarray(ti.f32, 3*PARTICLE_NUM)
veci_particle = ti.types.vector(PARTICLE_NUM, int)

edge = ti.Vector.field(2, dtype=ti.i32, shape=EDGE_NUM)
edge_stiff = ti.field(dtype=ti.f32, shape=EDGE_NUM)
edge_damp = ti.field(dtype=ti.f32, shape=EDGE_NUM)
rest_edge = ti.field(dtype=ti.f32, shape=EDGE_NUM)

element = ti.Vector.field(3, dtype=ti.i32, shape=ELEMENT_NUM)
rest_ele_size = ti.field(dtype=ti.f32, shape=ELEMENT_NUM)

particle_pos.from_numpy(node_np)
particle_init_pos.from_numpy(node_np)
edge.from_numpy(edge_np)
element.from_numpy(element_np)

particle_show = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
surf_show = ti.field(dtype=ti.i32, shape=int(ELEMENT_NUM*3))
surf_show.from_numpy(element_np.flatten('C'))
edge_show = ti.field(dtype=ti.i32, shape=int(EDGE_NUM*2))
edge_show.from_numpy(edge_np.flatten('C'))

MassBuilder = ti.linalg.SparseMatrixBuilder(3*PARTICLE_NUM, 3*PARTICLE_NUM, max_num_triplets=10000)
DBuiler = ti.linalg.SparseMatrixBuilder(3*PARTICLE_NUM, 3*PARTICLE_NUM, max_num_triplets=10000)
KBuilder = ti.linalg.SparseMatrixBuilder(3*PARTICLE_NUM, 3*PARTICLE_NUM, max_num_triplets=10000)


def fix_particle_No(L: float, W: float, seed_size: float):
    """
    Find the particle No. of fix constraint and grasping constraint
    """
    fix_flag = ti.field(dtype=ti.i32, shape=PARTICLE_NUM)
    grasp_flag = ti.field(dtype=ti.i32, shape=PARTICLE_NUM)

    @ti.kernel
    def cal_fix_constraint(L: float, W: float, seed_size: float):
        EPS = seed_size / 3
        # flag = np.array(PARTICLE_NUM, dtype=int)
        for idx in range(PARTICLE_NUM):
            x_temp = particle_pos[idx].x
            z_temp = particle_pos[idx].z
            # flag_temp = (x_temp > L - EPS or x_temp < 0. + EPS) and (z_temp > W/2 - EPS or z_temp < -W/2 + EPS)
            fix_flag_temp = (x_temp < 0. + EPS)
            grasp_flag_temp = (x_temp > L - EPS) and (z_temp > W/2 -EPS)
            fix_flag[idx] = fix_flag_temp
            grasp_flag[idx] = grasp_flag_temp

    cal_fix_constraint(L, W, seed_size)
    fix_particle_set = set()
    grasp_particle_set = set()
    for i in range(PARTICLE_NUM):
        if fix_flag[i]:
            fix_particle_set.add(i)
        if grasp_flag[i]:
            grasp_particle_set.add(i)
    fix_particle_list = list(fix_particle_set)
    grasp_particle_list = list(grasp_particle_set)

    return fix_particle_list, grasp_particle_list


@ti.kernel
def cal_rest_len():
    for i in range(EDGE_NUM):
        rest_edge[i] = (particle_pos[edge[i][0]] - particle_pos[edge[i][1]]).norm()


def edge_in_tri():
    ele_edge = np.zeros([ELEMENT_NUM, 3], dtype=int)
    for i in range(ELEMENT_NUM):
        for j in range(3):
            edge_temp = np.array(sorted(element_np[i, [j, (j+1)%3]]))
            idx = np.where(np.all(edge_np == edge_temp, axis=1))
            if idx[0].size == 0:
                print('Error for finding edge No.')
            ele_edge[i,j] = idx[0][0]

    return ele_edge


@ti.kernel
def init_stiff_damp(E:float, gamma: float, ele_edge:ti.types.ndarray()):
    for i in range(ELEMENT_NUM):
        for j in ti.static(range(3)):
            edge_stiff[ele_edge[i,j]] += E * rest_ele_size[i] / rest_edge[ele_edge[i,j]]

    for i in range(EDGE_NUM):
        edge_damp[i] = gamma


fix_particle_list, grasp_particle_list = fix_particle_No(LL, WW, global_size)
print('fix constraint particles', fix_particle_list)
print('grasp constraint particles', grasp_particle_list)

cal_rest_len()
ele_edge_np = edge_in_tri()
# np.savetxt('edge_in_element.csv', ele_edge_np, fmt='%d', delimiter=',')


@ti.kernel
def init_mass(density: float):
    for i in range(ELEMENT_NUM):
        idx = tm.ivec3(0., 0., 0.)
        for j in ti.static(range(3)):
            idx[j] = element[i][j]

        total_mas = density * (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
                    particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2
        for j in ti.static(range(3)):
            particle_mass[idx[j]] += total_mas / 3


@ti.kernel
def init_mass_sp(Mass: ti.types.sparse_matrix_builder(), density: float):
    for i in range(ELEMENT_NUM):
        idx = tm.ivec3(0., 0., 0.)
        for j in ti.static(range(3)):
            idx[j] = element[i][j]

        rest_ele_size[i] = (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
                    particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2

        total_mas = density * (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
                    particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2
        for j in ti.static(range(3)):
            particle_mass[idx[j]] += total_mas / 3

            Mass[3*idx[j]+0, 3*idx[j]+0] += total_mas / 3
            Mass[3*idx[j]+1, 3*idx[j]+1] += total_mas / 3
            Mass[3*idx[j]+2, 3*idx[j]+2] += total_mas / 3


init_mass_sp(MassBuilder, 1137.)
M = MassBuilder.build()
# M_np = np.ones((3*PARTICLE_NUM, 3*PARTICLE_NUM))
# for i, j in ti.ndrange(3*PARTICLE_NUM, 3*PARTICLE_NUM):
#     M_np[i,j] = M[i,j]
# np.savetxt('M_np.csv', M_np, delimiter=',')
# MassBuilder.print_triplets()

# np.savetxt('rest_ele_size.csv', rest_ele_size.to_numpy(), fmt='%.8f', delimiter=',')
init_stiff_damp(E=global_E, gamma=global_damp,  ele_edge=ele_edge_np)
# np.savetxt('edge_stiffness.csv', edge_stiff.to_numpy(), fmt='%.8f', delimiter=',')
np.savetxt('edge_damping.csv', edge_damp.to_numpy(), fmt='%.8f', delimiter=',')

jx = ti.Matrix.field(3, 3, dtype=ti.f32, shape=EDGE_NUM)
jv = ti.Matrix.field(3, 3, dtype=ti.f32, shape=EDGE_NUM)
jf = ti.Matrix.field(3, 3, dtype=ti.f32, shape=len(fix_particle_list))
b = ti.ndarray(ti.f32, 3*PARTICLE_NUM)
# exit(0)


# --------------- This code is used in main.py ---------------
@ti.kernel
def copy_to(des: ti.types.ndarray(), source: ti.template()):
    # pad the N*3 ti Vector field to 1D ti array
    for i in range(PARTICLE_NUM):
        des[3*i] = source[i][0]
        des[3*i+1] = source[i][1]
        des[3*i+2] = source[i][2]


if __name__ == '__main__':
    node, edge, element = mesh_object(0.1, 0.1)
