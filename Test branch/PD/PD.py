

import taichi as ti
ti.init(arch=ti.gpu, debug=True)
import taichi.math as tm
import numpy as np
from scipy.spatial import Delaunay


"""-----------Initialization-----------"""
dim = 2         # dimension
dt = 1./60
E = 5.e2
Poisson = 0.25
mu, labda = E / (2 * (1 + Poisson)), E * Poisson / ((1 + Poisson) * (1 - 2 * Poisson))
# 需要完整的推导过程
w_strain = 2 * mu
w_volume = dim * labda


def mesh_object(L, W, seed_size=0.005):
    """
    Generate the mesh of the 2D object
    :param seed_size: The size of the element
    """
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
# add the Y coordinate
node_np = np.insert(node_np, 1, 0.*np.ones(node_np.shape[0]), axis=1)

PARTICLE_NUM = node_np.shape[0]
EDGE_NUM = edge_np.shape[0]
ELEMENT_NUM = element_np.shape[0]

particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
particle_init_pos = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
particle_mass = ti.field(dtype=ti.f32, shape=PARTICLE_NUM)
vel = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)

edge = ti.Vector.field(2, dtype=ti.i32, shape=EDGE_NUM)
rest_edge = ti.field(dtype=ti.f32, shape=EDGE_NUM)

element = ti.Vector.field(3, dtype=ti.i32, shape=ELEMENT_NUM)
rest_ele_size = ti.field(dtype=ti.f32, shape=ELEMENT_NUM)

particle_pos.from_numpy(node_np)
particle_init_pos.from_numpy(node_np)
edge.from_numpy(edge_np)
element.from_numpy(element_np)

particle_show = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
lhs_matrix = ti.field(ti.f32, shape=(2*PARTICLE_NUM, 2*PARTICLE_NUM))

MassBuilder = ti.linalg.SparseMatrixBuilder(3*PARTICLE_NUM, 3*PARTICLE_NUM, max_num_triplets=10000)

# Temporarily consider 2D problem
Xf = ti.Matrix.field(2, 2, ti.f32, shape=ELEMENT_NUM)
Xg_inv = ti.Matrix.field(2, 2, ti.f32, shape=ELEMENT_NUM)
A_strain = ti.Matrix.field(4, 6, ti.f32, shape=ELEMENT_NUM)
A_volume = ti.Matrix.field(4, 6, ti.f32, shape=ELEMENT_NUM)


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


fix_particle_list, grasp_particle_list = fix_particle_No(LL, WW, global_size)
print('fix constraint particles', fix_particle_list)
print('grasp constraint particles', grasp_particle_list)

cal_rest_len()
ele_edge_np = edge_in_tri()
# np.savetxt('edge_in_element.csv', ele_edge_np, fmt='%d', delimiter=',')


# @ti.kernel
# def init_mass_sp(Mass: ti.types.sparse_matrix_builder(), density: float):
#     for i in range(ELEMENT_NUM):
#         idx = tm.ivec3(0., 0., 0.)
#         for j in ti.static(range(3)):
#             idx[j] = element[i][j]
#
#         Xg_temp = ti.Matrix.cols([particle_pos[idx[1]] - particle_pos[idx[0]],
#                               particle_pos[idx[2]] - particle_pos[idx[0]]])
#         Xg_inv[i] = Xg_temp.inverse()
#
#         rest_ele_size[i] = (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
#                     particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2
#
#         total_mas = density * (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
#                     particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2
#         for j in ti.static(range(3)):
#             particle_mass[idx[j]] += total_mas / 3
#
#             Mass[3*idx[j]+0, 3*idx[j]+0] += total_mas / 3
#             Mass[3*idx[j]+1, 3*idx[j]+1] += total_mas / 3
#             Mass[3*idx[j]+2, 3*idx[j]+2] += total_mas / 3
#
#
# init_mass_sp(MassBuilder, 1137.)
# M = MassBuilder.build()


@ti.kernel
def init_mass(density: float):
    for i in range(ELEMENT_NUM):
        idx = tm.ivec3(0., 0., 0.)
        for j in ti.static(range(3)):
            idx[j] = element[i][j]

        Xg_temp = ti.Matrix.cols([particle_pos[idx[1]] - particle_pos[idx[0]],
                              particle_pos[idx[2]] - particle_pos[idx[0]]])
        Xg_inv[i] = Xg_temp.inverse()

        rest_ele_size[i] = (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
                    particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2

        total_mas = density * (particle_pos[idx[0]] - particle_pos[idx[1]]).cross(
                    particle_pos[idx[0]] - particle_pos[idx[2]]).norm() / 2
        for j in ti.static(range(3)):
            particle_mass[idx[j]] += total_mas / 3


@ti.kernel
def precompute():
    # Strain constraint
    for i in range(ELEMENT_NUM):
        temp = Xg_inv[i]
        a = temp[0, 0]
        b = temp[0, 1]
        c = temp[1, 0]
        d = temp[1, 1]
        A_strain[i] = ti.Matrix([[-a-c, 0, a, 0, c, 0],
                                 [0, -a-c, 0, a, 0, c],
                                 [-b-d, 0, b, 0, d, 0],
                                 [0, -b-d, 0, b, 0, d]])

    # Area constraint
    for i in range(ELEMENT_NUM):
        A_volume[i] = A_strain[i]

    # M/h^2
    for i in range(PARTICLE_NUM):
        for d in ti.static(range(2)):
            lhs_matrix[i*2+d, i*2+d] = particle_mass[i] / dt**2

    for ele_idx in range(ELEMENT_NUM):
        idx0, idx1, idx2 = element[ele_idx]
        idx0_x, idx0_y = idx0*2, idx0*2+1
        idx1_x, idx1_y = idx1*2, idx1*2+1
        idx2_x, idx2_y = idx2*2, idx2*2+1
        # vectorizaztion of particle's position
        q_idx_vec = ti.Vector([idx0_x, idx0_y, idx1_x, idx1_y, idx2_x, idx2_y])
        ATA_strain = A_strain[ele_idx].transpose() @ A_strain[ele_idx]
        ATA_volume = A_volume[ele_idx].transpose() @ A_volume[ele_idx]
        for row in ti.static(range(6)):
            for col in ti.static(range(6)):
                lhs_row = q_idx_vec[row]
                lhs_col = q_idx_vec[col]
                lhs_matrix[lhs_row, lhs_col] += \
                    w_strain * rest_ele_size[ele_idx] * ATA_strain[row, col] + \
                    w_volume * rest_ele_size[ele_idx] * ATA_volume[row, col]



"""------------The GUI setting------------"""
def gui_set(pos, target, FOV=60):
    # init the window, canvas, scene and camerea
    window = ti.ui.Window("MSS", (1080, 720), vsync=True)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initialize camera position
    camera.position(pos[0], pos[1], pos[2])
    camera.lookat(target[0], target[1], target[2])
    camera.fov(FOV)

    # set the camera, you can move around by pressing 'wasdeq'
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # set the light
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    scene.point_light(pos=(0.01, 0, 3), color=(1., 1., 1.))
    scene.ambient_light((0.7, 0.7, 0.7))
    return window, camera, scene


def gui_show(window, canvas, scene, SHOW_FLAG=True):
    """
    Show the GUI
    """
    if SHOW_FLAG is False:
        return
    # the conversion of object particles, etc. the ggui of the taichi only support float32
    particle_show.from_numpy(particle_pos.to_numpy(dtype=np.float32))

    # particle_test = ti.Vector.field(3, dtype=ti.f32, shape=1)
    # particle_test[0] = ti.Vector([0.0, 0., -0.0])

    # scene.mesh(particle_show, indices=surf_show, color=(1, 1, 0))
    scene.particles(particle_show, radius=0.001, color=(1., 1., 1.))
    # scene.lines(particle_show, width=0.0005, indices=edge_show, color=(1. ,1. ,1.))
    # scene.particles(particle_test, radius=0.005, color=(0., 1., 0.))
    canvas.scene(scene)
    # if particle_pos[399].x > 0.14:
    #     window.save_image(f'Figure/1.png')
    #     exit(0)
    window.show()
"""------------The GUI setting------------"""


def main():
    window, camera, scene = gui_set(pos=[0.04, 0.3, 0.], target=[0.05, 0., 0.])
    canvas = window.get_canvas()

    while window.running:
        gui_show(window, canvas, scene)



if __name__ == '__main__':
    main()