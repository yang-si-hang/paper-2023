"""
This file is used to implement minimize loss by DiffPD method for a triangle element.
Gradient of Loss w.r.t. p is computed by DiffTaichi.
- The forward simulation doesn't work!
- construct sn has checked.
  Next check local solve
"""

import taichi as ti
ti.init(arch=ti.cpu, debug=True)
import taichi.math as tm
import numpy as np
np.set_printoptions(linewidth=120, suppress=True)
from scipy import sparse
import warnings


weight = 1.
dt = 1./50
node_pos_init = ti.Vector.field(2, dtype=ti.f32, shape=3)
node_pos = ti.Vector.field(2, dtype=ti.f32, shape=3, needs_grad=True)
node_pos_new = ti.Vector.field(2, dtype=ti.f32, shape=3)
node_vel = ti.Vector.field(2, dtype=ti.f32, shape=3)
node_force = ti.Vector.field(2, dtype=ti.f32, shape=3)
node_mass = ti.field(dtype=ti.f32, shape=3)
force_all = ti.Vector.field(2, dtype=ti.f32, shape=())

node_show = ti.Vector.field(3, dtype=ti.f32, shape=3)
edge_show = ti.Vector.field(2, dtype=ti.i32, shape=3)

A = ti.Matrix.field(4, 6, dtype=ti.f32, shape=1)
B = ti.Matrix.field(2, 2, dtype=ti.f32, shape=1)
Bp = ti.Matrix.field(2, 2, dtype=ti.f32, shape=1)

sn = ti.field(ti.f32, shape=6)
Lhs = ti.field(ti.f32, shape=(6, 6))
Rhs = ti.field(ti.f32, shape=6)
Rhs_dA = ti.field(ti.f32, shape=(6, 6))
z = ti.field(ti.f32, shape=6)
L = ti.field(ti.f32, shape=(), needs_grad=True)
dL_f = ti.field(ti.f32, shape=6)


def mesh_init():
    node_pos_init_np = np.array([[0.0, 0.0],
                                 [0.1, 0.0],
                                 [0.0, 0.1]])
    node_pos_init.from_numpy(node_pos_init_np)
    node_vel.fill(0.)
    node_pos.copy_from(node_pos_init)


def construct_mass():
    for i in range(3):
        node_mass[i] = 0.01/3


def precomputation():
    pa, pb, pc = node_pos_init[0], node_pos_init[1], node_pos_init[2]
    B[0] = ti.Matrix.cols([pb - pa, pc - pa]).inverse()
    # print('B:\n', B[0].to_numpy())

    Ba, Bb, Bc, Bd = B[0][0, 0], B[0][0, 1], B[0][1, 0], B[0][1, 1]

    A[0][0, 0] = -Ba - Bc
    A[0][0, 2] = Ba
    A[0][0, 4] = Bc
    A[0][1, 0] = -Bb - Bd
    A[0][1, 2] = Bb
    A[0][1, 4] = Bd
    A[0][2, 1] = -Ba - Bc
    A[0][2, 3] = Ba
    A[0][2, 5] = Bc
    A[0][3, 1] = -Bb - Bd
    A[0][3, 3] = Bb
    A[0][3, 5] = Bd

    A_np = A[0].to_numpy()
    # print('A_np:\n', A_np.transpose() @ A_np)

    Lhs.from_numpy(weight*A_np.transpose() @ A_np)

    for i in range(3):
        Lhs[2*i, 2*i] += node_mass[i]/dt**2
        Lhs[2*i+1, 2*i+1] += node_mass[i]/dt**2

    # print(Lhs.to_numpy())


def construct_sn():
    for i in range(3):
        pos = node_pos[i]
        vel = node_vel[i]
        mass = node_mass[i]
        force = node_force[i]
        sn[2*i] = pos[0] + dt*vel[0] + dt**2*force[0]/mass
        sn[2*i+1] = pos[1] + dt*vel[1] + dt**2*force[1]/mass
    # print('node_pos:', node_pos[0][1])
    # print('sn:', sn[1])
    # print('force add:', dt**2*node_force[0][1]/node_mass[0])


@ti.kernel
def local_solve():
    pa, pb, pc = node_pos_new[0], node_pos_new[1], node_pos_new[2]
    # print('pa:', pa, 'pb:', pb, 'pc:', pc)
    D = ti.Matrix.cols([pb - pa, pc - pa])
    # F = ti.cast(D @ B[0], ti.f32)
    F = D @ B[0]

    U, S, V = ti.svd(F)
    Bp[0] = U @ V.transpose()


def construct_Rhs():
    for i in range(3):
        Rhs[2*i] = node_mass[i]*sn[2*i]/dt**2
        Rhs[2*i+1] = node_mass[i]*sn[2*i+1]/dt**2

    Bp_vec = ti.Vector([Bp[0][0, 0], Bp[0][0, 1], Bp[0][1, 0], Bp[0][1, 1]])
    AT_Bp = A[0].transpose() @ Bp_vec

    AT_Bp *= weight

    for i in range(6):
        Rhs[i] += AT_Bp[i]


def partial_p_test():
    A_np = A[0].to_numpy()
    qa, qb, ac = node_pos[0], node_pos[1], node_pos[2]
    D = ti.Matrix.cols([qb - qa, ac - qa])
    F = D @ B[0]
    # Numpy svd is U@diag(S)@V
    U, S, V = np.linalg.svd(F)

    dT_dF = np.zeros((4, 4))
    for i, j in np.ndindex(2, 2):
        Omega_uv = np.zeros((2, 2))
        Omega_uv[0, 1] = (U[i, 0]*V[1, j] - U[i, 1]*V[0, j])/(S[0] + S[1])
        Omega_uv[1, 0] = -Omega_uv[0, 1]
        dT_df = U @ Omega_uv @ V
        dT_df_vec = dT_df.reshape(-1, order='C')
        dT_dF[:, 2*i+j] = dT_df_vec

    dT = dT_dF @ A_np
    # print('dT:\n', dT)
    AT_dT_dq = A_np.T @ dT
    # print('AT_dT_dq:\n', AT_dT_dq)

    Rhs_dA.fill(0.)
    q_idx_vec = [0, 1, 2, 3, 4, 5]
    for row_idx, col_idx in np.ndindex(6, 6):
        rhs_row_idx = q_idx_vec[row_idx]
        rhs_col_idx = q_idx_vec[col_idx]
        Rhs_dA[rhs_row_idx, rhs_col_idx] += weight * AT_dT_dq[row_idx, col_idx]

    """
    S_eps = ti.abs(S[0] - S[1])
    # Two same singualr values
    if S_eps < 1e-5:
        S_tmp = (S[0] + S[1])/2
        dT_dF = S_tmp * np.eye(4)
    else:
        for m, n in ti.ndrange(2, 2):
            B_tmp = np.array([U[m,1]*V[n,0], -U[m,0]*V[n,1]])
            UV = np.linalg.inv(A_tmp) @ B_tmp
            Omega_U = np.array([[0., -UV[0]], [UV[0], 0.]])
            Omega_V = np.array([[0., -UV[1]], [UV[1], 0.]])

            dT_df = U @ Omega_U @ V.transpose() + U @ Omega_V @ V.transpose()
            dT_df_vec = np.array([dT_df[0,0], dT_df[0,1], dT_df[1,0], dT_df[1,1]])

            dT_dF[:, 2*m+n] = dT_df_vec

    dF_dq = A[0].to_numpy()

    dT_dq = dT_dF @ dF_dq
    AT_dT_dq = A_np.transpose() @ dT_dq

    q_idx_vec = [0, 1, 2, 3, 4, 5]
    for row_idx, col_idx in ti.ndrange(6, 6):
        rhs_row_idx = q_idx_vec[row_idx]
        rhs_col_idx = q_idx_vec[col_idx]
        Rhs_dA[rhs_row_idx, rhs_col_idx] += weight * AT_dT_dq[row_idx, col_idx]
    """


@ti.kernel
def compute_L(desired_area: ti.f32):
    a, b, c = node_pos[0], node_pos[1], node_pos[2]
    ab, ac = b - a, c - a
    area_current  =ti.abs(ab.cross(ac))/2.
    L[None] = (area_current - desired_area)**2


def derivative():
    partial_p_test()
    dA_np = Rhs_dA.to_numpy()
    dq_np = node_pos.grad.to_numpy().reshape(-1, order='C')
    z_np = z.to_numpy()
    for itr in ti.static(range(10)):
        rhs_diff_np = dA_np @ z_np + dq_np
        z_new_np = np.linalg.solve(Lhs.to_numpy(), rhs_diff_np)
        z_np = z_new_np

    dL_f.from_numpy(z_np.transpose())
    print('Node gradient: ', dq_np)
    # print('Delta A gradient:\n', dA_np)
    print('Force gradient: ', dL_f.to_numpy())


def warm_up():
    for i in range(3):
        node_pos_new[i].x = sn[2*i]
        node_pos_new[i].y = sn[2*i+1]


def update_pos_new(sol):
    for i in range(3):
        node_pos_new[i].x = sol[2*i]
        node_pos_new[i].y = sol[2*i+1]


def update_pos():
    for i in range(3):
        node_vel[i] = (node_pos_new[i] - node_pos[i])/dt
        node_pos[i] = node_pos_new[i]


def gui_set(pos, target, FOV=60):
    # init the window, canvas, scene and camerea
    window = ti.ui.Window("Projective Dynamics", (1080, 720), vsync=True)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initialize camera position
    camera.position(pos[0], pos[1], pos[2])
    camera.lookat(target[0], target[1], target[2])
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    # 设置相机的向上轴的方向，在相机模型中是-Y轴
    camera.up(0., 0., -1.)
    camera.z_near(0.01)
    camera.fov(FOV)

    # set the camera, you can move around by pressing 'wasdeq'
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # set the light
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    # scene.point_light(pos=(0.01, 0, 3), color=(1., 1., 1.))
    scene.ambient_light((1., 1., 1.))
    return window, camera, scene


def show_preset():
    edge_np = np.array([[0, 1], [1, 2], [2, 0]])
    edge_show.from_numpy(edge_np)


def gui_show(window, canvas, scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0):
    """
    Show the GUI
    """
    if SHOW_FLAG is False:
        return
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    scene.ambient_light((0.8, 0.8, 0.8))
    # the conversion of object particles, etc. the ggui of the taichi only support float32
    node_show.from_numpy(np.insert(node_pos.to_numpy(dtype=np.float32), 1, np.zeros(3), axis=1))

    # particle_test = ti.Vector.field(3, dtype=ti.f32, shape=1)
    # particle_test[0] = ti.Vector([0.0, 0., -0.0])

    scene.particles(node_show, radius=0.001, color=(0., 0., 0.))
    scene.lines(node_show, width=1., indices=edge_show, color=(0., 0., 0.))
    # scene.particles(particle_marker, radius=0.001, color=(1., 0., 0.))
    # scene.particles(particle_test, radius=0.005, color=(0., 1., 0.))
    canvas.scene(scene)
    canvas.set_background_color((1.0, 1.0, 1.0))
    if WRITE_FLAG is True:
        window.save_image(f'FigureWrite/{itr_num}.png')
    window.show()


def preset():
    window, camera, scene = gui_set(pos=[0.1, 0.2, 0.], target=[0.1, 0., 0.])
    canvas = window.get_canvas()
    show_preset()
    return window, canvas, scene


def substep(pre_fact_Lhs_solve):
    construct_sn()
    warm_up()
    for itr in range(10):
        local_solve()
        construct_Rhs()
        node_pos_new_np = pre_fact_Lhs_solve(Rhs.to_numpy())
        update_pos_new(node_pos_new_np)

    update_pos()
    with ti.ad.Tape(loss=L):
        compute_L(0.01)
    print('Loss:', L)

    # print('Node gradient', node_pos.grad.to_numpy())
    derivative()
    update(1000.)

    # node_force[0] = ti.Vector([0., -9.8])

    # partial_p_test()
    # dA_np = Rhs_dA.to_numpy()
    # dL_np = dL.to_numpy()
    # z_np = z.to_numpy()
    # with warnings.catch_warnings(record=True):
    #     warnings.simplefilter('error')
    #     try:
    #         for itr in ti.static(range(10)):
    #             rhs_diff_np = dA_np @ z_np + dL_np
    #             z_new_np = pre_fact_Lhs_solve(rhs_diff_np)
    #             z_np = z_new_np
    #     except RuntimeWarning as e:
    #         print(z_np)
    # print('Gradient p:', z_np)


# @ti.kernel
def update(learning_rate: ti.f32):
    force_all_tmp = ti.Vector([0., 0.])
    for i in range(3):
        node_force[i][0] -= learning_rate*dL_f[2*i]
        node_force[i][1] -= learning_rate*dL_f[2*i+1]
        force_all_tmp -= node_force[i]
    force_all[None] = force_all_tmp/3.
    # print('force all: ', force_all_tmp)


def init_vel():
    node_force[0] = ti.Vector([0., -9.8])


def main():
    window, canvas, scene = preset()
    mesh_init()
    construct_mass()
    # init_vel()
    precomputation()
    Lhs_np = Lhs.to_numpy()
    s_Lhs_np = sparse.csr_matrix(Lhs_np)
    pre_fact_Lhs_solve = sparse.linalg.factorized(s_Lhs_np)

    for i in range(1000):
        substep(pre_fact_Lhs_solve)
        print('Node postions:', node_pos.to_numpy())
        # print('Gravity position: ', node_pos.to_numpy().sum(axis=0))
        print('Node force:\n', node_force.to_numpy())
        # print('All force: ', node_force.to_numpy().sum(axis=0))
        gui_show(window, canvas, scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0)


if __name__ == '__main__':
    main()