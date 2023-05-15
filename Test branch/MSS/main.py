"""
This file simulate the object deformation with Mass Spring Method.
The solver uses Sparse Matrix Solver in taichi.
"""

import taichi as ti
ti.init(arch=ti.gpu, debug=True, default_fp=ti.f32)
import numpy as np

from MeshLoad import *


"""-----------Initialization-----------"""
constraint_stiff = 5.e6
Gravity = ti.Vector([0., -9.8, 0.])
GRASP_VEL = ti.Vector([0.005, 0., 0.005])


@ti.kernel
def compute_force():
    # Gravity for test!
    for i in range(PARTICLE_NUM):
        force[i] = ti.Vector([0., 0., 0.])
        # force[i] += Gravity * particle_mass[i]
        # print(particle_mass[i])

    # f = -k * (||x_ij||-r_ij) * x_ij / ||x_ij||
    for i in range(EDGE_NUM):
        idx1, idx2 = edge[i][0], edge[i][1]
        delta_dist = (particle_pos[idx1] - particle_pos[idx2]).norm() - rest_edge[i]
        direction = (particle_pos[idx1] - particle_pos[idx2]).normalized()
        force[idx1] += - edge_stiff[i] * delta_dist * direction
        force[idx2] += edge_stiff[i] * delta_dist * direction

    # fix particles' constraint
    # for i in ti.static(fix_particle_list):
    #     force[i] += constraint_stiff * (particle_init_pos[i] - particle_pos[i])


@ti.kernel
def update_pos(dv:ti.types.ndarray(), dt:float):
    for i in range(PARTICLE_NUM):
        particle_latest_pos[i] = particle_pos[i]
        vel[i] += ti.Vector([dv[3*i], dv[3*i+1], dv[3*i+2]])
        particle_pos[i] += dt * vel[i]
        particle_pos[i].y = 0.

    # fix constraint
    for i in ti.static(fix_particle_list):
        particle_pos[i] = particle_init_pos[i]

    for i in ti.static(grasp_particle_list):
        particle_pos[i] = particle_latest_pos[i] + dt*GRASP_VEL


@ti.kernel
def compute_Jacobian():
    """
    Compute the Jacobian matrix of the edge force, then assemble to the particle.
    Jx = -k * (I - r_ij/||x_ij|| * (I - x_ij*x_ij/||x_ij||^2))
    Jv = -d * I
    \parital f / \parital x_i = \parital f / \parital x_j = Jx
    """
    I = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    for i in range(EDGE_NUM):
        idx1, idx2 = edge[i][0], edge[i][1]
        dist = particle_pos[idx1] - particle_pos[idx2]
        jx[i] = - edge_stiff[i] * (I - rest_edge[i]/dist.norm() *
                                   (I - dist.outer_product(dist)/dist.norm()**2))
        jv[i] = - edge_damp[i] * I

    # fix particles' constraint
    # for i in ti.static(range(len(fix_particle_list))):
    #     jf[i] = ti.Matrix([[-constraint_stiff, 0, 0], [0, -constraint_stiff, 0], [0, 0, -constraint_stiff]])


@ti.kernel
def assemble_K(K:ti.types.sparse_matrix_builder()):
    """
    \partial f_i / \partial x_j = - \partial f_j / \partial x_j
    """
    for i in range(EDGE_NUM):
        idx1, idx2 = edge[i][0], edge[i][1]
        for m, n in ti.static(ti.ndrange(3, 3)):
            K[3*idx1+m, 3*idx1+n] += jx[i][m, n]
            K[3*idx2+m, 3*idx2+n] += jx[i][m, n]
            K[3*idx1+m, 3*idx2+n] += -jx[i][m, n]
            K[3*idx2+m, 3*idx1+n] += -jx[i][m, n]

    # fix constraint
    # j = int(0)
    # ti.loop_config(serialize=True)
    # for i in ti.static(fix_particle_list):
    #     for m, n in ti.static(ti.ndrange(3, 3)):
    #         K[3*i+m, 3*i+n] += jf[j][m, n]
    #     j += 1


@ti.kernel
def assemble_D(D:ti.types.sparse_matrix_builder()):
    for i in range(EDGE_NUM):
        idx1, idx2 = edge[i][0], edge[i][1]
        for m, n in ti.static(ti.ndrange(3, 3)):
            D[3*idx1+m, 3*idx1+n] += jv[i][m, n]
            D[3*idx2+m, 3*idx2+n] += jv[i][m, n]
            D[3*idx1+m, 3*idx2+n] += -jv[i][m, n]
            D[3*idx2+m, 3*idx1+n] += -jv[i][m, n]


@ti.kernel
def compute_b(b:ti.types.ndarray(), f:ti.types.ndarray(), Kv:ti.types.ndarray(), dt:float):
    for i in range(3*PARTICLE_NUM):
        b[i] = (f[i] + Kv[i]*dt)*dt
        # print(f[i])


def substep(dt:float=0.01):
    compute_force()
    compute_Jacobian()
    assemble_D(DBuiler)
    assemble_K(KBuilder)

    D = DBuiler.build()
    K = KBuilder.build()
    # DBuiler.print_triplets()
    # KBuilder.print_triplets()

    A = M - dt*D - dt**2*K

    copy_to(vel_1d, vel)
    copy_to(force_1d, force)

    Kv = K @ vel_1d
    compute_b(b,force_1d, Kv, dt)

    solver = ti.linalg.SparseSolver(solver_type='LDLT')
    solver.analyze_pattern(A)
    solver.factorize(A)
    dv = solver.solve(b)
    # print(dv[0])

    update_pos(dv, dt)


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
    if particle_pos[399].x > 0.14:
        window.save_image(f'Figure/{global_E}.png')
        exit(0)
    window.show()


def main():
    window, camera, scene = gui_set(pos=[0.04, 0.3, 0.], target=[0.05, 0., 0.])
    canvas = window.get_canvas()
    # print('code running here!')

    while window.running:
        # show the GUI
        gui_show(window, canvas, scene)
        substep(0.001)
        print('particle pos', particle_pos[399])
        # print('force', force[0])


if __name__ == '__main__':
    main()