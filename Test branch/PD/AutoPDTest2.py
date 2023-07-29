"""
This file is used to run auto differentiation test of PD simulation. The sparse matrix is solved
by Taichi.
"""

import taichi as ti
ti.init(arch=ti.cpu, default_fp=ti.f32, debug=True)
import taichi.math as tm
import numpy as np
import time
import csv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


@ti.data_oriented
class PDTest():
    def __init__(self):
        self.dim = 2
        self.dt = 1./60
        self.rho = 1.0
        self.E = 5.e5
        self.nu = 0.4
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.positional_mass = 1.e6
        self.grasp_mass = 1.e6
        self.solve_itr = 10
        self.GRASP_VEL = ti.Vector.field(2, dtype=ti.f64, shape=1)
        self.GRASP_VEL[0] = ti.Vector([0.0, 0.0])

        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=4)
        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=4, needs_grad=True)
        self.node_pos_new = ti.Vector.field(2, dtype=ti.f64, shape=4)
        self.node_mass = ti.field(dtype=ti.f64, shape=4)
        self.node_vel = ti.Vector.field(2, dtype=ti.f64, shape=4)
        self.NODE_NUM = 4

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=5)
        self.EDGE_NUM = 5

        self.element = ti.Vector.field(3, dtype=ti.i32, shape=2)
        self.ele_vol = ti.field(dtype=ti.f64, shape=2)
        self.strain_weight = ti.field(dtype=ti.f64, shape=2)
        self.vol_weight = ti.field(dtype=ti.f64, shape=2)
        self.ELE_NUM = 2

        self.B = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELE_NUM)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELE_NUM)
        self.A = ti.Matrix.field(4, 6, dtype=ti.f64, shape=2*self.ELE_NUM)
        self.Bp = ti.Matrix.field(2, 2, dtype=ti.f64, shape=2*self.ELE_NUM)

        self.sn = ti.field(dtype=ti.f64, shape=self.NODE_NUM*2)
        self.lhs = ti.field(ti.f64, shape=(2*self.NODE_NUM, 2*self.NODE_NUM))
        self.rhs = ti.field(ti.f64, shape=2*self.NODE_NUM)

        # self.lhs_t_K = ti.linalg.SparseMatrixBuilder(2*self.NODE_NUM, 2*self.NODE_NUM,
        #                                              max_num_triplets=100)
        self.lhs_t_builder = ti.linalg.SparseMatrixBuilder(2*self.NODE_NUM, 2*self.NODE_NUM, max_num_triplets=100)
        # self.lhs_t = self.lhs_t_K.build()
        self.rhs_t = ti.ndarray(dtype=ti.f32, shape=2*self.NODE_NUM)
        self.sn_t = ti.ndarray(dtype=ti.f32, shape=2*self.NODE_NUM)

        self.assign_variable()
        self.construct_B()
        self.construct_volume()


    def assign_variable(self):
        node_pos_np = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        node_mass_np = np.array([1.0, 1.0, 1.0, 1.0])
        self.node_pos_init.from_numpy(node_pos_np)
        self.node_pos.from_numpy(node_pos_np)
        self.node_mass.from_numpy(node_mass_np)

        edge_np = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
        self.edge.from_numpy(edge_np)
        self.edge_np = edge_np

        element_np = np.array([[0, 1, 2], [0, 2, 3]])
        self.element.from_numpy(element_np)

        self.fix_node_list = [0, 1]
        self.grasp_node_list = [2]


    @ti.kernel
    def construct_B(self):
        for i in range(self.ELE_NUM):
            ia, ib, ic = self.element[i]
            pa, pb, pc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            B_i_inv = ti.Matrix.cols([pb - pa, pc - pa])
            self.B[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_volume(self):
        for i in range(self.ELE_NUM):
            ia, ib, ic = self.element[i]
            pa, pb, pc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            ele_volume = abs((pb - pa).cross(pc - pa)) / 2.0
            self.ele_vol[i] = ele_volume
            self.strain_weight[i] = self.mu * 2 * ele_volume
            self.vol_weight[i] = self.lam * self.dim * ele_volume


    @ti.kernel
    def precomputation(self, lhs_t:ti.types.sparse_matrix_builder()):
        ELE_NUM = self.ELE_NUM
        dim = self.dim

        for i in range(self.NODE_NUM):
            for d in ti.static(range(2)):
                lhs_t[i*dim + d, i*dim + d] += self.node_mass[i]/self.dt**2

        for i in range(self.ELE_NUM):
            B_i = self.B[i]
            a = B_i[0, 0]
            b = B_i[0, 1]
            c = B_i[1, 0]
            d = B_i[1, 1]

            for t in range(2):
                self.A[t*ELE_NUM + i][0, 0] = -a - c
                self.A[t*ELE_NUM + i][0, 2] = a
                self.A[t*ELE_NUM + i][0, 4] = c
                self.A[t*ELE_NUM + i][1, 0] = -b - d
                self.A[t*ELE_NUM + i][1, 2] = b
                self.A[t*ELE_NUM + i][1, 4] = d
                self.A[t*ELE_NUM + i][2, 1] = -a - c
                self.A[t*ELE_NUM + i][2, 3] = a
                self.A[t*ELE_NUM + i][2, 5] = c
                self.A[t*ELE_NUM + i][3, 1] = -b - d
                self.A[t*ELE_NUM + i][3, 3] = b
                self.A[t*ELE_NUM + i][3, 5] = d

        for ele_idx in range(self.ELE_NUM):
            ia, ib, ic = self.element[ele_idx]
            ia_x, ia_y = ia * dim, ia * dim + 1
            ib_x, ib_y = ib * dim, ib * dim + 1
            ic_x, ic_y = ic * dim, ic * dim + 1
            q_idx_vec = ti.Vector([ia_x, ia_y, ib_x, ib_y, ic_x, ic_y])
            for t in range(2):
                A_i = self.A[t*ELE_NUM + ele_idx]
                # for A_row_idx, A_col_idx in ti.static(ti.ndrange(6,6)):
                for A_row_idx, A_col_idx in ti.ndrange(6, 6):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    matrix_temp = ti.f64(0.)
                    for idx in range(dim**2):
                        weight = ti.f64(0.)
                        if t == 0:
                            weight = self.strain_weight[ele_idx]
                        else:
                            weight = self.vol_weight[ele_idx]
                        matrix_temp += weight * A_i[idx, A_row_idx] * A_i[idx, A_col_idx]
                        # lhs_t[lhs_row_idx, lhs_col_idx] += \
                        #     weight * A_i[idx, A_row_idx] * A_i[idx, A_col_idx]
                    # print('Matrix Index:', lhs_row_idx, lhs_col_idx)
                    lhs_t[lhs_row_idx, lhs_col_idx] += matrix_temp

        print('code running here!')
        for i in ti.static(self.fix_node_list):
            q_i_x_idx = i * dim
            q_i_y_idx = i * dim + 1
            lhs_t[q_i_x_idx, q_i_x_idx] += self.positional_mass
            lhs_t[q_i_y_idx, q_i_y_idx] += self.positional_mass

        for i in ti.static(self.grasp_node_list):
            q_i_x_idx = i * 2
            q_i_y_idx = i * 2 + 1
            lhs_t[q_i_x_idx, q_i_x_idx] += self.grasp_mass
            lhs_t[q_i_y_idx, q_i_y_idx] += self.grasp_mass


    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for i in range(self.NODE_NUM):
            idx1, idx2 = dim*i, dim*i+1
            pos = self.node_pos[i]
            vel = self.node_vel[i]
            self.sn[idx1] = pos[0] + dt * vel[0]
            self.sn[idx2] = pos[1] + dt * vel[1]


    @ti.kernel
    def local_solve(self):
        """
        Minimize the energy function
        """
        for i in range(self.ELE_NUM):
            ia, ib, ic = self.element[i]
            a, b, c = self.node_pos_new[ia], self.node_pos_new[ib], self.node_pos_new[ic]
            D_i = ti.Matrix.cols([b-a, c-a])
            F_i = ti.cast(D_i @ self.B[i], ti.f64)
            self.F[i] = F_i

            U, sig, V = ti.svd(F_i, ti.f64)
            self.Bp[i] = U @ V.transpose()

            # Solve the volume constraint
            D, max_it, tol = ti.Vector([10., 10.]), 80, 1.e-6
            for it in range(max_it):
                aa, bb = D[0] + sig[0,0], D[1] + sig[1,1]
                C = aa * bb - 1
                partial_C = ti.Vector([bb, aa])

                D_temp = (partial_C.dot(D)-C) / partial_C.norm()**2 * partial_C
                D_error = (D-D_temp).norm()
                D = D_temp
                if D_error < tol:
                    break

            PP = ti.Matrix.rows([[D[0]+sig[0,0], 0.], [0., D[1]+sig[1,1]]])
            self.Bp[self.ELE_NUM + i] = U @ PP @ V.transpose()


    @ti.kernel
    def construct_rhs(self, rhs_t:ti.types.ndarray()):
        # self.rhs_t.fill(0.)
        dim = self.dim
        for i in range(self.NODE_NUM):
            idx1, idx2 = dim*i, dim*i+1
            rhs_t[idx1] += self.node_mass[i] * self.sn[idx1] / self.dt**2
            rhs_t[idx2] += self.node_mass[i] * self.sn[idx2] / self.dt**2

        # ti.loop_config(serialize=True)
        for i in range(self.ELE_NUM):
            ia, ib, ic = self.element[i]
            for t in ti.static(range(2)):
                Bp_i = self.Bp[t*self.ELE_NUM + i]
                Bp_i_vec = ti.Vector([Bp_i[0,0], Bp_i[0,1], Bp_i[1,0], Bp_i[1,1]])
                A_i = self.A[t*self.ELE_NUM + i]
                AT_Bp = A_i.transpose() @ Bp_i_vec
                weight = 0.
                if t == 0:
                    weight = self.strain_weight[i]
                else:
                    weight = self.vol_weight[i]
                AT_Bp *= weight

                q_ia_x_idx = ia * dim
                q_ia_y_idx = ia * dim + 1
                rhs_t[q_ia_x_idx] += AT_Bp[0]
                rhs_t[q_ia_y_idx] += AT_Bp[1]

                q_ib_x_idx = ib * dim
                q_ib_y_idx = ib * dim + 1
                rhs_t[q_ib_x_idx] += AT_Bp[2]
                rhs_t[q_ib_y_idx] += AT_Bp[3]

                q_ic_x_idx = ic * dim
                q_ic_y_idx = ic * dim + 1
                rhs_t[q_ic_x_idx] += AT_Bp[4]
                rhs_t[q_ic_y_idx] += AT_Bp[5]

        # The positional mass constraint of the rhs matrix need match the lhs matrix
        for i in ti.static(self.fix_node_list):
            pos_init = self.node_pos_init[i]
            q_i_x_idx = i * dim
            q_i_y_idx = i * dim + 1
            rhs_t[q_i_x_idx] += self.positional_mass * pos_init[0]# / self.dt**2
            rhs_t[q_i_y_idx] += self.positional_mass * pos_init[1]# / self.dt**2

        for i in ti.static(self.grasp_node_list):
            pos_new_i = self.node_pos_new[i]
            q_i_x_idx = i * dim
            q_i_y_idx = i * dim + 1
            rhs_t[q_i_x_idx] += (pos_new_i[0] * self.grasp_mass)
            rhs_t[q_i_y_idx] += (pos_new_i[1] * self.grasp_mass)


    def itration_solve(self):
        pass


    @ti.kernel
    def warm_up(self):
        dim = self.dim
        for i in range(self.NODE_NUM):
            idx0, idx1 = i*dim, i*dim+1
            self.node_pos_new[i].x = self.sn[idx0]
            self.node_pos_new[i].y = self.sn[idx1]


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        for i in range(self.NODE_NUM):
            idx0, idx1 = i*self.dim, i*self.dim+1
            self.node_pos_new[i].x = sol[idx0]
            self.node_pos_new[i].y = sol[idx1]


    @ti.kernel
    def update_vel_pos(self):
        for i in ti.static(self.grasp_node_list):
            self.node_pos_new[i] = self.node_pos[i] + self.GRASP_VEL[0] * self.dt

        for i in range(self.NODE_NUM):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]


    def gui_set(self, pos, target, FOV=60):
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


    def show_preset(self):
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.NODE_NUM)
        self.edge_show = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_NUM)
        self.edge_show.from_numpy(self.edge_np)


    def gui_show(self, window, canvas, scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0):
        """
        Show the GUI
        """
        if SHOW_FLAG is False:
            return
        scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        scene.ambient_light((0.8, 0.8, 0.8))
        # the conversion of object particles, etc. the ggui of the taichi only support float32
        self.node_show.from_numpy(np.insert(self.node_pos.to_numpy(dtype=np.float32), 1,
                                            np.zeros(self.NODE_NUM), axis=1))

        # particle_test = ti.Vector.field(3, dtype=ti.f32, shape=1)
        # particle_test[0] = ti.Vector([0.0, 0., -0.0])

        scene.particles(self.node_show, radius=0.001, color=(0., 0., 0.))
        scene.lines(self.node_show, width=1., indices=self.edge_show, color=(0., 0., 0.))
        # scene.particles(particle_marker, radius=0.001, color=(1., 0., 0.))
        # scene.particles(particle_test, radius=0.005, color=(0., 1., 0.))
        canvas.scene(scene)
        canvas.set_background_color((1.0, 1.0, 1.0))
        # if pos[440].x > 0.144014:
        # window.save_image(f'Figure/{E}-{nu}.png')
        # exit(0)
        # if WRITE_FLAG is True and itr_num % 10 == 0:
        if WRITE_FLAG is True:
            window.save_image(f'FigureWrite/{itr_num}.png')
        window.show()


    def preset(self):
        self.window, self.camera, self.scene = self.gui_set(
            pos=[0.9, 2., 0.5], target=[0.9, 0., 0.5])
        self.canvas = self.window.get_canvas()
        self.show_preset()


    def substep(self):
        self.construct_sn()
        self.warm_up()
        # self.itration_solve()
        for itr in ti.static(range(self.solve_itr)):
            self.local_solve()
            self.rhs_t.fill(0.)
            self.construct_rhs(self.rhs_t)

            x = self.solver.solve(self.rhs_t)
            self.update_pos_new(x)

            # rhs_np = self.rhs.to_numpy()
            # node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            # self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0)


    @ti.kernel
    def init_vel(self):
        for i in range(self.NODE_NUM):
            if self.node_pos_init[i].x > 0.8:
                self.node_vel[i].x = 8.
            else:
                self.node_vel[i].x = 0.


def main():
    class AutoDiffPD(PDTest):
        def __init__(self):
            super().__init__()
            self.my_loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)


        @ti.kernel
        def compute_loss(self):
            desired_pos = ti.Vector([1.2, 0.])
            # now_pos = self.node_pos[3]
            # my_loss = (desired_pos - now_pos).norm()
            self.my_loss[None] = (desired_pos - self.node_pos[3]).norm()


    ti.root.lazy_grad()
    test = AutoDiffPD()
    test.preset()

    test.precomputation(test.lhs_t_builder)

    # lhs_np = test.lhs.to_numpy()
    # # s_lhs_np = sparse.csr_matrix(lhs_np)
    # s_lhs_np = sparse.csc_matrix(lhs_np)
    # test.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    test.lhs_t_builder.print_triplets()
    test.lhs_t = test.lhs_t_builder.build()
    test.solver = ti.linalg.SparseSolver(solver_type="LLT")
    test.solver.analyze_pattern(test.lhs_t)
    test.solver.factorize(test.lhs_t)

    test.init_vel()

    window = test.window
    while window.running:
        test.substep()
        # coss_now = test.compute_loss()
        # print(coss_now)
        # with ti.ad.Tape(loss=test.my_loss):
        #     # test.substep()
        #     test.compute_loss()
        # print('Loss:',test.my_loss[None])
        # print('Gradiant:',test.node_pos.grad.to_numpy())



if __name__ == '__main__':
    main()