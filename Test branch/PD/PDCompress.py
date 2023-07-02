"""
This file simulates the compression of a soft object by Projective Dynamics.
"""

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)
import taichi.math as tm
import numpy as np
from scipy import sparse
from scipy.spatial import Delaunay
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized


@ti.data_oriented
class SoftObject:
    def __init__(self, shape, seed_size):
        self.shape = shape
        self.seed_size = seed_size
        self.dt = 1./120
        self.rho = 1.145e3
        self.E = 5.e5
        self.nu = 0.4
        self.positional_mass = 1.e9
        self.grasp_mass = 1.e9
        self.solve_iteration = 10
        self.dim = len(shape)
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        node_np, edge_np, element_np = self.mesh_object()
        # node_np = np.insert(node_np, 1, 0.*np.ones(node_np.shape[0]), axis=1)
        self.edge_np = edge_np

        self.PARTICLE_NUM = node_np.shape[0]
        self.EDGE_NUM = edge_np.shape[0]
        self.ELEMENT_NUM = element_np.shape[0]

        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_init_pos = ti.Vector.field(2, dtype=ti.f32, shape=self.PARTICLE_NUM)
        self.node_pos_new = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_init_pos.from_numpy(node_np.astype(np.float32))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_NUM)
        self.edge.from_numpy(edge_np.astype(np.int32))

        # This is only for 2D, should be changed for 3D!!!
        self.element = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.element_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.strain_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.volume_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element.from_numpy(element_np.astype(np.int32))

        self.B = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.A = ti.Matrix.field(4, 6, dtype=ti.f64, shape=self.ELEMENT_NUM*2)
        self.Bp = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_NUM*2)

        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*2)
        self.lhs = ti.field(ti.f64, shape=(2*self.ELEMENT_NUM, 2*self.ELEMENT_NUM))
        self.rhs = ti.field(ti.f64, shape=2*self.ELEMENT_NUM)

        self.construct_B()
        self.construct_volume()
        self.construct_mass()

        self.fix_particle_list = self.fix_particle_No()


        # Print the information
        print('Particle number: ', self.PARTICLE_NUM)
        print('Element number: ', self.ELEMENT_NUM)
        # print('Grasp particle No.: ', self.grasp_particle_list)



    def mesh_object(self):
        if self.dim == 2:
            return self.mesh_object_2d(self.shape, self.seed_size)
        elif self.dim == 3:
            return self.mesh_object_3d()
        else:
            raise ValueError("Only 2D and 3D objects are supported.")


    def mesh_object_2d(self, shape, seed_size):
        L = shape[0]
        W = shape[1]
        LN = int(np.ceil(L / seed_size))
        WN = int(np.ceil(W / seed_size))

        # Generate the nodes' position
        xx, yy = np.meshgrid(np.linspace(0, L, LN), np.linspace(-W / 2, W / 2, WN))
        xx_pad = xx.flatten('C')
        yy_pad = yy.flatten('C')
        node = np.array([xx_pad, yy_pad]).T

        # Generate the elements' index
        tri = Delaunay(node)

        element = tri.simplices

        edge_set = set()
        for simplices in element:
            for i in range(3):
                edge_temp = tuple(sorted(simplices[[i, (i + 1) % 3]]))
                edge_set.add(edge_temp)

        edge = np.array(list(edge_set))

        return node, element, edge


    def mesh_object_3d(self):
        pass


    @ti.kernel
    def construct_B(self):
        for i in range(self.ELEMENT_NUM):
            ia, ib, ic = self.element[i]
            a, b, c = self.node_pos[ia], self.node_pos[ib], self.node_pos[ic]
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.B[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_volume(self):
        for i in range(self.ELEMENT_NUM):
            ia, ib, ic = self.element[i]
            a, b, c = self.node_init_pos[ia], self.node_init_pos[ib], self.node_init_pos[ic]
            element_volume_i = abs((b - a).cross(c - a)) / 2
            self.element_volume[i] = element_volume_i
            self.strain_weight[i] = self.mu * 2 * element_volume_i
            self.volume_weight[i] = self.lam * self.dim * element_volume_i


    @ti.kernel
    def construct_mass(self):
        for i in range(self.ELEMENT_NUM):
            ia, ib, ic = self.element[i]
            self.node_mass[ia] += self.element_volume[i] * self.rho / 3
            self.node_mass[ib] += self.element_volume[i] * self.rho / 3
            self.node_mass[ic] += self.element_volume[i] * self.rho / 3


    @ti.kernel
    def precomputation(self):
        ELEMENT_NUM = self.ELEMENT_NUM
        dim = self.dim

        for i in range(ELEMENT_NUM):
            for d in ti.static(range(2)):
                self.lhs[i*dim + d, i*dim + d] = self.node_mass[i]/self.dt**2

        for i in range(ELEMENT_NUM):
            B_i = self.B[i]
            a = B_i[0,0]
            b = B_i[0,1]
            c = B_i[1,0]
            d = B_i[1,1]

            for t in range(2):
                # X_f*X_g^{-1}=Aq is 2*2, and flatten to 4*1 by row
                self.A[t*ELEMENT_NUM + i][0,0] = -a - c
                self.A[t*ELEMENT_NUM + i][0,2] = a
                self.A[t*ELEMENT_NUM + i][0,4] = c
                self.A[t*ELEMENT_NUM + i][1,0] = -b - d
                self.A[t*ELEMENT_NUM + i][1,2] = b
                self.A[t*ELEMENT_NUM + i][1,4] = d
                self.A[t*ELEMENT_NUM + i][2,1] = -a - c
                self.A[t*ELEMENT_NUM + i][2,3] = a
                self.A[t*ELEMENT_NUM + i][2,5] = c
                self.A[t*ELEMENT_NUM + i][3,1] = -b - d
                self.A[t*ELEMENT_NUM + i][3,3] = b
                self.A[t*ELEMENT_NUM + i][3,5] = d

        for ele_idx in range(ELEMENT_NUM):
            for t in range(2):
                A_i = self.A[t*ELEMENT_NUM + ele_idx]
                ia, ib, ic = self.element[ele_idx]
                ia_x, ia_y = ia*dim, ia*dim+1
                ib_x, ib_y = ib*dim, ib*dim+1
                ic_x, ic_y = ic*dim, ic*dim+1
                q_idx_vec = ti.Vector([ia_x, ia_y, ib_x, ib_y, ic_x, ic_y])
                for A_row_idx, A_col_idx in ti.static(ti.ndrange(6,6)):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    for idx in range(dim**2):
                        weight = 0.
                        if t == 0:
                            weight = self.strain_weight[ele_idx]
                        else:
                            weight = self.volume_weight[ele_idx]
                        self.lhs[lhs_row_idx, lhs_col_idx] += weight * A_i[idx, A_row_idx] * A_i[idx, A_col_idx]

        for i in ti.static(self.fix_particle_list):
            q_i_x_idx = i * dim
            q_i_y_idx = i * dim + 1
            self.lhs[q_i_x_idx, q_i_x_idx] += self.positional_mass
            self.lhs[q_i_y_idx, q_i_y_idx] += self.positional_mass

        # for i in ti.static(self.grasp_particle_list):
        #     q_i_x_idx = i * 2
        #     q_i_y_idx = i * 2 + 1
        #     self.lhs[q_i_x_idx, q_i_x_idx] += self.grasp_mass
        #     self.lhs[q_i_y_idx, q_i_y_idx] += self.grasp_mass


    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        for i in range(self.PARTICLE_NUM):
            idx1, idx2 = dim*i, dim*i+1
            pos = self.node_pos[i]
            vel = self.node_vel[i]
            self.sn[idx1] = pos[0] + dt * vel[0]
            self.sn[idx2] = pos[1] + dt * vel[1]


    @ti.kernel
    def construct_rhs(self):
        dim = self.dim
        for i in range(self.PARTICLE_NUM):
            idx1, idx2 = dim*i, dim*i+1
            self.rhs[idx1] = self.node_mass[i] * self.sn[idx1] / self.dt**2
            self.rhs[idx2] = self.node_mass[i] * self.sn[idx2] / self.dt**2

        for i in range(self.ELEMENT_NUM):
            for t in ti.static(range(2)):
                ia, ib, ic = self.element[i]
                Bp_i = self.Bp[t*self.ELEMENT_NUM + i]
                Bp_i_vec = ti.Vector([Bp_i[0,0], Bp_i[0,1], Bp_i[1,0], Bp_i[1,1]])
                A_i = self.A[t*self.ELEMENT_NUM + i]
                AT_Bp = A_i.transpose() @ Bp_i_vec
                if t == 0:
                    weight = self.strain_weight[i]
                else:
                    weight = self.volume_weight[i]
                AT_Bp *= weight

                q_ia_x_idx = ia * dim
                q_ia_y_idx = ia * dim + 1
                self.rhs[q_ia_x_idx] += AT_Bp[0]
                self.rhs[q_ia_y_idx] += AT_Bp[1]

                q_ib_x_idx = ib * dim
                q_ib_y_idx = ib * dim + 1
                self.rhs[q_ib_x_idx] += AT_Bp[2]
                self.rhs[q_ib_y_idx] += AT_Bp[3]

                q_ic_x_idx = ic * dim
                q_ic_y_idx = ic * dim + 1
                self.rhs[q_ic_x_idx] += AT_Bp[4]
                self.rhs[q_ic_y_idx] += AT_Bp[5]

        for i in self.fix_particle_list:
            pos_init = self.node_init_pos[i]
            q_i_x_idx = i * dim
            q_i_y_idx = i * dim + 1
            self.rhs[q_i_x_idx] = self.positional_mass * pos_init[0] / self.dt**2
            self.rhs[q_i_y_idx] = self.positional_mass * pos_init[1] / self.dt**2


    @ti.kernel
    def warm_up(self):
        """
        Warm start the solver
        """
        dim = self.dim
        for i in range(self.PARTICLE_NUM):
            idx0, idx1 = i*dim, i*dim+1
            self.node_pos_new[i].x = self.sn[idx0]
            self.node_pos_new[i].y = self.sn[idx1]


    @ti.kernel
    def local_solve(self):
        for i in range(self.ELEMENT_NUM):
            ia, ib, ic = self.element[0]
            a, b, c = self.node_pos[ia], self.node_pos[ib], self.node_pos[ic]
            D_i = ti.Matrix.cols([b-a, c-a])
            F_i = ti.cast(D_i @ self.B[i], ti.f64)
            self.F[i] = F_i

            U, sig, V = ti.svd(F_i, ti.f64)
            self.Bp[i] = U @ V.transpose()

            # Solve the volume constraint
            D, max_it, tol = ti.Vector([10., 10.]), 80, 1.e-6
            for it in range(max_it):
                aa, bb = D[0,] + sig[0,0], D[1] + sig[1,1]
                C = aa * bb - 1
                partial_C = ti.Vector([bb, aa])

                D_temp = (partial_C.dot(D)-C) / partial_C.norm()**2 * partial_C
                if (D-D_temp).norm() < tol:
                    break
                D = D_temp

            PP = ti.Matrix.rows([[D[0]+sig[0,0],0.], [0., D[1]+sig[1,1]]])
            self.Bp[self.ELEMENT_NUM + i] = U @ PP @ V.transpose()


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        for i in range(self.PARTICLE_NUM):
            idx0, idx1 = i*self.dim, i*self.dim+1
            self.node_pos_new[i].x = sol[idx0]
            self.node_pos_new[i].y = sol[idx1]


    @ti.kernel
    def update_vel_pos(self):
        for i in range(self.PARTICLE_NUM):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]


    def fix_particle_No(self):
        """
        Find the particle No. of fix constraint
        """
        fix_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)
        L = self.shape[0]
        W = self.shape[1]
        seed_size = self.seed_size

        @ti.kernel
        def cal_fix_constraint(L: float, W: float, seed_size: float):
            EPS = seed_size / 3
            for idx in range(self.PARTICLE_NUM):
                x_temp = self.node_init_pos[idx].x
                z_temp = self.node_init_pos[idx].y  # 2D dimension
                # flag_temp = (x_temp > L - EPS or x_temp < 0. + EPS) and (z_temp > W/2 - EPS or z_temp < -W/2 + EPS)
                fix_flag_temp = (x_temp < 0. + EPS)
                fix_flag[idx] = fix_flag_temp

        cal_fix_constraint(L, W, seed_size)
        fix_particle_set = set()
        for i in range(self.PARTICLE_NUM):
            if fix_flag[i]:
                fix_particle_set.add(i)
        fix_particle_list = list(fix_particle_set)

        return fix_particle_list


    def grasp_particle_No(self):
        """
        Find the particle No. and grasping constraint
        """
        grasp_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)
        L = self.shape[0]
        W = self.shape[1]
        seed_size = self.seed_size

        @ti.kernel
        def cal_grasp_constraint(L: float, W: float, seed_size: float):
            EPS = seed_size / 3
            for idx in range(self.PARTICLE_NUM):
                x_temp = self.node_init_pos[idx].x
                z_temp = self.node_init_pos[idx].y  # 2D dimension
                grasp_flag_temp = (x_temp > L - EPS) and (z_temp > W / 2 - EPS)
                grasp_flag[idx] = grasp_flag_temp

        cal_grasp_constraint(L, W, seed_size)
        grasp_particle_set = set()
        for i in range(self.PARTICLE_NUM):
            if grasp_flag[i]:
                grasp_particle_set.add(i)
        grasp_particle_list = list(grasp_particle_set)
        grasp_idx = grasp_particle_list[0]
        grasp_ele_list = []
        for i in range(self.ELEMENT_NUM):
            ele_temp = self.element[i].to_numpy()
            if grasp_idx in ele_temp:
                grasp_ele_list.append(i)

            return grasp_particle_list, grasp_ele_list


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
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_NUM)
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
        self.node_show.from_numpy(np.insert(self.node_pos.to_numpy(dtype=np.float32), 1, np.zeros(self.PARTICLE_NUM), axis=1))

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
        self.window, self.camera, self.scene = self.gui_set(pos=[0.1, 0.2, 0.], target=[0.1, 0., 0.])
        self.show_preset()


    def substep(self):
        self.construct_sn()
        self.warm_up()
        for itr in ti.static(range(self.solve_iteration)):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0)


    @ti.kernel
    def init_vel(self):
        for i in range(self.PARTICLE_NUM):
            if self.node_init_pos.x > self.shape[0] - self.seed_size/3:
                self.node_vel[i].x = 5.
            else:
                self.node_vel[i].x = 0.


def main():
    soft_obj = SoftObject(shape=[0.1, 0.1], seed_size=0.01)
    soft_obj.preset()
    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    # np.savetxt('node_pos_init.csv', soft_obj.node_init_pos.to_numpy())
    np.savetxt('element.csv', soft_obj.element.to_numpy(), fmt='%d')
    # np.savetxt('strain_weight.csv', soft_obj.strain_weight.to_numpy())
    np.savetxt('volume.csv', soft_obj.element_volume.to_numpy())
    np.savetxt('lhs.txt', lhs_np)
    s_lhs_np = sparse.csr_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    soft_obj.init_vel()

    window = soft_obj.window
    while window.running:
        soft_obj.substep()


if __name__ == '__main__':
    main()