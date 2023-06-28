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
        self.dim = len(shape)

        node_np, edge_np, element_np = self.mesh_object()
        node_np = np.insert(node_np, 1, 0.*np.ones(node_np.shape[0]), axis=1)
        self.edge_np = edge_np

        self.PARTICLE_NUM = node_np.shape[0]
        self.EDGE_NUM = edge_np.shape[0]
        self.ELEMENT_NUM = element_np.shape[0]

        self.node_pos = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_init_pos = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_NUM)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM)

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_NUM)

        # This is only for 2D, should be changed for 3D!!!
        self.element = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_NUM)

        self.B = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_NUM)

        self.construct_B()


        # Print the information
        print('Particle number: ', self.PARTICLE_NUM)



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
            B_i_inv = ti.Matrix.cols([a - c, b - c])
            self.B[i] = B_i_inv.inverse()


    @ti.kernel
    def precomputation(self):



    def fix_particle_No(self):
        """
        Find the particle No. of fix constraint and grasping constraint
        """
        fix_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)
        grasp_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)
        L = self.shape[0]
        W = self.shape[1]
        seed_size = self.seed_size

        @ti.kernel
        def cal_fix_constraint(L: float, W: float, seed_size: float):
            EPS = seed_size / 3
            # flag = np.array(PARTICLE_NUM, dtype=int)
            for idx in range(self.PARTICLE_NUM):
                x_temp = self.node_init_pos[idx].x
                z_temp = self.node_init_pos[idx].y  # 2D dimension
                # flag_temp = (x_temp > L - EPS or x_temp < 0. + EPS) and (z_temp > W/2 - EPS or z_temp < -W/2 + EPS)
                fix_flag_temp = (x_temp < 0. + EPS)
                grasp_flag_temp = (x_temp > L - EPS) and (z_temp > W / 2 - EPS)
                fix_flag[idx] = fix_flag_temp
                grasp_flag[idx] = grasp_flag_temp

        cal_fix_constraint(L, W, seed_size)
        fix_particle_set = set()
        grasp_particle_set = set()
        for i in range(self.PARTICLE_NUM):
            if fix_flag[i]:
                fix_particle_set.add(i)
            if grasp_flag[i]:
                grasp_particle_set.add(i)
        fix_particle_list = list(fix_particle_set)
        grasp_particle_list = list(grasp_particle_set)

        grasp_idx = grasp_particle_list[0]
        grasp_ele_list = []
        for i in range(self.ELEMENT_NUM):
            ele_temp = self.element[i].to_numpy()
            if grasp_idx in ele_temp:
                grasp_ele_list.append(i)

        return fix_particle_list, grasp_particle_list, grasp_ele_list


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
        self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0)



def main():
    soft_obj = SoftObject(shape=[0.1, 0.1], seed_size=0.01)
    soft_obj.preset()
    window = soft_obj.window
    while window.running:
        soft_obj.substep()

if __name__ == '__main__':
    main()