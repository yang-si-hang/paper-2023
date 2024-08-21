"""
使用PD仿真1D的绳变形,基于Cosserat理论
只考虑平面上的弯曲,使用两侧端点作为约束
created at 2024-08-21 by hsy
"""

import time
import numpy as np
from _CVVideo import *
from scipy import sparse
import taichi as ti
import taichi.math as tm
# ti.init(arch=ti.gpu, device_memory_GB=6.0, debug=True,default_fp=ti.f64)
np.set_printoptions(linewidth=160)
ti.init(arch=ti.gpu, debug=True, default_fp=ti.f64)

output_folder = 'FigureWrite'

def generate_geometric(length, particle_num:int):
    node_np = np.zeros((particle_num, 2), dtype=np.float64)
    X = length * np.cos(0 * np.pi / 2)
    Y = length * np.sin(0 * np.pi / 2)
    node_np[:, 0] = np.linspace(0, X, particle_num)
    node_np[:, 1] = np.linspace(0, Y, particle_num)

    element_np = np.zeros((particle_num-1, 2), dtype=np.int32)
    for i in range(particle_num-1):
        element_np[i] = [i, i+1]

    element_quat_np = np.zeros((particle_num-1, 2), dtype=np.float64)
    element_quat_np[:, 0] = np.cos(0 * np.pi / 2)
    element_quat_np[:, 1] = np.sin(0 * np.pi / 2)

    return node_np, element_np, element_quat_np


@ti.func
def quat2rot(u):
    return ti.Matrix([[u[0], -u[1]], [u[1], u[0]]])


@ti.func
def quatconj(u):
    return ti.Vector([u[0], -u[1]])


@ti.data_oriented
class PD1D:
    def __init__(self, length, radius, seed_size:float):
        self.length = length
        self.radius = radius
        self.dt = 1./100
        self.rho = 1.e3
        self.E = 2.e7
        self.mu = 0.45
        self.positional_node_weight = 1.e5
        self.positional_ele_weight = 1.e5
        self.contact_node_weight = 1.e5
        self.contact_ele_weight = 0.
        self.dim:int = 2
        self.quat_dim:int = 2
        self.solve_iteration = 200
        self.G = self.E / 2 / (1 + self.mu)
        self.section_area = tm.pi * self.radius ** 2

        self.PARTICLE_NUM:int = np.ceil(length / seed_size).astype(int) + 1
        self.ELEMENT_NUM:int = self.PARTICLE_NUM - 1
        self.ANGLE_NUM:int = self.ELEMENT_NUM - 1
        self.l = length / self.ELEMENT_NUM

        node_np, element_np, element_quat_np = generate_geometric(self.length, self.PARTICLE_NUM)
        np.savetxt('node_pos_init.csv', node_np, delimiter=',', fmt='%.6f')
        np.savetxt('element.csv', element_np, delimiter=',', fmt='%d')
        np.savetxt('element_quat_init.csv', element_quat_np, delimiter=',', fmt='%.6f')

        self.node_end_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=2)
        self.node_end_pos_init = ti.Vector.field(self.dim, dtype=ti.f64, shape=2)
        self.node_end_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=2)
        self.node_end_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=2)
        self.node_end_sn = ti.Vector.field(self.dim, dtype=ti.f64, shape=2)
        self.node_end_pos_init.from_numpy(np.array([node_np[0,:], node_np[-1,:]]))
        self.node_end_pos.from_numpy(np.array([node_np[0,:], node_np[-1,:]]))
        self.node_end_vel.fill(0.)

        self.ele_indices = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.ele_quat = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_quat_init = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_quat_new = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_quat_residual = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_angle_vel = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_torque = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_sn = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_inertia_vector = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_inertia = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_inv_inertia = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_indices.from_numpy(element_np)
        self.ele_quat_init.from_numpy(element_quat_np)
        self.ele_quat.from_numpy(element_quat_np)
        self.ele_angle_vel.fill(0.)
        self.ele_torque.fill(0.)

        self.bend_weight = 0.
        self.ele_offset = 2 * self.dim

        self.A_bend = ti.Matrix.field(self.quat_dim, 2*self.quat_dim, dtype=ti.f64, shape=())
        self.A_normalize = ti.Matrix.field(self.quat_dim, self.quat_dim, dtype=ti.f64, shape=())
        self.A_length = ti.field(dtype=ti.f64, shape=(2, 2*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.Bp_bend = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.Bp_length = ti.Vector.field(self.dim, dtype=ti.f64, shape=())
        self.Bp_normalize = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.lhs = ti.field(dtype=ti.f64, shape=(2*self.dim+self.ELEMENT_NUM*self.quat_dim, 2*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.bend_constraint = ti.field(dtype=ti.f64, shape=(2*self.dim+self.ELEMENT_NUM*self.quat_dim, 2*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.rhs = ti.field(dtype=ti.f64, shape=2*self.dim+self.ELEMENT_NUM*self.quat_dim)
        self.state_sol = ti.field(dtype=ti.f64, shape=2*self.dim+self.ELEMENT_NUM*self.quat_dim)
        self.lhs.fill(0.)

        # 固定尾部的节点和单元
        self.fix_particle_list = [1]
        self.fix_quaternion_list = [self.ELEMENT_NUM-1]
        # 接触首部的节点和单元
        self.contact_particle_list = [0]
        self.contact_element_list = [0]
        self.contact_par_num = len(self.contact_particle_list)
        self.contact_ele_num = len(self.contact_element_list)
        self.contact_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.contact_par_num)
        self.contact_ang_vel = ti.field(dtype=ti.f64, shape=self.contact_ele_num)         # 逆时针为正
        self.contact_vel.fill(0.)
        self.contact_ang_vel.fill(0.)

        self.node_desired_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.contact_par_num)
        self.element_desired_quat = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.contact_ele_num)

        self.construct_mass()
        self.construct_weight()

        print(f'Particle Num: {self.PARTICLE_NUM}; Element Num: {self.ELEMENT_NUM}; Angle Num: {self.ANGLE_NUM}')
        print(f'element_inertia: {self.ele_inertia[0]:.6e}, bend_weight: {self.bend_weight:.6e}, length_weight: {self.length_weight}')
        print(f'Contact Node: {self.contact_particle_list}, Contact Element: {self.contact_element_list}')
        print(f'Fix Node: {self.fix_particle_list}, Fix Element: {self.fix_quaternion_list}')


    def construct_mass(self):
        # for idx in range(self.PARTICLE_NUM):
        #     self.node_mass[idx] = tm.pi * self.radius ** 2 * self.length * self.rho / self.PARTICLE_NUM
        
        J1 = self.rho * tm.pi * self.radius ** 2 * self.l * (3*self.radius ** 2 + self.l ** 2) / 12
        for u_idx in range(self.ELEMENT_NUM):
            # 都做了近似,忽略了element的长度
            self.ele_inertia[u_idx] = J1
            self.ele_inv_inertia[u_idx] = 1. / self.ele_inertia[u_idx]
            # 参考soler2018cosserat,未进行修改(理论方法未知)
            self.ele_inertia_vector[u_idx] = ti.Vector([J1, J1])
            # 实际测试:[J1, J1]与[0, J1]的结果是一样的


    def construct_weight(self):
        # self.bend_weight = 2 * self.G * tm.pi * self.radius ** 4 / self.l
        self.bend_weight = 4 * self.E / self.l * tm.pi * self.radius ** 4 / 4
        self.length_weight = 1.e4


    @ti.kernel
    def precomputation(self):
        dim = self.dim
        quat_dim  = self.quat_dim
        ele_offset = self.ele_offset
        
        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.quat_dim)):
                self.lhs[ele_offset+u_idx*quat_dim+d, ele_offset+u_idx*quat_dim+d] = self.ele_inertia_vector[u_idx][d] / self.dt ** 2
        # print('Element Inertia:', self.ele_inertia_vector[0].y / self.dt ** 2)

        for d in range(self.quat_dim):
            self.A_bend[None][d, d] = 1.
            self.A_bend[None][d, d+2] = -1.

        # Rod length Constraint
        for d in range(self.dim):
            self.A_length[d, d] = 1.
            self.A_length[d, d+2] = -1.
        for u_idx in range(self.ELEMENT_NUM):
            self.A_length[0, ele_offset+u_idx*quat_dim] = self.l
            self.A_length[1, ele_offset+u_idx*quat_dim+1] = self.l

        for d in range(self.quat_dim):
            self.A_normalize[None][d, d] = 1.

        # Bend Constraint
        for angle_idx in range(self.ANGLE_NUM):
            idx1_x, idx1_y = ele_offset + angle_idx*quat_dim, ele_offset + angle_idx*quat_dim+1
            idx2_x, idx2_y = ele_offset + angle_idx*quat_dim+2, ele_offset + angle_idx*quat_dim+3
            qu_idx_vec = ti.Vector([idx1_x, idx1_y, idx2_x, idx2_y])
            A_i = self.A_bend[None]
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(4, 4)):
                self.lhs[qu_idx_vec[A_row_idx], qu_idx_vec[A_col_idx]] += self.bend_weight * ATA[A_row_idx, A_col_idx]
                self.bend_constraint[qu_idx_vec[A_row_idx], qu_idx_vec[A_col_idx]] += self.bend_weight * ATA[A_row_idx, A_col_idx]

        # Rod length Constraint
        for row_idx in range(2*self.dim+self.ELEMENT_NUM*self.quat_dim):
            for col_idx in range(2*self.dim+self.ELEMENT_NUM*self.quat_dim):
                tmp = 0.
                for d in range(2):
                    tmp += self.A_length[d, row_idx] * self.A_length[d, col_idx]
                self.lhs[row_idx, col_idx] += self.length_weight * tmp
                # self.length_constraint[row_idx, col_idx] += self.length_weight * tmp
        
        # Normalize Element Quaternion
        # for u_idx in range(self.ELEMENT_NUM):
        #     qu_idx_vec = ti.Vector([u_idx*quat_dim, u_idx*quat_dim+1]) + ele_offset
        #     A_i = self.A_normalize[None]
        #     ATA = A_i.transpose() @ A_i
        #     for A_row_idx, A_col_idx in ti.static(ti.ndrange(2, 2)):
        #         self.lhs[qu_idx_vec[A_row_idx], qu_idx_vec[A_col_idx]] += self.ele_normalize_weight * ATA[A_row_idx, A_col_idx]

        # Positional Constraint
        for q_idx in ti.static(self.fix_particle_list):
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.positional_node_weight * A_i_eye[d, d]

        for u_idx in ti.static(self.fix_quaternion_list):
            qu_idx = ele_offset + u_idx*quat_dim
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.quat_dim)):
                self.lhs[qu_idx+d, qu_idx+d] += self.positional_ele_weight * A_i_eye[d, d]

        # Contact Constraint
        for q_idx in ti.static(self.contact_particle_list):
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.contact_node_weight * A_i_eye[d, d]


    @ti.kernel
    def construct_desired_pos(self):
        for idx in ti.static(range(self.contact_par_num)):
            q_idx = self.contact_particle_list[idx]
            self.node_desired_pos[idx] = self.node_end_pos[q_idx] + self.dt * self.contact_vel[idx]

        for idx in ti.static(range(self.contact_ele_num)):
            u_idx = self.contact_element_list[idx]
            delta_quat = self.dt * quat2rot(self.ele_quat[u_idx]) @ ti.Vector([0., self.contact_ang_vel[idx]])
            ele_desired_quat_tmp = self.ele_quat[u_idx] + delta_quat
            self.element_desired_quat[idx] = ele_desired_quat_tmp.normalized()


    @ti.kernel
    def construct_sn(self):
        # 参考soler2018cosserat的更新公式
        for q_idx in range(2):
            self.node_end_sn[q_idx] = self.node_end_pos[q_idx] + self.dt * self.node_end_vel[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            ele_angle_vel_old = self.ele_angle_vel[u_idx]
            ele_angle_vel_new = ele_angle_vel_old + self.dt * self.ele_inv_inertia[u_idx] * self.ele_torque[u_idx]
            delta_quat = self.dt * quat2rot(self.ele_quat[u_idx]) @ ti.Vector([0., ele_angle_vel_new])
            # print(f'Delta Quat: {delta_quat}; Rotation: {quat2rot(self.ele_quat[u_idx])}, Angle Vel: {ele_angle_vel_new}')
            ele_sn_tmp = self.ele_quat[u_idx] + delta_quat
            self.ele_sn[u_idx] = ele_sn_tmp.normalized()


    @ti.kernel
    def warm_start(self):
        for q_idx in range(2):
            # self.node_end_pos_new[q_idx] = self.node_end_pos[q_idx]
            self.node_end_pos_new[q_idx] = self.node_end_sn[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            # self.ele_quat_new[u_idx] = self.ele_quat[u_idx]
            self.ele_quat_new[u_idx] = self.ele_sn[u_idx]


    @ti.kernel
    def local_solve(self):
        for angle_idx in range(self.ANGLE_NUM):
            u1, u2 = self.ele_quat_new[angle_idx], self.ele_quat_new[angle_idx+1]
            u1_residual = (1 - 1/u1.norm()) * u1
            u2_residual = (1 - 1/u2.norm()) * u2
            self.Bp_bend[angle_idx] = u1_residual - u2_residual

        self.Bp_length[None].fill(0.)
        for u_idx in range(self.ELEMENT_NUM):
            u = self.ele_quat_new[u_idx]
            u_residual = (1 - 1/u.norm()) * u
            self.Bp_length[None] += u_residual * self.l


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        quat_dim = self.quat_dim
        ele_offset = self.ele_offset

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.quat_dim)):
                self.rhs[ele_offset+u_idx*quat_dim+d] = self.ele_inertia_vector[u_idx][d] * self.ele_sn[u_idx][d] / self.dt ** 2

        # Bend Constraint
        for angle_idx in range(self.ANGLE_NUM):
            qu_idx_vec = ti.Vector([angle_idx*quat_dim, angle_idx*quat_dim+1, angle_idx*quat_dim+2, angle_idx*quat_dim+3]) + ele_offset
            AT_Bp_i = self.A_bend[None].transpose() @ self.Bp_bend[angle_idx]
            for d in range(4):
                self.rhs[qu_idx_vec[d]] += self.bend_weight * AT_Bp_i[d]

        # Rod length Constraint
        for row_idx in range(2*self.dim+self.ELEMENT_NUM*self.quat_dim):
            tmp = 0.
            for d in range(2):
                tmp += self.A_length[d, row_idx] * self.Bp_length[None][d]
            self.rhs[row_idx] += self.length_weight * tmp

        # Normalize Element Quaternion
        # for u_idx in range(self.ELEMENT_NUM):
        #     qu_idx_vec = ti.Vector([u_idx*quat_dim, u_idx*quat_dim+1]) + ele_offset
        #     AT_Bp_i = self.ele_normalize_weight * self.A_normalize[None].transpose() @ self.Bp_normalize[u_idx]
        #     for d in range(2):
        #         self.rhs[qu_idx_vec[d]] += AT_Bp_i[d]

        for q_idx in ti.static(self.fix_particle_list):
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] += self.positional_node_weight * self.node_end_pos_init[q_idx][d]

        for u_idx in ti.static(self.fix_quaternion_list):
            for d in ti.static(range(self.quat_dim)):
                self.rhs[ele_offset+u_idx*quat_dim+d] += self.positional_ele_weight * self.ele_quat_init[u_idx][d]

        for idx in ti.static(range(self.contact_par_num)):
            q_idx = self.contact_particle_list[idx]
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] += self.contact_node_weight * self.node_desired_pos[idx][d]

        for idx in ti.static(range(self.contact_ele_num)):
            u_idx = self.contact_element_list[idx]
            for d in ti.static(range(self.dim)):
                self.rhs[ele_offset+u_idx*quat_dim+d] += self.contact_ele_weight * self.element_desired_quat[idx][d]


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        for q_idx in range(2):
            for d in ti.static(range(self.dim)):
                self.node_end_pos_new[q_idx][d] = sol[q_idx*self.dim+d]

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.quat_dim)):
                self.ele_quat_new[u_idx][d] = sol[self.ele_offset+u_idx*self.quat_dim+d]


    @ti.kernel
    def quat_normalize(self):
        for u_idx in range(self.ELEMENT_NUM):
            u_tmp = self.ele_quat_new[u_idx]
            u_normalized = u_tmp.normalized()
            self.ele_quat_new[u_idx] = u_normalized

    
    @ti.kernel
    def update_vel_pos(self):
        for q_idx in range(2):
            for d in ti.static(range(self.dim)):
                self.node_end_vel[q_idx][d] = (self.node_end_pos_new[q_idx][d] - self.node_end_pos[q_idx][d]) / self.dt
                self.node_end_pos[q_idx][d] = self.node_end_pos_new[q_idx][d]

        for u_idx in range(self.ELEMENT_NUM):
            ele_quat_tmp = self.ele_quat_new[u_idx].normalized()
            delta_quat = quat2rot(quatconj(self.ele_quat[u_idx])) @ ele_quat_tmp
            angle_vel_tmp = ti.atan2(delta_quat[1], delta_quat[0]) / self.dt
            self.ele_angle_vel[u_idx] = angle_vel_tmp
            self.ele_quat[u_idx] = ele_quat_tmp


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
        camera.up(0., 1., 0.)
        camera.z_near(0.01)
        camera.fov(FOV)

        # set the camera, you can move around by pressing 'wasdeq'
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        # set the light
        scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        scene.ambient_light((1., 1., 1.))
        return window, camera, scene


    def show_preset(self):
        # Define the data for GGUI
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=2)
        self.quat_node = ti.Vector.field(2, dtype=ti.f32, shape=self.ELEMENT_NUM*2)
        self.quat_node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.ELEMENT_NUM*2)
        self.quat_show = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.quat_show.from_numpy(np.arange(0, self.ELEMENT_NUM*2).reshape(-1, 2))

    
    @ti.kernel
    def quat_node_update_ghost(self):
        self.quat_node[self.ELEMENT_NUM*2-1] = ti.cast(self.node_end_pos[1], ti.f32)
        ti.loop_config(serialize=True)
        for u_idx in range(self.ELEMENT_NUM-1):
            u_idx_inv = self.ELEMENT_NUM - u_idx - 1
            quat = self.ele_quat[u_idx_inv]
            vec = quat
            self.quat_node[u_idx_inv*2] = ti.cast(self.quat_node[u_idx_inv*2+1] - vec * self.l, ti.f32)
            self.quat_node[u_idx_inv*2-1] = ti.cast(self.quat_node[u_idx_inv*2], ti.f32)
        quat = self.ele_quat[0]
        vec = quat
        self.quat_node[0] = self.quat_node[1] - ti.cast(vec * self.l, ti.f32)


    def gui_show(self, ggui_set, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=None, name_list=None):
        # Show the GGUI
        window, canvas, scene = ggui_set['window'], ggui_set['canvas'], ggui_set['scene']
        if SHOW_FLAG is False:
            return
        scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        scene.ambient_light((0.8, 0.8, 0.8))
        self.node_show.from_numpy(np.hstack((self.node_end_pos.to_numpy(), np.zeros((2, 1)))))
        self.quat_node_update_ghost()
        self.quat_node_show.from_numpy(np.hstack((self.quat_node.to_numpy(), np.zeros((self.ELEMENT_NUM*2, 1)))))
        # print(f'Quat Node Show: {self.quat_node_show.to_numpy()}')
        # print(f'Node Show: {self.node_show.to_numpy()}')

        scene.particles(self.quat_node_show, radius=0.003, color=(0., 0., 0.))
        scene.particles(self.node_show, radius=0.005, color=(0., 0., 1.))
        # scene.lines(self.node_show, width=2., indices=self.edge_show, color=(0., 0., 0.), vertex_count=0)
        scene.lines(self.quat_node_show, width=2., indices=self.quat_show, color=(1., 0., 0.), vertex_count=0)
        canvas.scene(scene)
        canvas.set_background_color((1.0, 1.0, 1.0))
        # if WRITE_FLAG is True and itr_num % 10 == 0:
        if WRITE_FLAG is True:
            filename = os.path.join(output_folder, f'frame_{itr_num:04d}.png')
            window.save_image(f'{filename}')
            name_list.append(filename)
        window.show()
        return name_list


    def preset_gui(self, camera_pos:list, camera_target:list):
        # Define the camera position & target
        self.window, self.camera, self.scene = self.gui_set(pos=camera_pos, target=camera_target)
        self.canvas = self.window.get_canvas()
        self.show_preset()


    @ti.kernel
    def init_vel(self):
        # self.node_vel[0][1] = 1.
        # self.node_force[0][1] = -9.8 * self.node_mass[0]
        self.ele_torque[0] = 9.8 * 0.05 * self.l


    def substep(self, step_num, frame_name_list):
        self.construct_desired_pos()
        self.construct_sn()
        self.warm_start()

        ele_quat_theta1_list = []
        ele_quat_theta2_list = []
        distance_vec1_list = []
        distance_vec2_list = []
        distance_theta1_list = []
        distance_theta2_list = []

        for itr in range(self.solve_iteration):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()

            # np.savetxt('rhs.csv', rhs_np, delimiter=',', fmt='%.8f')
            # exit(0)
            node_pos_new_bf = self.node_end_pos_new.to_numpy()
            ele_quat_new_bf = self.ele_quat_new.to_numpy()
            state_sol = self.pre_fact_lhs_solve(rhs_np)
            # if itr == self.solve_iteration-1:
            print(f'Bp_bend: \n{self.Bp_bend.to_numpy()}')
            print(f'Bp_length: \n{self.Bp_length.to_numpy()}')
            print(f'Rhs: \n{rhs_np}')
            print(f'State Sol: \n{state_sol}')

            self.state_sol.from_numpy(state_sol)
            self.update_pos_new(state_sol)
            # self.quat_normalize()

            ele_quat1 = self.ele_quat_new[0]
            ele_quat2 = self.ele_quat_new[1]
            quat_theta1 = ti.atan2(ele_quat1[1], ele_quat1[0])
            quat_theta2 = ti.atan2(ele_quat2[1], ele_quat2[0])

            ele_quat_theta1_list.append(quat_theta1*180/tm.pi)
            ele_quat_theta2_list.append(quat_theta2*180/tm.pi)

            # if itr == self.solve_iteration-1:
                # print(f'Itr:{itr}------------------------------------------------------')
                # print(f'Node pos new: {node_pos_new_bf}')
                # print(f'Ele new: {ele_quat_new_bf}')

                # print(f'State Sol: {state_sol}')
                # print(f'Length Error: {self.A_length.to_numpy() @ state_sol}; Norm: {np.linalg.norm(self.A_length.to_numpy() @ state_sol)}')

                # print(f'Bp Bend: {self.Bp_bend.to_numpy()}')

                # print('1:', 'Ele Quat:', ele_quat1, 'Ele Theta:', quat_theta1*180/tm.pi)
                # print('2:', 'Ele Quat:', ele_quat2, 'Ele Theta:', quat_theta2*180/tm.pi)

        data_dict = {
            'distance_vec1': distance_vec1_list,
            'distance_vec2': distance_vec2_list,
            'distance_theta1': distance_theta1_list,
            'distance_theta2': distance_theta2_list,
            'ele_quat_theta1': ele_quat_theta1_list,
            'ele_quat_theta2': ele_quat_theta2_list
        }
        if step_num == 0:
            np.savez(f'DataWrite/local_solve_{step_num}.npz', **data_dict)
            print('Save Local Solve Data')

        self.update_vel_pos()
        ggui_set = {'window': self.window, 'canvas': self.canvas, 'scene': self.scene}
        frame_name_list = self.gui_show(ggui_set, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=step_num, name_list=frame_name_list)
        return frame_name_list


def main():
    class MyObject(PD1D):
        def __init__(self, length, radius, seed_size):
            super(MyObject, self).__init__(length, radius, seed_size)

    soft_obj = MyObject(length=1., radius=0.01, seed_size=0.2)
    soft_obj.preset_gui(camera_pos=[0.5, -0.3, 0.75], camera_target=[0.5, -0.3, 0.])

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    np.savetxt('lhs_pd1d.csv', lhs_np, delimiter=',', fmt='%.8f')
    np.savetxt('A_length.csv', soft_obj.A_length.to_numpy(), delimiter=',', fmt='%.8f')
    # np.savetxt('bend_constraint.csv', soft_obj.bend_constraint.to_numpy(), delimiter=',', fmt='%.8f')
    # soft_obj.init_vel()
    soft_obj.contact_vel[0] = ti.Vector([0.1, -0.1]) * 1.e0

    frame_name_list = []

    for i in range(1):
        frame_name_list = soft_obj.substep(i, frame_name_list)
        # time.sleep(1.)
        # np.savetxt('rhs.csv', soft_obj.rhs.to_numpy(), delimiter=',', fmt='%.8f')
        print(f'Iter: {i} --------------------------------------')
        print(f'Node Position: {soft_obj.node_end_pos.to_numpy()}')
        print(f'Element Quaternion: {soft_obj.ele_quat.to_numpy()}')
        print(f'Element Angle Vel: {soft_obj.ele_angle_vel.to_numpy()}')
        print(f'Length Error: {soft_obj.A_length.to_numpy() @ soft_obj.state_sol.to_numpy()}; Norm: {np.linalg.norm(soft_obj.A_length.to_numpy() @ soft_obj.state_sol.to_numpy())}')
        print(f'End Pos Error: {soft_obj.node_end_pos[0].to_numpy() + soft_obj.l * np.sum(soft_obj.ele_quat.to_numpy(), axis=0) - soft_obj.node_end_pos[1].to_numpy()}')
        # print(f'Element Quaternion: {soft_obj.ele_quat[0].to_numpy()}; Vel: {soft_obj.ele_angle_vel[0]}')

    # image_to_video(frame_name_list, video_filename='output_video.mp4')

if __name__ == '__main__':
    main()