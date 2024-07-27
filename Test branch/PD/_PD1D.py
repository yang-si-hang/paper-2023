"""
使用PD仿真1D的绳变形,基于Cosserat理论，只考虑拉伸和弯曲
created at 2024-07-23 by hsy
"""

import time
import numpy as np
from _CVVideo import *
from scipy import sparse
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, device_memory_GB=6.0, debug=True,default_fp=ti.f64)

output_folder = 'FigureWrite'

def generate_geometric(length, particle_num:int):
    node_np = np.zeros((particle_num, 2), dtype=np.float64)
    node_np[:, 0] = np.linspace(0, length, particle_num)

    element_np = np.zeros((particle_num-1, 2), dtype=np.int32)
    for i in range(particle_num-1):
        element_np[i] = [i, i+1]

    element_quat_np = np.zeros((particle_num-1, 2), dtype=np.float64)
    element_quat_np[:, 0] = 1.

    return node_np, element_np, element_quat_np


@ti.func
def theta2rot_matrix(theta:float):
    return ti.Matrix([[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]])


@ti.func
def quatconj2d(u):
    # 实部在前,虚部在后
    return ti.Vector([u[0], -u[1]])


@ti.func
def quatmul2d(u1, u2):
    return ti.Vector([u1[0]*u2[0]-u1[1]*u2[1], u1[0]*u2[1]+u1[1]*u2[0]])


@ti.func
def valiunitquat(u):
    norm = u.norm()
    if ti.abs(norm - 1) > 1e-6:
        print(f'Quaternion is not normalized: {norm}')
    else:
        print(f'Quaternion is normalized: {norm}')


@ti.data_oriented
class PD1D:
    def __init__(self, length, radius, seed_size:float):
        self.length = length
        self.radius = radius
        self.dt = 1./100
        self.rho = 1.e3
        self.E = 2.e6
        self.mu = 0.45
        self.positional_node_weight = 1.e6
        self.positional_element_weight = 1.e3
        self.contact_weight = 0.
        self.dim:int = 2
        self.solve_iteration = 20
        self.G = self.E / 2 / (1 + self.mu)
        self.section_area = tm.pi * self.radius ** 2

        self.PARTICLE_NUM:int = np.ceil(length / seed_size).astype(int) + 1
        self.ELEMENT_NUM:int = self.PARTICLE_NUM - 1
        self.ANGLE_NUM:int = self.ELEMENT_NUM - 1
        self.l = length / self.ELEMENT_NUM

        node_np, element_np, element_quat_np = generate_geometric(self.length, self.PARTICLE_NUM)
        np.savetxt('node_pos_init.csv', node_np, delimiter=',', fmt='%.6f')
        np.savetxt('element.csv', element_np, delimiter=',', fmt='%d')
        np.savetxt('element_quat.csv', element_quat_np, delimiter=',', fmt='%.6f')

        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_force = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_sn = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = 0.
        self.node_pos_init.from_numpy(node_np)
        self.node_pos.from_numpy(node_np)
        self.node_vel.fill(0.)
        self.node_force.fill(0.)

        self.element = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.element_quat = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)            # 单位四元数只取实部和虚部的Y轴部分
        self.element_quat_init = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_new = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_delta = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        # self.element_angle_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)       # 为了符合四元数乘法,实部为0,虚部为角速度
        self.element_sn = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        # self.element_ang_vel_sn = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.node_distance_unit = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_length = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element.from_numpy(element_np)
        self.element_quat_init.from_numpy(element_quat_np)
        self.element_quat.from_numpy(element_quat_np)
        self.element_quat_delta.from_numpy(np.insert(np.ones((self.ELEMENT_NUM, 1)), 1, np.zeros((self.ELEMENT_NUM,)), axis=1))
        self.element_inertia = ti.Vector([0., 0.])
        self.stretch_weight = 0.
        self.bend_weight = 0.

        self.A_stretch = ti.Matrix.field(4, 6, dtype=ti.f64, shape=())
        self.A_bend = ti.Matrix.field(4, 4, dtype=ti.f64, shape=())
        self.Bp_stretch = ti.Vector.field(4, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.Bp_bend = ti.Vector.field(4, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*2+self.ELEMENT_NUM*2, self.PARTICLE_NUM*2+self.ELEMENT_NUM*2))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*2+self.ELEMENT_NUM*2)

        # 固定尾部的节点和单元
        self.fix_particle_list = [self.PARTICLE_NUM-1]
        self.fix_quaternion_list = [self.ELEMENT_NUM-1]
        # 接触首部的节点和单元
        self.contact_particle_list = [0]
        self.contact_element_list = [0]
        self.contact_num = len(self.contact_particle_list)
        self.contact_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)
        self.contact_ang_vel = ti.field(dtype=ti.f64, shape=self.contact_num)         # 逆时针为正
        self.contact_vel.fill(0.)
        self.contact_ang_vel.fill(0.)

        self.node_desired_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)
        self.element_desired_quat = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)

        self.construct_mass()
        self.construct_weight()

        print(f'Particle Num: {self.PARTICLE_NUM}, Element Num: {self.ELEMENT_NUM}, Angle Num: {self.ANGLE_NUM}')
        print(f'node_mass: {self.node_mass}, element_inertia: {self.element_inertia}, stretch_weight: {self.stretch_weight}, bend_weight: {self.bend_weight}')
        print(f'Contact Node: {self.contact_particle_list}, Contact Element: {self.contact_element_list}')
        print(f'Fix Node: {self.fix_particle_list}, Fix Element: {self.fix_quaternion_list}')


    def construct_mass(self):
        self.node_mass = tm.pi * self.radius ** 2 * self.l * self.rho
        
        J1 = J2 = tm.pi * self.radius ** 4 / 4
        J3 = J1 + J2
        self.element_inertia = self.l * self.rho * ti.Vector([0., J1])


    def construct_weight(self):
        self.stretch_weight = self.E * self.section_area * self.l
        self.bend_weight = 2 * self.G * tm.pi * self.radius ** 4 / self.l
        # self.bend_weight = 1.e1


    @ti.kernel
    def precomputation(self):
        dim = self.dim

        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] = self.node_mass / self.dt ** 2
        
        for u_idx in range(self.PARTICLE_NUM, self.PARTICLE_NUM+self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.lhs[u_idx*dim+d, u_idx*dim+d] = self.element_inertia[d] / self.dt ** 2
        
        for d in ti.static(range(self.dim)):
            self.A_stretch[None][d, d] = -1. / self.l
        for d in ti.static(range(self.dim)):
            self.A_stretch[None][d, d+2] = 1. / self.l
        for d in ti.static(range(self.dim)):
            self.A_stretch[None][d+2, d+4] = 1

        for d in ti.static(range(self.dim+self.dim)):
            self.A_bend[None][d, d] = tm.sqrt(2)

        # Stretch Constraint
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            idx1_x, idx1_y = idx1*dim, idx1*dim+1
            idx2_x, idx2_y = idx2*dim, idx2*dim+1
            u_idx_s, u_idx_y = (self.PARTICLE_NUM+ele_idx)*dim, (self.PARTICLE_NUM+ele_idx)*dim+1       # 只取四元数的部分
            q_idx_vec= ti.Vector([idx1_x, idx1_y, idx2_x, idx2_y, u_idx_s, u_idx_y])
            A_i = self.A_stretch[None]
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(6, 6)):
                self.lhs[q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]] += self.stretch_weight * ATA[A_row_idx, A_col_idx]

        # Bend Constraint
        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            idx1_s, idx1_y = idx1*dim, idx1*dim+1
            idx2_s, idx2_y = idx2*dim, idx2*dim+1
            u_idx_vec= ti.Vector([idx1_s, idx1_y, idx2_s, idx2_y])
            A_i = self.A_bend[None]
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(4, 4)):
                self.lhs[u_idx_vec[A_row_idx], u_idx_vec[A_col_idx]] += self.bend_weight * ATA[A_row_idx, A_col_idx]

        for q_idx in ti.static(self.fix_particle_list):
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.positional_node_weight * A_i_eye[d, d]

        for u_idx in ti.static(self.fix_quaternion_list):
            s_idx = u_idx + self.PARTICLE_NUM
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.dim)):
                self.lhs[s_idx*dim+d, s_idx*dim+d] += self.positional_element_weight * A_i_eye[d, d]

        
    @ti.kernel
    def construct_desired_pos(self):
        for idx in ti.static(range(self.contact_num)):
            q_idx = self.contact_particle_list[idx]
            u_idx = self.contact_element_list[idx]

            self.node_desired_pos[idx] = self.node_pos[q_idx] + self.dt * self.contact_vel[idx]
            delta_theta = self.dt * self.contact_ang_vel[idx]
            self.element_desired_quat[idx] = quatmul2d(self.element_quat[u_idx], ti.Vector([tm.cos(delta_theta), tm.sin(delta_theta)]))

    
    @ti.kernel
    def construct_sn(self):
        # 参考soler2018cosserat的更新公式
        for q_idx in range(self.PARTICLE_NUM):
            self.node_sn[q_idx] = self.node_pos[q_idx] + self.dt * self.node_vel[q_idx] + self.dt**2 * self.node_force[q_idx]        # shape: (2, 1)

        for u_idx in range(self.ELEMENT_NUM):
            # 2d的情况
            # self.element_ang_vel_sn[u_idx] = self.element_angle_vel[u_idx]                      # shape: (2, 1)
            # quat_tmp = ti.Vector([0., self.element_ang_vel_sn[u_idx][1]])                       # 只取虚部
            # element_sn_tmp = self.element_quat[u_idx] + self.dt * quatmul2d(self.element_quat[u_idx], quat_tmp) / 2
            # valiunitquat(element_sn_tmp)
            # 不需要显式计算角速度，可以只计算单位时间步长下的姿态变化量
            delta_quat = self.element_quat_delta[u_idx]
            self.element_sn[u_idx] = quatmul2d(self.element_quat[u_idx], delta_quat)            # shape: (2, 1)


    @ti.kernel
    def warm_start(self):
        for q_idx in range(self.PARTICLE_NUM):
            self.node_pos_new[q_idx] = self.node_pos[q_idx]
            # self.node_pos_new[q_idx] = self.node_sn[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            self.element_quat_new[u_idx] = self.element_quat[u_idx]
            # self.element_quat_new[u_idx] = self.element_sn[u_idx]


    @ti.kernel
    def local_solve(self):
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            quat_new = self.element_quat_new[ele_idx]
            distance_vec = (self.node_pos_new[idx2] - self.node_pos_new[idx1])
            distance_vec_unit = distance_vec.normalized()
            # self.distance_vec[ele_idx] = distance_vec_unit
            self.element_length[ele_idx] = distance_vec.norm()
            d3 = quat_new
            # 确保element的方向的d3的方向一致,可以考虑单独作为一个constraint
            u_constaint = distance_vec_unit
            self.Bp_stretch[ele_idx] = ti.Vector([d3[0], d3[1], u_constaint[0], u_constaint[1]])

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = angle_idx, angle_idx + 1
            u1, u2 = self.element_quat_new[idx1], self.element_quat_new[idx2]
            u_average = (u1 + u2)
            u_average = u_average.normalized()
            self.Bp_bend[angle_idx] = ti.Vector([u_average[0], u_average[1], u_average[0], u_average[1]])


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] = self.node_mass * self.node_sn[q_idx][d] / self.dt ** 2

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[(u_idx+self.PARTICLE_NUM)*dim+d] = self.element_inertia[d] * self.element_sn[u_idx][d] / self.dt ** 2

        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            u_idx = self.PARTICLE_NUM + ele_idx
            q_idx_vec = ti.Vector([idx1*dim, idx1*dim+1, idx2*dim, idx2*dim+1, u_idx*dim, u_idx*dim+1])
            A_i = self.A_stretch[None]
            AT_Bp_i = self.stretch_weight * A_i.transpose() @ self.Bp_stretch[ele_idx]
            for d in ti.static(range(6)):
                self.rhs[q_idx_vec[d]] += AT_Bp_i[d]

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            u_idx_vec = ti.Vector([idx1*dim, idx1*dim+1, idx2*dim, idx2*dim+1])
            A_i = self.A_bend[None]
            AT_Bp_i = self.bend_weight * A_i.transpose() @ self.Bp_bend[angle_idx]
            for d in ti.static(range(4)):
                self.rhs[u_idx_vec[d]] += AT_Bp_i[d]

        for q_idx in ti.static(self.fix_particle_list):
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] += self.positional_node_weight * self.node_pos_init[q_idx][d]

        for u_idx in ti.static(self.fix_quaternion_list):
            s_idx = u_idx + self.PARTICLE_NUM
            for d in ti.static(range(self.dim)):
                self.rhs[s_idx*dim+d] += self.positional_element_weight * self.element_quat_init[u_idx][d]

        for idx in ti.static(range(self.contact_num)):
            q_idx = self.contact_particle_list[idx]
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] += self.contact_weight * self.node_desired_pos[idx][d]

        for idx in ti.static(range(self.contact_num)):
            u_idx = self.contact_element_list[idx] + self.PARTICLE_NUM
            for d in ti.static(range(self.dim)):
                self.rhs[u_idx*dim+d] += self.contact_weight * self.element_desired_quat[idx][d]


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.node_pos_new[q_idx][d] = sol[q_idx*self.dim+d]

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.element_quat_new[u_idx][d] = sol[(u_idx+self.PARTICLE_NUM)*self.dim+d]    


    @ti.kernel
    def quat_normalize(self):
        for u_idx in range(self.ELEMENT_NUM):
            u_tmp = self.element_quat_new[u_idx]
            u_normalized = u_tmp.normalized()
            self.element_quat_new[u_idx] = u_normalized

    
    @ti.kernel
    def update_vel_pos(self):
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.node_vel[q_idx][d] = (self.node_pos_new[q_idx][d] - self.node_pos[q_idx][d]) / self.dt
                self.node_pos[q_idx][d] = self.node_pos_new[q_idx][d]

        for u_idx in range(self.ELEMENT_NUM):
            # tmp = quatmul2d(quatconj2d(self.element_quat[u_idx]), self.element_quat_new[u_idx])[0]            # 角速度只能取虚部
            # self.element_angle_vel[u_idx] = 2 * ti.Vector([0, tmp]) / self.dt
            self.element_quat_delta[u_idx] = quatmul2d(quatconj2d(self.element_quat[u_idx]), self.element_quat_new[u_idx])
            self.element_quat[u_idx] = self.element_quat_new[u_idx]
            idx1, idx2 = self.element[u_idx]
            self.node_distance_unit[u_idx] = (self.node_pos_new[idx2] - self.node_pos_new[idx1]).normalized()


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
        scene.ambient_light((1., 1., 1.))
        return window, camera, scene


    def show_preset(self):
        """
        Define the data for GGUI
        """
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_NUM)
        self.edge_show = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.edge_show.from_numpy(self.element.to_numpy(dtype=np.int32))


    def gui_show(self, ggui_set, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=None, name_list=None):
        """
        Show the GGUI
        """
        window, canvas, scene = ggui_set['window'], ggui_set['canvas'], ggui_set['scene']
        if SHOW_FLAG is False:
            return
        scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        scene.ambient_light((0.8, 0.8, 0.8))
        self.node_show.from_numpy(np.insert(self.node_pos.to_numpy(dtype=np.float32), 1, np.zeros(self.PARTICLE_NUM), axis=1))

        scene.particles(self.node_show, radius=0.003, color=(0., 0., 0.))
        scene.lines(self.node_show, width=2., indices=self.edge_show, color=(0., 0., 0.),
                    vertex_count=0)
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
        """
        Define the camera position & target
        """
        self.window, self.camera, self.scene = self.gui_set(pos=camera_pos, target=camera_target)
        self.canvas = self.window.get_canvas()
        self.show_preset()


    @ti.kernel
    def init_vel(self):
        # self.node_vel[0][1] = 10.
        self.node_force[0][1] = 9.8 * self.node_mass * 20


    def substep(self, step_num, frame_name_list):
        self.construct_desired_pos()
        self.construct_sn()
        self.warm_start()

        # for itr in range(self.solve_iteration):
        for itr in range(20):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            state_sol = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(state_sol)
            self.quat_normalize()
        # np.savetxt('rhs.csv', self.rhs.to_numpy(), delimiter=',', fmt='%.8f')
        # exit(0)

        self.update_vel_pos()
        ggui_set = {'window': self.window, 'canvas': self.canvas, 'scene': self.scene}
        frame_name_list = self.gui_show(ggui_set, SHOW_FLAG=True, WRITE_FLAG=True, itr_num=step_num, name_list=frame_name_list)
        return frame_name_list


def main():
    class MyObject(PD1D):
        def __init__(self, length, radius, seed_size):
            super(MyObject, self).__init__(length, radius, seed_size)

    soft_obj = MyObject(length=1., radius=0.01, seed_size=0.05)
    soft_obj.preset_gui(camera_pos=[0.5, 0.75, 0.3], camera_target=[0.5, 0., 0.3])

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    # np.savetxt('lhs.csv', lhs_np, delimiter=',', fmt='%.8f')
    soft_obj.init_vel()

    frame_name_list = []

    for i in range(2000):
        frame_name_list = soft_obj.substep(i, frame_name_list)
        # np.savetxt('rhs.csv', soft_obj.rhs.to_numpy(), delimiter=',', fmt='%.8f')
        # print(f'Iter: {i}--------------------------------------')
        # print(f'Node Position: {soft_obj.node_pos.to_numpy()}')
        # print(f'Node Distace Normalized: {soft_obj.node_distance_unit.to_numpy()}')
        # print(f'Element Quaternion: {soft_obj.element_quat.to_numpy()}')

    image_to_video(frame_name_list, video_filename='output_video.mp4')

if __name__ == '__main__':
    main()