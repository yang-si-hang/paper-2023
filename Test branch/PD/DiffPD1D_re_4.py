"""
DiffPD的一维版本,基于Cosserat杆理论,只有弯曲和拉伸
由于观察到离散程度增大的时候,求解器的收敛速度会变慢(需要更大的迭代次数),与论文中的结果不一致
改为使用三维的情况,并且与论文中的参数全部一致
created at 2024-08-14 by hsy
"""

import time
import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)
np.set_printoptions(linewidth=200, precision=6)
from GGUI import * 


output_folder = 'FigureWrite'

def generate_geometric(length, particle_num:int):
    node_np = np.zeros((particle_num, 3), dtype=np.float64)
    X = length * np.cos(0 * np.pi / 2)
    Y = length * np.sin(0 * np.pi / 2)
    # 在X-Z平面
    node_np[:, 0] = np.linspace(0, X, particle_num)
    node_np[:, 1] = np.zeros(particle_num)
    node_np[:, 2] = np.linspace(0, Y, particle_num)

    element_np = np.zeros((particle_num-1, 2), dtype=np.int32)
    for i in range(particle_num-1):
        element_np[i] = [i, i+1]

    # rest_e3 = np.array([0., 0., 1.])
    # element_quat_np = np.zeros((particle_num-1, 4), dtype=np.float64)
    # for idx in range(particle_num-1):
    #     # element_quat_np[idx] = (node_np[idx+1] - node_np[idx]) / np.linalg.norm(node_np[idx+1] - node_np[idx])
    #     theta = np.arctan2(node_np[idx+1, 0] - node_np[idx, 0], node_np[idx+1, 1] - node_np[idx, 1])
    #     element_quat_np[idx] = [np.cos(theta/2), np.sin(theta/2)]

    return node_np, element_np


@ti.func
def quatconj(u):
    # 实部在前,虚部在后
    return ti.Vector([u[0], -u[1], -u[2], -u[3]])


@ti.func
def quatmul(u1, u2):
    tmp1 = u1[0] * u2[0] - u1[1] * u2[1] - u1[2] * u2[2] - u1[3] * u2[3]
    tmp2 = u1[0] * u2[1] + u1[1] * u2[0] + u1[2] * u2[3] - u1[3] * u2[2]
    tmp3 = u1[0] * u2[2] - u1[1] * u2[3] + u1[2] * u2[0] + u1[3] * u2[1]
    tmp4 = u1[0] * u2[3] + u1[1] * u2[2] - u1[2] * u2[1] + u1[3] * u2[0]
    return ti.Vector([tmp1, tmp2, tmp3, tmp4])


@ti.func
def quatnormalize(u):
    return u.normalized()


@ti.func
def quatfromtwovectors(a, b):
    # a -> b的旋转四元数
    v1 = a.normalized()
    v2 = b.normalized()
    cos_theta = v1.dot(v2)

    quat = ti.Vector.zero(ti.f64, 4)
    if cos_theta < -1 + 1e-6:
        cos_theta = max(cos_theta, -1)
        m = ti.Matrix.rows([v1, v2])
        u, s, v = ti.svd(m, ti.f64)             # 奇异值分解得到垂直的特征向量v3
        axis = v[:, 2]
        w2 = (1 + cos_theta) * 0.5
        w = np.sqrt(w2)
        vec = axis * np.sqrt(1 - w2)
        quat[0] = w
        quat[1:] = vec
    else:
        axis = v1.cross(v2)                     # 旋转轴*sin(theta)
        s = ti.sqrt((1 + cos_theta) * 2)        # s=2*cos(theta/2)
        invs = 1 / s
        vec = axis * invs
        w = s * 0.5
        quat[0] = w
        quat[1:] = vec
    
    return quat


@ti.func
def quatrotvec(u, v):
    # 四元数u对向量v进行旋转
    q = ti.Vector([0., v[0], v[1], v[2]])
    q_conj = quatconj(u)
    q_rot = quatmul(quatmul(u, q), q_conj)
    return ti.Vector([q_rot[1], q_rot[2], q_rot[3]])


@ti.data_oriented
class PD1D:
    def __init__(self, length, radius, seed_size:float):
        self.length = length
        self.radius = radius
        self.dt = 1./100
        self.rho = 1.e3
        self.E = 2.e6
        self.mu = 0.45
        self.positional_node_weight = 1.e8
        self.positional_element_weight = 1.e8
        # self.contact_node_weight = 1.e8
        # self.contact_element_weight = 1.e8
        self.contact_node_weight = 0.
        self.contact_element_weight = 0.
        self.dim:int = 3
        self.quat_dim:int = 4
        self.solve_iteration = 201
        # self.infinity = 1.e10
        self.epsilon = 1.e-10
        self.G = self.E / 2 / (1 + self.mu)
        self.section_area = tm.pi * self.radius ** 2

        self.PARTICLE_NUM:int = np.ceil(length / seed_size).astype(int) + 1
        self.ELEMENT_NUM:int = self.PARTICLE_NUM - 1
        self.ANGLE_NUM:int = self.ELEMENT_NUM - 1
        self.l = length / self.ELEMENT_NUM

        node_np, element_np = generate_geometric(self.length, self.PARTICLE_NUM)
        self.cal_element_quat()
        np.savetxt('node_pos_init.csv', node_np, delimiter=',', fmt='%.6f')
        np.savetxt('element.csv', element_np, delimiter=',', fmt='%d')
        np.savetxt('element_quat.csv', element_quat_np, delimiter=',', fmt='%.6f')

        self.node_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_force = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_sn = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init.from_numpy(node_np)
        self.node_pos.from_numpy(node_np)
        self.node_vel.fill(0.)
        self.node_force.fill(0.)

        self.element_indices = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.element_quat = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)            # 单位四元数只取实部和虚部的Y轴部分
        self.element_quat_init = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_new = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_angle_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_torque = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_sn = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_inertia_matrix = ti.Matrix.field(3, 3, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_inv_inertia_matrix = ti.Matrix.field(3, 3, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.ele_inerita_vector = ti.Vector.field(4, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.node_distance_unit = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_delta = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)

        self.cal_element_quat()

        self.element_indices.from_numpy(element_np)
        self.element_quat_init.from_numpy(element_quat_np)
        self.element_quat.from_numpy(element_quat_np)
        self.element_angle_vel.fill(0.)
        # self.element_quat_delta.from_numpy(np.insert(np.ones((self.ELEMENT_NUM, 1)), 1, np.zeros((self.ELEMENT_NUM,)), axis=1))

        self.stretch_weight = 0.
        self.bend_weight = 0.

        self.A_stretch = ti.Matrix.field(self.dim+self.quat_dim, 2*self.dim+self.quat_dim, dtype=ti.f64, shape=())
        self.A_bend = ti.Matrix.field(2*self.quat_dim, 2*self.quat_dim, dtype=ti.f64, shape=())
        self.Bp_stretch = ti.Vector.field(self.dim+self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.Bp_bend = ti.Vector.field(2*self.quat_dim, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim, self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim)
        self.lhs.fill(0.)

        self.stretch_constraint = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim, self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.bend_constraint = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim, self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim))

        self.dBp_stretch_dqu = ti.Matrix.field(self.dim+self.quat_dim, 2*self.dim+self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.dBp_bend_du = ti.Matrix.field(2*self.quat_dim, 2*self.quat_dim, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.AT_dBp_stretch_dqu = ti.Matrix.field(2*self.dim+self.quat_dim, 2*self.dim+self.quat_dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.AT_dBp_bend_du = ti.Matrix.field(2*self.quat_dim, 2*self.quat_dim, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.rhs_dA = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim, self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.z = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim))
        self.dqu_const = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim)
        self.dL_dy = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim)
        self.grad_diffdata = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim)

        # Finite difference for validation
        # self.delta = ti.cast(1e-6, ti.f64)
        self.grad_finite = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*self.quat_dim)

        # 固定尾部的节点和单元
        self.fix_particle_list = [self.PARTICLE_NUM-1]
        self.fix_quaternion_list = [self.ELEMENT_NUM-1]
        # 接触首部的节点和单元
        self.contact_particle_list = [0]
        self.contact_element_list = [0]
        self.contact_par_num = len(self.contact_particle_list)
        self.contact_ele_num = len(self.contact_element_list)
        self.contact_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.contact_par_num)
        self.contact_ang_vel = ti.field(self.dim, dtype=ti.f64, shape=self.contact_ele_num)         # 逆时针为正
        self.contact_vel.fill(0.)
        self.contact_ang_vel.fill(0.)

        # self.contact_vel[0] = ti.Vector([0., 1.e-6]) / self.dt
        self.contact_vel[0] = ti.Vector([0., 0., 0.]) / self.dt
        self.contact_ang_vel[0] = ti.Vector([0., 0., 0.]) / self.dt

        self.node_desired_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.contact_par_num)
        self.element_desired_quat = ti.Vector.field(self.quat_dim, dtype=ti.f64, shape=self.contact_ele_num)

        self.construct_mass()
        self.construct_weight()

        print(f'Particle Num: {self.PARTICLE_NUM}, Element Num: {self.ELEMENT_NUM}, Angle Num: {self.ANGLE_NUM}')
        print(f'node_mass: {self.node_mass[0]}, element_inertia: {self.element_inertia[0]}, stretch_weight: {self.stretch_weight}, bend_weight: {self.bend_weight}')
        print(f'Contact Node: {self.contact_particle_list}, Contact Element: {self.contact_element_list}')
        print(f'Fix Node: {self.fix_particle_list}, Fix Element: {self.fix_quaternion_list}')


    @ti.kernel
    def cal_element_quat(self):
        e3 = ti.Vector([0., 0., 1.])
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element_indices[ele_idx]
            v = self.node_pos[idx2] - self.node_pos[idx1]
            quat = quatfromtwovectors(e3, v)
            self.element_quat[ele_idx] = quat


    def construct_mass(self):
        self.node_mass.fill(tm.pi * self.radius ** 2 * self.length * self.rho / self.PARTICLE_NUM)
        
        J1 = J2 = tm.pi * self.radius ** 4 / 4
        J3 = J1 + J2
        for ele_idx in range(self.ELEMENT_NUM):
            self.ele_inerita_vector[ele_idx] = self.l * self.rho * ti.Vector([0, J1, J2, J3])
            self.ele_inertia_matrix[ele_idx] = self.l * self.rho * ti.Matrix([[J1, 0., 0.], [0., J2, 0.], [0., 0., J3]])
            self.ele_inv_inertia_matrix[ele_idx] = self.ele_inertia_matrix[ele_idx].inverse()


    def construct_weight(self):
        self.stretch_weight = self.E * self.section_area * self.l
        # self.stretch_weight = 1.e5
        self.bend_weight = 2 * self.G * tm.pi * self.radius ** 4 / self.l
        # self.bend_weight = 0.


    @ti.kernel
    def precomputation(self):
        dim = self.dim
        quat_dim = self.quat_dim
        ele_indices_offset = self.PARTICLE_NUM*dim
        I3 = ti.Matrix([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]], ti.f64)
        I4 = ti.Matrix([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], ti.f64)
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.node_mass[q_idx] / self.dt ** 2
        
        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.quat_dim)):
                self.lhs[ele_indices_offset+u_idx*quat_dim+d, ele_indices_offset+u_idx*quat_dim+d] += self.ele_inerita_vector[u_idx][d] / self.dt ** 2
        
        self.A_stretch[None][0:3, 0:3] = -I3 / self.l
        self.A_stretch[None][0:3, 3:6] = I3 / self.l
        self.A_stretch[None][3:7, 6:10] = I4

        self.A_bend[None][0:4, 0:4] = I4
        self.A_bend[None][4:8, 4:8] = I4

        # Stretch Constraint
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element_indices[ele_idx]
            idx1_qu, idx2_qu = idx1*dim, idx2*dim
            u_idx_qu = ele_indices_offset + ele_idx*quat_dim
            q_idx_vec= ti.Vector([idx1_qu, idx1_qu+1, idx1_qu+2, idx2_qu, idx2_qu+1, idx2_qu+2, u_idx_qu, u_idx_qu+1, u_idx_qu+2, u_idx_qu+3])
            A_i = self.A_stretch[None]
            ATA = A_i.transpose() @ A_i
            # print('ATA:', ATA)
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(10, 10)):
                self.lhs[q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]] += self.stretch_weight * ATA[A_row_idx, A_col_idx]
                self.stretch_constraint[q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]] += self.stretch_weight * ATA[A_row_idx, A_col_idx]

        # Bend Constraint
        for angle_idx in range(self.ANGLE_NUM):
            u_idx1, u_idx2 = angle_idx, angle_idx + 1
            u_idx1_qu, u_idx2_qu = ele_indices_offset + u_idx1*dim, ele_indices_offset + u_idx2*dim
            u_idx_vec= ti.Vector([u_idx1_qu, u_idx1_qu+1, u_idx1_qu+2, u_idx1_qu+3, u_idx2_qu, u_idx2_qu+1, u_idx2_qu+2, u_idx2_qu+3])
            A_i = self.A_bend[None]
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(8, 8)):
                self.lhs[u_idx_vec[A_row_idx], u_idx_vec[A_col_idx]] += self.bend_weight * ATA[A_row_idx, A_col_idx]
                self.bend_constraint[u_idx_vec[A_row_idx], u_idx_vec[A_col_idx]] += self.bend_weight * ATA[A_row_idx, A_col_idx]

        for q_idx in ti.static(self.fix_particle_list):
            A_i_eye = I3
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.positional_node_weight * A_i_eye[d, d]

        for u_idx in ti.static(self.fix_quaternion_list):
            A_i_eye = I4
            for d in ti.static(range(self.quat_dim)):
                self.lhs[ele_indices_offset+u_idx*dim+d, ele_indices_offset+u_idx*dim+d] += self.positional_element_weight * A_i_eye[d, d]

        # for q_idx in ti.static(self.contact_particle_list):
        #     A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
        #     for d in ti.static(range(self.dim)):
        #         self.lhs[q_idx*dim+d, q_idx*dim+d] += self.contact_node_weight * A_i_eye[d, d]

        # for u_idx in ti.static(self.contact_element_list):
        #     s_idx = u_idx + self.PARTICLE_NUM
        #     A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
        #     for d in ti.static(range(self.dim)):
        #         self.lhs[s_idx*dim+d, s_idx*dim+d] += self.contact_element_weight * A_i_eye[d, d]


    # @ti.kernel
    # def construct_desired_pos(self):
    #     # for idx in ti.static(range(self.contact_num)):
    #     #     q_idx = self.contact_particle_list[idx]
    #     #     u_idx = self.contact_element_list[idx]

    #     #     self.node_desired_pos[idx] = self.node_pos[q_idx] + self.dt * self.contact_vel[idx]
    #     #     delta_theta = self.dt * self.contact_ang_vel[idx]
    #     #     self.element_desired_quat[idx] = quatmul2d(self.element_quat[u_idx], ti.Vector([tm.cos(delta_theta), tm.sin(delta_theta)]))

    #     for idx in ti.static(range(self.contact_par_num)):
    #         q_idx = self.contact_particle_list[idx]
    #         self.node_desired_pos[idx] = self.node_pos[q_idx] + self.dt * self.contact_vel[idx]

    #     for idx in ti.static(range(self.contact_ele_num)):
    #         u_idx = self.contact_element_list[idx]
    #         delta_theta = self.dt * self.contact_ang_vel[idx]
    #         self.element_desired_quat[idx] = quatmul(self.element_quat[u_idx], ti.Vector([tm.cos(delta_theta/2), tm.sin(delta_theta/2)]))


    @ti.kernel
    def construct_sn(self):
        # 参考soler2018cosserat的更新公式
        for q_idx in range(self.PARTICLE_NUM):
            self.node_sn[q_idx] = self.node_pos[q_idx] + self.dt * self.node_vel[q_idx] \
                                + self.dt**2 * self.node_force[q_idx] / self.node_mass[q_idx]        # shape: (2, 1)

        for u_idx in range(self.ELEMENT_NUM):
            ele_angle_vel_old = self.element_angle_vel[u_idx]
            ele_angle_vel_new = ele_angle_vel_old + self.dt * self.ele_inv_inertia_matrix[u_idx] @ \
                                (self.element_torque[u_idx] - ele_angle_vel_old.cross(self.ele_inertia_matrix[u_idx] @ ele_angle_vel_old))
            angle_vel_quat = ti.Vector([0., ele_angle_vel_new[0], ele_angle_vel_new[1], ele_angle_vel_new[2]], ti.f64)
            delta_quat = self.dt / 2 * quatmul(self.element_quat[u_idx], angle_vel_quat)
            element_sn_tmp = self.element_quat[u_idx] + delta_quat
            self.element_sn[u_idx] = quatnormalize(element_sn_tmp)

        # for idx in ti.static(self.contact_particle_list):
        #     q_idx = self.contact_particle_list[idx]
        #     # self.node_sn[q_idx] += self.contact_vel[0] * self.dt
        #     self.node_sn[q_idx] = self.node_desired_pos[idx]

        # for idx in ti.static(self.contact_element_list):
        #     u_idx = self.contact_element_list[idx]
        #     self.element_sn[u_idx] = self.element_desired_quat[idx]


    @ti.kernel
    def warm_start(self):
        for q_idx in range(self.PARTICLE_NUM):
            # self.node_pos_new[q_idx] = self.node_pos[q_idx]
            self.node_pos_new[q_idx] = self.node_sn[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            # self.element_quat_new[u_idx] = self.element_quat[u_idx]
            self.element_quat_new[u_idx] = self.element_sn[u_idx]


    @ti.kernel
    def local_solve(self):
        e3 = ti.Vector([0., 0., 1.])
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element_indices[ele_idx]
            quat_new = self.element_quat_new[ele_idx]
            distance_vec = (self.node_pos_new[idx2] - self.node_pos_new[idx1])
            # theta = tm.atan2(distance_vec[0], distance_vec[2])
            # print('ele_sn:', quat_new, 'quat theta', 2 * ti.atan2(quat_new[1], quat_new[0]) *180/tm.pi, 'node_sn1:', self.node_pos_new[idx1], 'distance theta:', theta*180/tm.pi)

            d3 = quatrotvec(quat_new, e3)
            u_constaint = quatfromtwovectors(e3, distance_vec)
            self.Bp_stretch[ele_idx][0:3] = d3
            self.Bp_stretch[ele_idx][3:7] = u_constaint
            print('d3:', d3, 'u_constaint:', u_constaint)

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = angle_idx, angle_idx + 1
            u1, u2 = self.element_quat_new[idx1], self.element_quat_new[idx2]
            cos_theta = u1.dot(u2)
            if cos_theta < 0:
                cos_theta = -cos_theta
                u1_inv, u2_inv = -u1, -u2
                u_average1 = (u1 + u2_inv) / ti.sqrt((1 + cos_theta) / 2) / 2
                u_average2 = (u1_inv + u2) / ti.sqrt((1 + cos_theta) / 2) / 2
            else:
                u_average1 = (u1 + u2) / ti.sqrt((1 + cos_theta) / 2) / 2
                u_average2 = u_average1
            # print('u1:', u1, 'u2:', u2, 'u_average:', u_average_unit)
            # u_average = (u1 + u2)
            # u_average_unit = u_average.normalized()
            self.Bp_bend[angle_idx][0:4] = u_average1
            self.Bp_bend[angle_idx][4:8] = u_average2


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        quat_dim = self.quat_dim
        ele_indices_offset = self.PARTICLE_NUM*dim
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] += self.node_mass[q_idx] * self.node_sn[q_idx][d] / self.dt ** 2
                # print('rhs diag:', self.rhs[q_idx*dim+d], 'sn:', self.node_sn[q_idx][d])

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[ele_indices_offset+u_idx*dim+d] += self.ele_inerita_vector[u_idx][d] * self.element_sn[u_idx][d] / self.dt ** 2
                # print('rhs diag:', self.rhs[(u_idx+self.PARTICLE_NUM)*dim+d], 'sn:', self.element_sn[u_idx][d])

        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element_indices[ele_idx]
            idx1_qu, idx2_qu = idx1*dim, idx2*dim
            u_idx = ele_indices_offset + ele_idx*quat_dim
            q_idx_vec = ti.Vector([idx1_qu, idx1_qu+1, idx1_qu+2, idx2_qu, idx2_qu+1, idx2_qu+2, u_idx, u_idx+1, u_idx+2, u_idx+3])
            A_i = self.A_stretch[None]
            AT_Bp_i = self.stretch_weight * A_i.transpose() @ self.Bp_stretch[ele_idx]
            for d in ti.static(range(10)):
                self.rhs[q_idx_vec[d]] += AT_Bp_i[d]

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = angle_idx, angle_idx + 1
            u_idx1_qu, u_idx2_qu = ele_indices_offset + idx1*quat_dim, ele_indices_offset + idx2*quat_dim
            u_idx_vec = ti.Vector([u_idx1_qu, u_idx1_qu+1, u_idx1_qu+2, u_idx1_qu+3, u_idx2_qu, u_idx2_qu+1, u_idx2_qu+2, u_idx2_qu+3])
            A_i = self.A_bend[None]
            AT_Bp_i = self.bend_weight * A_i.transpose() @ self.Bp_bend[angle_idx]
            for d in ti.static(range(8)):
                self.rhs[u_idx_vec[d]] += AT_Bp_i[d]

        for q_idx in ti.static(self.fix_particle_list):
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] += self.positional_node_weight * self.node_pos_init[q_idx][d]

        for u_idx in ti.static(self.fix_quaternion_list):
            u_idx_qu = ele_indices_offset + u_idx*quat_dim
            for d in ti.static(range(self.quat_dim)):
                self.rhs[u_idx_qu+d] += self.positional_element_weight * self.element_quat_init[u_idx][d]

        # for idx in ti.static(range(self.contact_par_num)):
        #     q_idx = self.contact_particle_list[idx]
        #     for d in ti.static(range(self.dim)):
        #         self.rhs[q_idx*dim+d] += self.contact_node_weight * self.node_desired_pos[idx][d]

        # for idx in ti.static(range(self.contact_ele_num)):
        #     u_idx = self.contact_element_list[idx] + self.PARTICLE_NUM
        #     for d in range(2):
        #         self.rhs[u_idx*dim+d] += self.contact_element_weight * self.element_desired_quat[idx][d]


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.node_pos_new[q_idx][d] = sol[q_idx*self.dim+d]

        for u_idx in range(self.ELEMENT_NUM):
            quat_sn_tmp = ti.Vector.zero(ti.f64, 4)
            for d in ti.static(range(self.quat_dim)):
                quat_sn_tmp[d] = sol[self.PARTICLE_NUM*self.dim + u_idx*self.quat_dim + d]
            self.element_quat_new[u_idx] = quatnormalize(quat_sn_tmp)


    @ti.kernel
    def update_vel_pos(self):
        for q_idx in range(self.PARTICLE_NUM):
            self.node_vel[q_idx] = (self.node_pos_new[q_idx] - self.node_pos[q_idx]) / self.dt
            self.node_pos[q_idx] = self.node_pos_new[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            self.element_angle_vel[u_idx] = 2 * quatmul(quatconj(self.element_quat[u_idx]), self.element_quat_new[u_idx]) / self.dt
            self.element_quat[u_idx] = self.element_quat_new[u_idx]


    def construct_L(self):
        # 用于与Finite difference比较进行验证
        for q_idx in self.contact_particle_list:
            self.dL_dqu[q_idx*self.dim + 0] = 1.
            self.dL_dqu[q_idx*self.dim + 0] = 1.


    @ti.kernel
    def partial_p(self):
        I2 = ti.Matrix([[1., 0.], [0., 1.]], ti.f64)
        for ele_idx in range(self.ELEMENT_NUM):
            # 使用pos还是pos_new,实践上是没有区别的
            idx1, idx2 = self.element[ele_idx]
            # quat_new = self.element_quat[ele_idx]
            # distance_vec = (self.node_pos[idx2] - self.node_pos[idx1])
            quat_new = self.element_quat_new[ele_idx]
            distance_vec = (self.node_pos_new[idx2] - self.node_pos_new[idx1])

            distance_vec_norm = distance_vec.norm()
            tan_theta = distance_vec[1] / distance_vec[0]
            u_constraint = ti.Vector([ti.cos(tan_theta/2), ti.sin(tan_theta/2)])

            dq1 = -I2 / distance_vec_norm + distance_vec.outer_product(distance_vec) / tm.pow(distance_vec_norm, 3)
            dq2 = -dq1
            # self.dBp_stretch_dqu[ele_idx][0:2, 0:2] = dq1
            # self.dBp_stretch_dqu[ele_idx][0:2, 2:4] = dq2
            # self.dBp_stretch_dqu[ele_idx][2:4, 4:6] = ti.Matrix([[1., 0.], [0., 1.]], ti.f64)
            # dtheta_dtan = 1 / (1 + tan_theta ** 2)
            # dtan_dxn1 = ti.Vector([distance_vec[1] / distance_vec[0]**2, -1 / distance_vec[0]])
            # dtan_dxn2 = -dtan_dxn1
            # dq1 = ti.Matrix.rows([-])
            self.dBp_stretch_dqu[ele_idx][2:4, 0:2] = dq1 / 2
            self.dBp_stretch_dqu[ele_idx][2:4, 2:4] = dq2 / 2
            self.dBp_stretch_dqu[ele_idx][0:2, 4:6] = ti.Matrix([[2*quat_new[1], 2*quat_new[0]], [2*quat_new[0], -2*quat_new[1]]], ti.f64)
            # self.dBp_stretch_dqu[ele_idx][0:2, 4:6] = I2

            self.dBp_stretch_dqu[ele_idx] = self.stretch_weight * self.dBp_stretch_dqu[ele_idx]
            self.AT_dBp_stretch_dqu[ele_idx] = self.A_stretch[None].transpose() @ self.dBp_stretch_dqu[ele_idx]
            # print('dq1:', dq1)
            # print('A_stretch.T:', self.A_stretch[None].transpose())
            # print('dBp_stretch_dqu:', self.dBp_stretch_dqu[ele_idx])
            # print('AT_dBp_stretch_dqu:', self.AT_dBp_stretch_dqu[ele_idx])

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = angle_idx, angle_idx + 1
            u1, u2 = self.element_quat[idx1], self.element_quat[idx2]

            cos_theta = u1.dot(u2)
            # 没有考虑负数的情况
            tmp = 1 / ti.sqrt((1 + cos_theta) / 2) / 2
            self.dBp_bend_du[angle_idx][0:2, 0:2] = I2 * tmp
            self.dBp_bend_du[angle_idx][0:2, 2:4] = I2 * tmp
            self.dBp_bend_du[angle_idx][2:4, 0:2] = I2 * tmp
            self.dBp_bend_du[angle_idx][2:4, 2:4] = I2 * tmp
            # print('dBp_bend_du:', self.dBp_bend_du[angle_idx])

            self.dBp_bend_du[angle_idx] = self.bend_weight * self.dBp_bend_du[angle_idx]
            self.AT_dBp_bend_du[angle_idx] = self.A_bend[None].transpose() @ self.dBp_bend_du[angle_idx]
            # print('A_bend:', self.A_bend[None])
            # print('AT_dBp_bend_du:', self.AT_dBp_bend_du[angle_idx])

        self.rhs_dA.fill(0.)
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            u_idx = self.PARTICLE_NUM + ele_idx
            qu_idx_vec = ti.Vector([idx1*2, idx1*2+1, idx2*2, idx2*2+1, u_idx*2, u_idx*2+1])
            for row_idx, col_idx in ti.ndrange(6, 6):
                rhs_row_idx, rhs_col_idx = qu_idx_vec[row_idx], qu_idx_vec[col_idx]
                self.rhs_dA[rhs_row_idx, rhs_col_idx] += self.AT_dBp_stretch_dqu[ele_idx][row_idx, col_idx]
            # print('rhs_dA_stretch:', self.AT_dBp_stretch_dqu[ele_idx])

        for angle_idx in range(self.ANGLE_NUM):
            u_idx1, u_idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            u_idx_vec = ti.Vector([u_idx1*2, u_idx1*2+1, u_idx2*2, u_idx2*2+1])
            for row_idx, col_idx in ti.ndrange(4, 4):
                rhs_row_idx, rhs_col_idx = u_idx_vec[row_idx], u_idx_vec[col_idx]
                self.rhs_dA[rhs_row_idx, rhs_col_idx] += self.AT_dBp_bend_du[angle_idx][row_idx, col_idx]
                # self.rhs_dA[rhs_row_idx, rhs_col_idx] += 0.

    
    @ti.kernel
    def construct_dx_const(self):
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.dqu_const[q_idx*self.dim+d] = self.node_mass[q_idx] / self.dt ** 2

        for u_idx in range(self.ELEMENT_NUM):
            u_idx = u_idx + self.PARTICLE_NUM
            for d in range(2):
                self.dqu_const[u_idx*2+d] = self.element_inertia[u_idx] / self.dt ** 2

        for q_i in ti.static(self.contact_particle_list):
            for d in ti.static(range(self.dim)):
                self.dqu_const[q_i*self.dim+d] += self.contact_node_weight

        for u_i in ti.static(self.contact_element_list):
            u_i = u_i + self.PARTICLE_NUM
            for d in ti.static(range(self.dim)):
                self.dqu_const[u_i*2+d] += self.contact_element_weight


    def diff_data(self):
        self.partial_p()
        mass_np = self.node_mass.to_numpy() / self.dt ** 2
        inertia_np = self.element_inertia.to_numpy() / self.dt ** 2
        mass_dim_np = np.repeat(mass_np, self.dim)
        inertia_dim_np = inertia_np.flatten()
        M_np = np.diag(np.concatenate([mass_dim_np, inertia_dim_np]))
        # np.savetxt('M.csv', M_np, delimiter=',', fmt='%.10f')
        A = self.lhs.to_numpy() - self.rhs_dA.to_numpy()
        B = M_np
        for q_idx in self.contact_particle_list:
            for d in range(self.dim):
                B[q_idx*self.dim+d, q_idx*self.dim+d] += self.contact_node_weight
        for u_idx in self.contact_element_list:
            for d in range(self.dim):
                B[(u_idx+self.PARTICLE_NUM)*2+d, (u_idx+self.PARTICLE_NUM)*2+d] += self.contact_element_weight
        np.savetxt('A.csv', A, delimiter=',', fmt='%.10f')
        np.savetxt('dA.csv', self.rhs_dA.to_numpy(), delimiter=',', fmt='%.10f')
        np.savetxt('B.csv', B, delimiter=',', fmt='%.10f')
        # exit()
        dx_dy_np = np.linalg.solve(A, B)
        np.savetxt('dx_dy.csv', dx_dy_np, delimiter=',', fmt='%.12f')
        for q_idx in self.contact_particle_list:
            idx = q_idx * self.dim + 0
            self.grad_diffdata.from_numpy(dx_dy_np[:, idx].reshape(-1, self.dim))
            # np.savetxt(f'grad_diffdata_{q_idx}.csv', dx_dy_np[:, idx].reshape(-1, self.dim), delimiter=',', fmt='%.8f')


    def diff_pd(self, itr_num:ti.i32):
        self.partial_p()
        dA = self.rhs_dA.to_numpy()
        par_L = self.dL_dqu.to_numpy()
        z_np = self.z.to_numpy()
        for itr in range(itr_num):
            rhs_diff_np = dA @ z_np + par_L
            z_new_np = self.pre_fact_lhs_solve(rhs_diff_np)
            z_np = z_new_np
        self.z.from_numpy(z_np)

    
    @ti.kernel
    def cal_ygrad(self):
        for idx in range(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*2):
            self.dL_dy[idx] = self.z[idx] * self.dqu_const[idx]

    
    @ti.kernel
    def cal_grad(self):
        for q_idx in range(self.PARTICLE_NUM):
            self.grad_finite[q_idx] = self.node_pos[q_idx] - self.node_pos_init[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            # self.grad_finite[u_idx+self.PARTICLE_NUM] = quatmul2d(quatconj2d(self.element_quat_init[u_idx]), self.element_quat[u_idx])
            self.grad_finite[u_idx+self.PARTICLE_NUM] = self.element_quat[u_idx] - self.element_quat_init[u_idx]
        

    def substep(self, step_num:ti.i32, frame_name_list:list):
        # self.construct_desired_pos()
        self.construct_sn()
        self.warm_start()
        
        ele_quat_theta1_list = []
        ele_quat_theta2_list = []
        distance_vec1_list = []
        distance_vec2_list = []
        distance_theta1_list = []
        distance_theta2_list = []

        # for itr in range(1):
        for itr in range(self.solve_iteration):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            state_sol = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(state_sol)
            # np.savetxt('rhs.csv', self.rhs.to_numpy(), delimiter=',', fmt='%.10f')
            # print('State Sol:', state_sol)
            distance_vec = state_sol[2:4] - state_sol[0:2]
            theta = tm.atan2(distance_vec[0], distance_vec[1])
            distance_vec2 = state_sol[4:6] - state_sol[2:4]
            theta2 = tm.atan2(distance_vec2[0], distance_vec2[1])

            # print('1:', 'distance vec:', distance_vec, 'quat:', ti.Vector([ti.cos(theta/2), ti.sin(theta/2)]), 'theta:', theta*180/tm.pi)
            # print('2:', 'distance vec:', state_sol[4:6] - state_sol[2:4], 'theta:', theta2*180/tm.pi)
            distance_vec1_list.append(distance_vec)
            distance_vec2_list.append(distance_vec2)
            distance_theta1_list.append(theta*180/tm.pi)
            distance_theta2_list.append(theta2*180/tm.pi)

            ele_quat = state_sol[2*self.PARTICLE_NUM:2*self.PARTICLE_NUM+2]
            quat_theta = 2 * tm.atan2(ele_quat[1], ele_quat[0])
            ele_quat2 = state_sol[2*self.PARTICLE_NUM+2:2*self.PARTICLE_NUM+4]
            quat_theta2 = 2 * tm.atan2(ele_quat2[1], ele_quat2[0])

            # print('1:', 'Ele Quat:', ele_quat, 'Ele Theta:', quat_theta*180/tm.pi)
            # print('2:', 'Ele Quat:', ele_quat2, 'Ele Theta:', quat_theta2*180/tm.pi)
            ele_quat_theta1_list.append(quat_theta*180/tm.pi)
            ele_quat_theta2_list.append(quat_theta2*180/tm.pi)

        data_dict = {
            'distance_vec1': distance_vec1_list,
            'distance_vec2': distance_vec2_list,
            'distance_theta1': distance_theta1_list,
            'distance_theta2': distance_theta2_list,
            'ele_quat_theta1': ele_quat_theta1_list,
            'ele_quat_theta2': ele_quat_theta2_list
        }
        np.savez('local_solve.npz', **data_dict)

        self.update_vel_pos()
        ggui_set = {'window': self.window, 'canvas': self.canvas, 'scene': self.scene}
        frame_name_list = self.gui_show(ggui_set, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=step_num, name_list=frame_name_list)
        return frame_name_list
    

    def show_preset(self):
        """
        Define the data for GGUI
        """
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_NUM)
        self.edge_show = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.edge_show.from_numpy(self.element_indices.to_numpy(dtype=np.int32))


    def preset_gui(self, camera_pos:list, camera_target:list):
        """
        Define the camera position & target
        """
        self.window, self.camera, self.scene = gui_set(pos=camera_pos, target=camera_target)
        self.canvas = self.window.get_canvas()
        self.show_preset()


    def gui_show(self, ggui_set, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=None, name_list=None):
        """
        Show the GGUI
        """
        window, canvas, scene = ggui_set['window'], ggui_set['canvas'], ggui_set['scene']
        if SHOW_FLAG is False:
            return
        scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        scene.ambient_light((0.8, 0.8, 0.8))
        self.node_show.from_numpy(self.node_pos.to_numpy(dtype=np.float32))

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
            
    
    @ti.kernel
    def init_vel(self):
        self.node_vel[0][0] = -0.5
        # self.node_force[0][1] = 9.8 * self.node_mass[0]
        # delta_theta = 5 / self.l * self.dt
        # self.element_quat_delta[0] = ti.Vector([ti.cos(delta_theta / 2), ti.sin(delta_theta / 2)])


def main():
    class MyObj(PD1D):
        def __init__(self, length, radius, seed_size):
            super(MyObj, self).__init__(length, radius, seed_size)

    soft_obj = MyObj(length=1., radius=0.01, seed_size=0.1)
    soft_obj.preset_gui(camera_pos=[0.5, 0.75, 0.3], camera_target=[0.5, 0., 0.3])

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    np.savetxt('lhs.csv', lhs_np, delimiter=',', fmt='%.8f')
    np.savetxt('stretch_constraint.csv', soft_obj.stretch_constraint.to_numpy(), delimiter=',', fmt='%.8f')
    np.savetxt('bend_constraint.csv', soft_obj.bend_constraint.to_numpy(), delimiter=',', fmt='%.8f')
    # soft_obj.init_vel()

    frame_name_list = []

    for step in range(100):
        frame_name_list = soft_obj.substep(step, frame_name_list)
        time.sleep(1)
        print(f'Frame: {step}----------------------')
        print(f'Node Pos 1: {soft_obj.node_pos[0]}', f'Node Vel 1: {soft_obj.node_vel[0]}')
        print(f'Ele Quar 1: {soft_obj.element_quat[0]}', f'Ele Delta 1: {soft_obj.element_quat_delta[0]}')
    exit(0)

    for step in range(1):
        frame_name_list = soft_obj.substep(step, frame_name_list)
        # np.savetxt('rhs.csv', soft_obj.rhs.to_numpy(), delimiter=',', fmt='%.8f')
        # exit()
        soft_obj.diff_data()

    for q_idx in soft_obj.contact_particle_list:
        grad_diffdata_tmp = soft_obj.grad_diffdata.to_numpy()
        np.savetxt(f'grad_diffdata_{q_idx}.csv', grad_diffdata_tmp, delimiter=',', fmt='%.12f')

    # soft_obj.contact_vel[0] = ti.Vector([0., 1.e-6]) / soft_obj.dt
    soft_obj.contact_vel[0] = ti.Vector([1.e-6, 0.]) / soft_obj.dt
    soft_obj.contact_ang_vel[0] = -1.e-6 / soft_obj.l / soft_obj.dt

    # soft_obj.node_vel[0] = ti.Vector([1.e-6, 0.]) / soft_obj.dt
    print('Finite Method:--------------------------------------------------')
    for step in range(1):
        frame_name_list = soft_obj.substep(step, frame_name_list)
        print(f'Frame: {step}----------------------')
        print(f'Node Pos 1: {soft_obj.node_pos[0]}')

    soft_obj.cal_grad()
    np.savetxt('grad_finite.csv', soft_obj.grad_finite.to_numpy()/1.e-6, delimiter=',', fmt='%.12f')

if __name__ == '__main__':
    main()