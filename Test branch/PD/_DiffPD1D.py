"""
DiffPD的一维版本,基于Cosserat杆理论,只有弯曲和拉伸
created at 2024-07-27 by hsy
"""

import os
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)
from GGUI import * 


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
def quatconj2d(u):
    # 实部在前,虚部在后
    return ti.Vector([u[0], -u[1]])


@ti.func
def quatmul2d(u1, u2):
    return ti.Vector([u1[0]*u2[0]-u1[1]*u2[1], u1[0]*u2[1]+u1[1]*u2[0]])


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
        self.contact_node_weight = 1.e6
        self.contact_element_weight = 1.e3
        self.dim:int = 2
        self.solve_iteration = 20
        # self.infinity = 1.e10
        self.epsilon = 1.e-5
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
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init.from_numpy(node_np)
        self.node_pos.from_numpy(node_np)
        self.node_vel.fill(0.)
        self.node_force.fill(0.)

        self.element = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.element_quat = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)            # 单位四元数只取实部和虚部的Y轴部分
        self.element_quat_init = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_new = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_delta = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_sn = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.node_distance_unit = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element.from_numpy(element_np)
        self.element_quat_init.from_numpy(element_quat_np)
        self.element_quat.from_numpy(element_quat_np)
        self.element_quat_delta.from_numpy(np.insert(np.ones((self.ELEMENT_NUM, 1)), 1, np.zeros((self.ELEMENT_NUM,)), axis=1))
        self.element_inertia = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)

        self.stretch_weight = 0.
        self.bend_weight = 0.

        self.A_stretch = ti.Matrix.field(4, 6, dtype=ti.f64, shape=())
        self.A_bend = ti.Matrix.field(4, 4, dtype=ti.f64, shape=())
        self.Bp_stretch = ti.Vector.field(4, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.Bp_bend = ti.Vector.field(4, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*2+self.ELEMENT_NUM*2, self.PARTICLE_NUM*2+self.ELEMENT_NUM*2))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*2+self.ELEMENT_NUM*2)
        self.lhs.fill(0.)

        self.dBp_stretch_dqu = ti.Matrix.field(4, 6, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.dBp_bend_du = ti.Matrix.field(4, 4, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.AT_dBp_stretch_dqu = ti.Matrix.field(6, 6, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.AT_dBp_bend_du = ti.Matrix.field(4, 4, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.rhs_dA = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*2, self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*2))
        self.z = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*2))
        self.dqu_const = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*2)
        self.dL_dy = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim+self.ELEMENT_NUM*2)
        self.grad_diffdata = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM+self.ELEMENT_NUM)

        # Finite difference for validation
        self.delta = ti.cast(1e-6, ti.f64)
        self.grad_finite = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM+self.ELEMENT_NUM)

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

        self.contact_vel[0] = ti.Vector([0., 1.e-6]) / self.dt

        self.node_desired_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)
        self.element_desired_quat = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)

        self.construct_mass()
        self.construct_weight()

        print(f'Particle Num: {self.PARTICLE_NUM}, Element Num: {self.ELEMENT_NUM}, Angle Num: {self.ANGLE_NUM}')
        print(f'node_mass: {self.node_mass[0]}, element_inertia: {self.element_inertia[0]}, stretch_weight: {self.stretch_weight}, bend_weight: {self.bend_weight}')
        print(f'Contact Node: {self.contact_particle_list}, Contact Element: {self.contact_element_list}')
        print(f'Fix Node: {self.fix_particle_list}, Fix Element: {self.fix_quaternion_list}')


    def construct_mass(self):
        self.node_mass.fill(tm.pi * self.radius ** 2 * self.l * self.rho)
        
        J1 = J2 = tm.pi * self.radius ** 4 / 4
        J3 = J1 + J2
        for ele_idx in range(self.ELEMENT_NUM):
            self.element_inertia[ele_idx] = self.l * self.rho * ti.Vector([0., J1])


    def construct_weight(self):
        self.stretch_weight = self.E * self.section_area * self.l
        self.bend_weight = 2 * self.G * tm.pi * self.radius ** 4 / self.l

    
    @ti.kernel
    def precomputation(self):
        dim = self.dim
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] = self.node_mass[q_idx] / self.dt ** 2
        
        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.lhs[(u_idx+self.PARTICLE_NUM)*dim+d, (u_idx+self.PARTICLE_NUM)*dim+d] = self.element_inertia[u_idx][d] / self.dt ** 2
        
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

        for q_idx in ti.static(self.contact_particle_list):
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.contact_node_weight * A_i_eye[d, d]

        for u_idx in ti.static(self.contact_element_list):
            s_idx = u_idx + self.PARTICLE_NUM
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(self.dim)):
                self.lhs[s_idx*dim+d, s_idx*dim+d] += self.contact_element_weight * A_i_eye[d, d]


    def construct_L(self):
        # 用于与Finite difference比较进行验证
        for q_idx in self.contact_particle_list:
            self.dL_dqu[q_idx*self.dim + 0] = 1.


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
            # 不需要显式计算角速度，可以只计算单位时间步长下的姿态变化量
            delta_quat = self.element_quat_delta[u_idx]
            self.element_sn[u_idx] = quatmul2d(self.element_quat[u_idx], delta_quat)            # shape: (2, 1)


    @ti.kernel
    def warm_start(self):
        for q_idx in range(self.PARTICLE_NUM):
            self.node_pos_new[q_idx] = self.node_pos[q_idx]

        for u_idx in range(self.ELEMENT_NUM):
            self.element_quat_new[u_idx] = self.element_quat[u_idx]


    @ti.kernel
    def local_solve(self):
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            quat_new = self.element_quat_new[ele_idx]
            distance_vec = (self.node_pos_new[idx2] - self.node_pos_new[idx1])
            distance_vec_unit = distance_vec.normalized()

            d3 = quat_new
            u_constaint = distance_vec_unit
            self.Bp_stretch[ele_idx] = ti.Vector([d3[0], d3[1], u_constaint[0], u_constaint[1]])

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = angle_idx, angle_idx + 1
            u1, u2 = self.element_quat_new[idx1], self.element_quat_new[idx2]
            u_average = (u1 + u2)
            u_average_unit = u_average.normalized()
            self.Bp_bend[angle_idx] = ti.Vector([u_average_unit[0], u_average_unit[1], u_average_unit[0], u_average_unit[1]])


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*dim+d] = self.node_mass[q_idx] * self.node_sn[q_idx][d] / self.dt ** 2

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[(u_idx+self.PARTICLE_NUM)*dim+d] = self.element_inertia[u_idx][d] * self.element_sn[u_idx][d] / self.dt ** 2

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
                self.rhs[q_idx*dim+d] += self.contact_node_weight * self.node_desired_pos[idx][d]

        for idx in ti.static(range(self.contact_num)):
            u_idx = self.contact_element_list[idx] + self.PARTICLE_NUM
            for d in ti.static(range(self.dim)):
                self.rhs[u_idx*dim+d] += self.contact_element_weight * self.element_desired_quat[idx][d]


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
            self.element_quat_delta[u_idx] = quatmul2d(quatconj2d(self.element_quat[u_idx]), self.element_quat_new[u_idx])
            self.element_quat[u_idx] = self.element_quat_new[u_idx]
            idx1, idx2 = self.element[u_idx]
            self.node_distance_unit[u_idx] = (self.node_pos_new[idx2] - self.node_pos_new[idx1]).normalized()


    @ti.kernel
    def partial_p(self):
        I2 = ti.Matrix([[1., 0.], [0., 1.]], ti.f64)
        for ele_idx in range(self.ELEMENT_NUM):
            # 使用pos还是pos_new,实践上是没有区别的
            idx1, idx2 = self.element[ele_idx]
            # quat_new = self.element_quat[ele_idx]
            distance_vec = (self.node_pos[idx2] - self.node_pos[idx1])
            distance_vec_norm = distance_vec.norm()

            dq1 = -I2 / distance_vec_norm + distance_vec.outer_product(distance_vec) / tm.pow(distance_vec_norm, 3)
            dq2 = -dq1
            self.dBp_stretch_dqu[ele_idx][0:2, 0:2] = dq1
            self.dBp_stretch_dqu[ele_idx][0:2, 2:4] = dq2
            self.dBp_stretch_dqu[ele_idx][2:4, 4:6] = ti.Matrix([[1., 0.], [0., 1.]], ti.f64)

            self.dBp_stretch_dqu[ele_idx] = self.stretch_weight * self.dBp_stretch_dqu[ele_idx]
            self.AT_dBp_stretch_dqu[ele_idx] = self.A_stretch[None].transpose() @ self.dBp_stretch_dqu[ele_idx]

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = angle_idx, angle_idx + 1
            u1, u2 = self.element_quat[idx1], self.element_quat[idx2]
            u_average = (u1 + u2)
            u_average_norm = u_average.norm()
            u_average_unit = u_average.normalized()

            du_star = ti.Vector([-u_average_unit[1], u_average_unit[0]]) / 4
            for d in range(2):
                if ti.abs(u1[d]) > self.epsilon:
                    pass
                else:
                    u1[d] = self.epsilon if u1[d] >= 0 else -self.epsilon
                if ti.abs(u2[d]) > self.epsilon:
                    pass
                else:
                    u2[d] = self.epsilon if u2[d] >= 0 else -self.epsilon
            # print('u1:', u1, 'u2:', u2)
            du1 = 2 / ti.Vector([-u1[1], u1[0]])
            du2 = 2 / ti.Vector([-u2[1], u2[0]])
            tmp1 = du_star.outer_product(du1)
            tmp2 = du_star.outer_product(du2)
            self.dBp_bend_du[angle_idx][0:2, 0:2] = tmp1
            self.dBp_bend_du[angle_idx][0:2, 2:4] = tmp2
            self.dBp_bend_du[angle_idx][2:4, 0:2] = tmp1
            self.dBp_bend_du[angle_idx][2:4, 2:4] = tmp2
            # print('du1:', du1)

            self.dBp_bend_du[angle_idx] = self.bend_weight * self.dBp_bend_du[angle_idx]
            self.AT_dBp_bend_du[angle_idx] = self.A_bend[None].transpose() @ self.dBp_bend_du[angle_idx]

        self.rhs_dA.fill(0.)
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            u_idx = self.PARTICLE_NUM + ele_idx
            qu_idx_vec = ti.Vector([idx1*2, idx1*2+1, idx2*2, idx2*2+1, u_idx*2, u_idx*2+1])
            for row_idx, col_idx in ti.ndrange(6, 6):
                rhs_row_idx, rhs_col_idx = qu_idx_vec[row_idx], qu_idx_vec[col_idx]
                self.rhs_dA[rhs_row_idx, rhs_col_idx] += self.AT_dBp_stretch_dqu[ele_idx][row_idx, col_idx]

        for angle_idx in range(self.ANGLE_NUM):
            u_idx1, u_idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            u_idx_vec = ti.Vector([u_idx1*2, u_idx1*2+1, u_idx2*2, u_idx2*2+1])
            for row_idx, col_idx in ti.ndrange(4, 4):
                rhs_row_idx, rhs_col_idx = u_idx_vec[row_idx], u_idx_vec[col_idx]
                self.rhs_dA[rhs_row_idx, rhs_col_idx] += self.AT_dBp_bend_du[angle_idx][row_idx, col_idx]

    
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
        np.savetxt('A.csv', A, delimiter=',', fmt='%.10f')
        np.savetxt('dA.csv', self.rhs_dA.to_numpy(), delimiter=',', fmt='%.10f')
        np.savetxt('B.csv', B, delimiter=',', fmt='%.10f')
        # exit()
        dx_dy_np = np.linalg.solve(A, B)
        for q_idx in self.contact_particle_list:
            idx = q_idx * self.dim + 1
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
            self.grad_finite[u_idx+self.PARTICLE_NUM] = quatmul2d(quatconj2d(self.element_quat_init[u_idx]), self.element_quat[u_idx])
        

    def substep(self, step_num:ti.i32, frame_name_list:list):
        self.construct_desired_pos()
        self.construct_sn()
        self.warm_start()
        for itr in range(self.solve_iteration):
        # for itr in range(1):
            self.local_solve()
            self.construct_rhs()
            # np.savetxt('rhs.csv', self.rhs.to_numpy(), delimiter=',', fmt='%.10f')
            # exit()
            rhs_np = self.rhs.to_numpy()
            state_sol = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(state_sol)
            self.quat_normalize()

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
        self.edge_show.from_numpy(self.element.to_numpy(dtype=np.int32))


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
            
    
    @ti.kernel
    def init_vel(self):
        # self.node_vel[0][1] = 10.
        self.node_force[0][1] = 9.8 * self.node_mass[0] * 20


def main():
    class MyObj(PD1D):
        def __init__(self, length, radius, seed_size):
            super(MyObj, self).__init__(length, radius, seed_size)

    soft_obj = MyObj(length=1., radius=0.01, seed_size=0.05)
    soft_obj.preset_gui(camera_pos=[0.5, 0.75, 0.3], camera_target=[0.5, 0., 0.3])

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    np.savetxt('lhs.csv', lhs_np, delimiter=',', fmt='%.8f')

    frame_name_list = []

    for step in range(1):
        frame_name_list = soft_obj.substep(step, frame_name_list)
        # np.savetxt('rhs.csv', soft_obj.rhs.to_numpy(), delimiter=',', fmt='%.8f')
        # exit()
        soft_obj.diff_data()
        print(f'Frame: {step}----------------------')
        print(f'Node Pos 1: {soft_obj.node_pos[0]}')
    for q_idx in soft_obj.contact_particle_list:
        grad_diffdata_tmp = soft_obj.grad_diffdata.to_numpy()
        np.savetxt(f'grad_diffdata_{q_idx}.csv', grad_diffdata_tmp, delimiter=',', fmt='%.10f')

    soft_obj.cal_grad()
    np.savetxt('grad_finite.csv', soft_obj.grad_finite.to_numpy()/1.e-6, delimiter=',', fmt='%.10f')

if __name__ == '__main__':
    main()