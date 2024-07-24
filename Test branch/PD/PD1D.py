"""
使用PD仿真1D的绳变形
"""

import numpy as np
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, debug=True)


def generate_node(length, num:int):
    node_np = np.zeros((num, 2), dtype=np.float64)
    node_np[:, 0] = np.linspace(0, length, num)
    return node_np


def generate_element(num:int):
    element_np = np.zeros((num-1, 2), dtype=np.int32)
    for i in range(num-1):
        element_np[i] = [i, i+1]
    return element_np


@ti.func
def theta2rot_matrix(theta):
    return ti.Matrix([[ti.cos(theta), -ti.sin(theta)], [ti.sin(theta), ti.cos(theta)]])


@ti.data_oriented
class PD1D:
    def __init__(self, length, radius, seed_size:float):
        self.length = length
        self.radius = radius
        self.dt = 1./100
        self.rho = 1.e3
        self.E = 1.e6
        self.mu = 0.3
        self.positional_weight = 1.e4
        self.dim = 2
        self.G = self.E / 2 / (1 + self.mu)
        self.section_area = tm.pi * self.radius ** 2

        self.PARTICLE_NUM:int = np.ceil(length / seed_size).astype(int) + 1
        self.ELEMENT_NUM:int = self.PARTICLE_NUM - 1
        self.ANGLE_NUM:int = self.ELEMENT_NUM - 2
        self.l = length / (self.PARTICLE_NUM - 1)

        node_np = generate_node(self.length, self.PARTICLE_NUM)
        element_np = generate_element(self.PARTICLE_NUM)
        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_sn = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = 0.
        self.node_pos_init.from_numpy(node_np)
        self.node_pos.from_numpy(node_np)

        self.element = ti.Vector.field(2, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.element_quat = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)            # 单位四元数
        self.element_quat_init = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_quat_new = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_angle_vel = ti.Vector.field(1, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element_sn = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element.from_numpy(element_np)
        self.element_inertia = ti.Vector([0., 0.])
        self.stretch_weight = 0.
        self.bend_weight = 0.

        self.A_stretch = ti.Matrix.zero(ti.f64, 4, 6)
        self.A_bend = ti.Matrix.zero(ti.f64, 4, 4)
        self.Bp_stretch = ti.Vector.field(4, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.Bp_bend = ti.Vector.field(4, dtype=ti.f64, shape=self.ANGLE_NUM)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM+self.ELEMENT_NUM, self.PARTICLE_NUM+self.ELEMENT_NUM))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM+self.ELEMENT_NUM)

        # 固定尾部的节点和单元
        self.fix_particle_list = [self.PARTICLE_NUM-1]
        self.fix_quaternion_list = [self.PARTICLE_NUM + self.ELEMENT_NUM-1]
        # 接触首部的节点和单元
        self.contact_particle_list = [0]
        self.contact_quaternion_list = [self.PARTICLE_NUM]
        self.contact_num = len(self.contact_particle_list)
        self.contact_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)
        self.contact_ang_vel = ti.Vector.field(1, dtype=ti.f64, shape=self.contact_num)         # 逆时针为正

        self.node_desired_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)
        self.element_desired_quat = ti.Vector.field(2, dtype=ti.f64, shape=self.contact_num)

        self.construct_mass()
        self.construct_weight()


    def construct_mass(self):
        self.node_mass = tm.pi * self.radius ** 2 * self.l * self.rho
        
        J1 = J2 = tm.pi * self.radius ** 4 / 4
        J3 = J1 + J2
        self.element_inertia = self.l * self.rho * ti.Vector([0., J1])

    
    def construct_weight(self):
        self.stretch_weight = self.E * self.section_area * self.l
        self.bend_weight = 2 * self.G * tm.pi * self.radius ** 4 / self.l

    
    @ti.kernel
    def precomputation(self):
        dim = self.dim
        for q_idx in ti.static(range(self.PARTICLE_NUM)):
            self.lhs[q_idx*dim, q_idx*dim] = self.node_mass / self.dt ** 2
            self.lhs[q_idx*dim+1, q_idx*dim+1] = self.node_mass / self.dt ** 2
        
        for u_idx in ti.static(range(self.PARTICLE_NUM, self.PARTICLE_NUM+self.ELEMENT_NUM)):
            self.lhs[u_idx*dim, u_idx*dim] = self.element_inertia[0] / self.dt ** 2
            self.lhs[u_idx*dim+1, u_idx*dim+1] = self.element_inertia[1] / self.dt ** 2
        
        for d in ti.static(range(dim)):
            self.A_stretch[d, d] = -1. / self.l
        for d in ti.static(range(dim)):
            self.A_stretch[d, d+2] = 1. / self.l
        for d in ti.static(range(dim)):
            self.A_stretch[d+2, d+4] = 1.

        for d in ti.static(range(dim+dim)):
            self.A_bend[d, d] = tm.sqrt(2)

        # Stretch Constraint
        for ele_idx in ti.static(range(self.ELEMENT_NUM)):
            idx1, idx2 = self.element[ele_idx]
            idx1_x, idx1_y = idx1*dim, idx1*dim+1
            idx2_x, idx2_y = idx2*dim, idx2*dim+1
            u_idx_s, u_idx_y = (self.PARTICLE_NUM+ele_idx)*dim, (self.PARTICLE_NUM+ele_idx)*dim+1       # 只取四元数的部分
            q_idx_vec= ti.Vector([idx1_x, idx1_y, idx2_x, idx2_y, u_idx_s, u_idx_y])
            A_i = self.A_stretch
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(6, 6)):
                self.lhs[q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]] += self.stretch_weight * ATA[A_row_idx, A_col_idx]

        # Bend Constraint
        for angle_idx in ti.static(range(self.ANGLE_NUM)):
            idx1, idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            idx1_s, idx1_y = idx1*dim, idx1*dim+1
            idx2_s, idx2_y = idx2*dim, idx2*dim+1
            u_idx_vec= ti.Vector([idx1_s, idx1_y, idx2_s, idx2_y])
            A_i = self.A_bend
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(4, 4)):
                self.lhs[u_idx_vec[A_row_idx], u_idx_vec[A_col_idx]] += self.bend_weight * ATA[A_row_idx, A_col_idx]

        for q_idx in self.fix_particle_list:
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(dim)):
                self.lhs[q_idx*dim+d, q_idx*dim+d] += self.positional_weight * A_i_eye[d, d]

        for u_idx in self.fix_quaternion_list:
            A_i_eye = ti.Matrix([[1., 0.], [0., 1.]])
            for d in ti.static(range(dim)):
                self.lhs[u_idx*dim+d, u_idx*dim+d] += self.positional_weight * A_i_eye[d, d]

        
    @ti.kernel
    def construct_desired_pos(self):
        for idx in ti.static(range(self.contact_num)):
            q_idx = self.contact_particle_list[idx]
            u_idx = self.contact_quaternion_list[idx]

            self.node_desired_pos[idx] = self.node_pos[q_idx] + self.dt * self.contact_vel[idx]
            rot_matrix = theta2rot_matrix(self.dt * self.contact_ang_vel[idx])
            self.element_desired_quat[idx] = rot_matrix @ self.element_quat[u_idx]

    
    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(dim)):
                self.node_sn[q_idx*dim + d] = self.node_pos[q_idx][d] + self.dt * self.node_vel[q_idx][d]

        for u_idx in range(self.ELEMENT_NUM):
            rot_matrix = theta2rot_matrix(self.dt * self.element_angle_vel[u_idx])
            quat_tmp = rot_matrix @ self.element_quat[u_idx]
            for d in range(2):
                self.element_sn[u_idx*dim + d] = quat_tmp[d]

        
    @ti.kernel
    def warm_start(self):
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.node_pos_new[q_idx][d] = self.node_pos[q_idx][d]

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(self.dim)):
                self.element_quat_new[u_idx][d] = self.element_quat[u_idx][d]


    @ti.kernel
    def local_solve(self):
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            distance_vec = (self.node_pos[idx2] - self.node_pos[idx1]).normalized()
            d3 = distance_vec
            # 确保element的方向的d3的方向一致,可以拿出来单独作为一个constraint
            u = distance_vec            
            self.Bp_stretch[ele_idx] = ti.Vector([d3[0], d3[1], u[0], u[1]])


        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            u1, u2 = self.element_quat[idx1], self.element_quat[idx2]
            u_average = (u1 + u2)
            u_average = u_average.normalized()
            self.Bp_bend[angle_idx] = ti.Vector([u_average[0], u_average[1], u_average[0], u_average[1]])


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        for q_idx in range(self.PARTICLE_NUM):
            for d in ti.static(range(dim)):
                self.rhs[q_idx*dim+d] = self.node_mass * self.node_sn[q_idx*dim+d] / self.dt ** 2

        for u_idx in range(self.ELEMENT_NUM):
            for d in ti.static(range(dim)):
                self.rhs[(u_idx+self.PARTICLE_NUM)*dim+d] = self.element_inertia[d] * self.element_sn[u_idx*dim+d] / self.dt ** 2

        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2 = self.element[ele_idx]
            q_idx_vec = ti.Vector([idx1*dim, idx1*dim+1, idx2*dim, idx2*dim+1])
            AT_Bp_i = self.stretch_weight * self.A_stretch.transpose() @ self.Bp_stretch[ele_idx]
            for d in ti.static(range(4)):
                self.rhs[q_idx_vec[d]] += AT_Bp_i[d]

        for angle_idx in range(self.ANGLE_NUM):
            idx1, idx2 = self.PARTICLE_NUM + angle_idx, self.PARTICLE_NUM + angle_idx + 1
            u_idx_vec = ti.Vector([idx1*dim, idx1*dim+1, idx2*dim, idx2*dim+1])
            AT_Bp_i = self.bend_weight * self.A_bend.transpose() @ self.Bp_bend[angle_idx]
            for d in ti.static(range(4)):
                self.rhs[u_idx_vec[d]] += AT_Bp_i[d]

        for q_idx in ti.static(self.fix_particle_list):
            for d in ti.static(range(dim)):
                self.rhs[q_idx*dim+d] += self.positional_weight * self.node_pos_init[q_idx][d]

        for u_idx in ti.static(self.fix_quaternion_list):
            for d in ti.static(range(dim)):
                self.rhs[u_idx*dim+d] += self.positional_weight * self.element_quat_init[u_idx-self.PARTICLE_NUM][d]

        for idx in ti.static(self.contact_num):
            q_idx = self.contact_particle_list[idx]
            for d in ti.static(range(dim)):
                self.rhs[q_idx*dim+d] += self.positional_weight * self.node_desired_pos[idx][d]

        for idx in ti.static(self.contact_num):
            u_idx = self.contact_quaternion_list[idx]
            for d in ti.static(range(dim)):
                self.rhs[u_idx*dim+d] += self.positional_weight * self.element_desired_quat[idx][d]



def main():



if __name__ == '__main__':
    main()