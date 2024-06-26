"""
This file simulates deformation by PD method in 3D
参考"_PDStrain.py"更改的三维版本,其中`strain_weight`和`volume_weight`和之前的文件都不同.
Positional Constraint改为设置'weight'的方式,而不是调大质量
"""

import time
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)
import numpy as np
from scipy import sparse
from GenMsh import generate_msh


def cal_tet_volume(vertices):
    # 计算四面体体积
    v0, v1, v2, v3 = vertices
    volume = np.abs(np.dot(v0 - v3, np.cross(v1 - v3, v2 - v3))) / 6.0
    return volume


def read_msh_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nodes = []
    cells = []
    is_node_section = False
    is_cell_section = False

    for line in lines:
        if line.strip() == "$NOD":
            is_node_section = True
            continue
        if line.strip() == "$ENDNOD":
            is_node_section = False
            continue
        if line.strip() == "$ELM":
            is_cell_section = True
            continue
        if line.strip() == "$ENDELM":
            is_cell_section = False
            continue

        if is_node_section:
            parts = line.strip().split()
            if len(parts) == 4:
                index, x, y, z = parts
                nodes.append([float(x), float(y), float(z)])

        if is_cell_section:
            parts = line.strip().split()
            if len(parts) > 4:
                index = parts[0]
                cell_nodes = parts[5:]
                cells.append([int(node) for node in cell_nodes])

    nodes_array = np.array(nodes)
    cells_array = np.array(cells)

    return nodes_array, cells_array


def get_tetrahedron_edges(tet_indices):
    # 四面体的6条边
    edge_combinations = [(0, 1), (0, 2), (0, 3),
                         (1, 2), (1, 3),
                         (2, 3)]
    
    # 使用集合来存储唯一的边
    unique_edges = set()
    
    for tet in tet_indices:
        for i, j in edge_combinations:
            # 确保边的顶点索引是有序的
            edge = tuple(sorted([tet[i], tet[j]]))
            unique_edges.add(edge)
    
    # 将集合转换为numpy数组
    return np.array(list(unique_edges))


@ti.data_oriented
class SoftObject:
    def __init__(self, shape, seed_size, mesh_file):
        self.shape = shape
        self.seed_size = seed_size
        self.dt = 1./100
        self.rho = 1.e3
        self.volume_sum = ti.field(ti.f64, shape=())
        self.positional_weight = 1.e8
        self.solve_iteration = 10
        self.E, self.nu = 5.e4, 0.45
        self.dim = len(shape)
        self.mu , self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        node_np, element_np, volume_np = self.load_msh(mesh_file)
        self.edge_np = get_tetrahedron_edges(element_np)
        np.savetxt('node_np.csv', node_np, fmt='%f', delimiter=',')
        np.savetxt('element_np.csv', element_np, fmt='%d', delimiter=',')
        np.savetxt('volume_np.csv', volume_np, fmt='%f', delimiter=',')

        self.PARTICLE_NUM = node_np.shape[0]
        self.EDGE_NUM = self.edge_np.shape[0]
        self.ELEMENT_NUM = element_np.shape[0]

        # node_pos: 3D position of each node in time step
        self.node_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_init_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        # For local solver
        self.node_pos_new = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_init_pos.from_numpy(node_np.astype(np.float64))
        self.node_pos.from_numpy(node_np.astype(np.float64))

        self.element = ti.Vector.field(4, dtype=ti.i32, shape=self.ELEMENT_NUM)
        self.element_volume = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.strain_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.volume_weight = ti.field(dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.element.from_numpy(element_np.astype(np.int32))
        self.element_volume.from_numpy(volume_np.astype(np.float64))

        self.B = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        # 单元的A矩阵
        self.A = ti.Matrix.field(self.dim**2, (self.dim+1)*self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM)
        self.Bp = ti.Matrix.field(self.dim, self.dim, dtype=ti.f64, shape=self.ELEMENT_NUM*2)           # 此处的`2`指约束个数

        self.sn = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim)
        self.lhs = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim, self.PARTICLE_NUM*self.dim))
        # 全局的A矩阵
        self.SA_strain = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim, self.PARTICLE_NUM*self.dim))
        self.SA_volume = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim, self.PARTICLE_NUM*self.dim))
        self.A_poistional = ti.field(dtype=ti.f64, shape=(self.PARTICLE_NUM*self.dim, self.PARTICLE_NUM*self.dim))
        self.rhs = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM*self.dim)

        self.fix_particle_list = self.fix_particle_No()
        self.bottom_particles_list, self.BOTTOM_NUM = self.extract_bottom_particles()
        self.contact_particles_list = self.contact_particles_indice()
        self.contact_num = len(self.contact_particles_list)
        self.contact_vel = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.contact_num)
        self.node_desired_pos = ti.Vector.field(self.dim, dtype=ti.f64, shape=self.contact_num)
        contact_vel_np = np.zeros((self.contact_num, self.dim))
        contact_vel_np[:, 0] = 0.005
        self.contact_vel.from_numpy(contact_vel_np)
        self.sample_particles_list = [132]

        self.construct_mass()
        self.construct_B()
        self.construct_weight()

        # Print the information
        print('Particle number:', self.PARTICLE_NUM)
        print('Element number:', self.ELEMENT_NUM)
        print('Contact particles number:', self.contact_num)


    @staticmethod
    def load_msh(file_path):
        nodes_array, elements_array = read_msh_file(file_path)
        elements_array = elements_array - 1
        # 打印节点和单元信息
        # print(f"节点信息：{nodes_array.shape[0]}")
        # print(f"单元信息：{elements_array.shape[0]}")

        volumes_list = []
        for ele in elements_array:
            tetra_nodes = nodes_array[ele]
            volume_tmp = cal_tet_volume(tetra_nodes)
            volumes_list.append(volume_tmp)

        return nodes_array, elements_array, np.array(volumes_list)


    def fix_particle_No(self):
        """
        Find the particle No. of fix constraint
        """
        fix_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)
        L = self.shape[0]
        H = self.shape[1]
        W = self.shape[2]
        seed_size = self.seed_size

        @ti.kernel
        def cal_fix_constraint(L: float, H:float, W: float, seed_size: float):
            EPS = seed_size / 3
            for idx in range(self.PARTICLE_NUM):
                x_temp = self.node_init_pos[idx].x
                y_temp = self.node_init_pos[idx].y
                z_temp = self.node_init_pos[idx].z  # 3D dimension
                # flag_temp = (x_temp > L - EPS or x_temp < 0. + EPS) and (z_temp > W/2 - EPS or z_temp < -W/2 + EPS)
                fix_flag_temp = (x_temp < 0. + EPS)# or (z_temp > W/2 - EPS)
                fix_flag[idx] = fix_flag_temp

        cal_fix_constraint(L, H, W, seed_size)
        fix_particles_set = set()
        for i in range(self.PARTICLE_NUM):
            if fix_flag[i]:
                fix_particles_set.add(i)
        fix_particles_list = list(fix_particles_set)

        return fix_particles_list


    def contact_particles_indice(self):
        """
        Find the indice of contact particles
        """
        contact_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)
        L = self.shape[0]
        H = self.shape[1]
        W = self.shape[2]
        seed_size = self.seed_size

        @ti.kernel
        def cal_contact_constraint(L: float, H:float, W: float, seed_size: float):
            EPS = seed_size / 3
            for idx in range(self.PARTICLE_NUM):
                x_temp = self.node_init_pos[idx].x
                y_temp = self.node_init_pos[idx].y
                z_temp = self.node_init_pos[idx].z
                contact_flag_temp = (x_temp > L - EPS)
                contact_flag[idx] = contact_flag_temp
            
        cal_contact_constraint(L, H, W, seed_size)
        contact_particles_set = set()
        for i in range(self.PARTICLE_NUM):
            if contact_flag[i]:
                contact_particles_set.add(i)
        contact_particles_list = list(contact_particles_set)

        return contact_particles_list


    @ti.kernel
    def construct_mass(self):
        # mass_tmp = self.rho * self.volume_sum / self.PARTICLE_NUM
        # self.node_mass.fill(mass_tmp)

        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2, idx3, idx4 = self.element[ele_idx]
            self.node_mass[idx1] += self.element_volume[ele_idx] * self.rho / 4
            self.node_mass[idx2] += self.element_volume[ele_idx] * self.rho / 4
            self.node_mass[idx3] += self.element_volume[ele_idx] * self.rho / 4
            self.node_mass[idx4] += self.element_volume[ele_idx] * self.rho / 4


    @ti.kernel
    def construct_B(self):
        for i in range(self.ELEMENT_NUM):
            idx1, idx2, idx3, idx4 = self.element[i]
            a, b, c, d = self.node_init_pos[idx1], self.node_init_pos[idx2], \
                         self.node_init_pos[idx3], self.node_init_pos[idx4]
            B_i_inv = ti.Matrix.cols([b - a, c - a, d - a])
            self.B[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_weight(self):
        for i in range(self.ELEMENT_NUM):
            self.strain_weight[i] = self.mu * self.element_volume[i]
            self.volume_weight[i] = self.element_volume[i] * (self.lam/2 + self.mu/self.dim) 


    @ti.kernel
    def precomputation(self):
        element_num = self.ELEMENT_NUM
        dim = self.dim

        for i in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.lhs[i*dim + d, i*dim + d] = self.node_mass[i] / self.dt**2

        for i in range(element_num):
            B_i = self.B[i]
            b11, b12, b13 = B_i[0,:]
            b21, b22, b23 = B_i[1,:]
            b31, b32, b33 = B_i[2,:]

            # Strain and Volume straint has the same A matrix
            for j in range(dim):
                self.A[i][j*dim+0, 0*dim+j] = -b11-b21-b31
                self.A[i][j*dim+0, 1*dim+j] = b11
                self.A[i][j*dim+0, 2*dim+j] = b21
                self.A[i][j*dim+0, 3*dim+j] = b31
                self.A[i][j*dim+1, 0*dim+j] = -b12-b22-b32
                self.A[i][j*dim+1, 1*dim+j] = b12
                self.A[i][j*dim+1, 2*dim+j] = b22
                self.A[i][j*dim+1, 3*dim+j] = b32
                self.A[i][j*dim+2, 0*dim+j] = -b13-b23-b33
                self.A[i][j*dim+2, 1*dim+j] = b13
                self.A[i][j*dim+2, 2*dim+j] = b23
                self.A[i][j*dim+2, 3*dim+j] = b33

        for ele_idx in range(element_num):
            idx1, idx2, idx3, idx4 = self.element[ele_idx]
            idx1_x, idx1_y, idx1_z = idx1 * self.dim, idx1 * self.dim + 1, idx1 * self.dim + 2
            idx2_x, idx2_y, idx2_z = idx2 * self.dim, idx2 * self.dim + 1, idx2 * self.dim + 2
            idx3_x, idx3_y, idx3_z = idx3 * self.dim, idx3 * self.dim + 1, idx3 * self.dim + 2
            idx4_x, idx4_y, idx4_z = idx4 * self.dim, idx4 * self.dim + 1, idx4 * self.dim + 2
            q_idx_vec = ti.Vector([idx1_x, idx1_y, idx1_z, idx2_x, idx2_y, idx2_z,
                                   idx3_x, idx3_y, idx3_z, idx4_x, idx4_y, idx4_z])

            strain_weight = self.strain_weight[ele_idx]
            volume_weight = self.volume_weight[ele_idx]
            A_i = self.A[ele_idx]
            ATA = A_i.transpose() @ A_i
            for A_row_idx, A_col_idx in ti.static(ti.ndrange(12, 12)):
                lhs_row_idx = q_idx_vec[A_row_idx]
                lhs_col_idx = q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += strain_weight * ATA[A_row_idx, A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += volume_weight * ATA[A_row_idx, A_col_idx]

        for q_idx in ti.static(self.fix_particle_list):
            A_i_eye = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            weight = self.positional_weight
            qi_idx_x, qi_idx_y, qi_idx_z = q_idx * self.dim, q_idx * self.dim + 1, q_idx * self.dim + 2
            q_idx_vec = ti.Vector([qi_idx_x, qi_idx_y, qi_idx_z])
            for dim_idx in ti.static(range(self.dim)):
                lhs_idx = q_idx_vec[dim_idx]
                self.lhs[lhs_idx, lhs_idx] += weight * A_i_eye[dim_idx, dim_idx]

        for q_idx in ti.static(self.contact_particles_list):
            A_i_eye = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            weight = self.positional_weight
            qi_idx_x, qi_idx_y, qi_idx_z = q_idx * self.dim, q_idx * self.dim + 1, q_idx * self.dim + 2
            q_idx_vec = ti.Vector([qi_idx_x, qi_idx_y, qi_idx_z])
            for dim_idx in ti.static(range(self.dim)):
                lhs_idx = q_idx_vec[dim_idx]
                self.lhs[lhs_idx, lhs_idx] += weight * A_i_eye[dim_idx, dim_idx]
            

    @ti.kernel
    def construct_desired_pos(self):
        for i in ti.static(range(self.contact_num)):
            q_idx = self.contact_particles_list[i]
            self.node_desired_pos[i] = self.node_pos[q_idx] + self.dt * self.contact_vel[i]


    @ti.kernel
    def construct_sn(self):
        for i in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.sn[i*self.dim + d] = self.node_pos[i][d] + self.dt * self.node_vel[i][d]


    @ti.kernel
    def warm_start(self):
        for i in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.node_pos_new[i][d] = self.node_pos[i][d]


    @ti.kernel
    def local_solve(self):
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2, idx3, idx4 = self.element[ele_idx]
            a, b, c, d = self.node_pos_new[idx1], self.node_pos_new[idx2], \
                         self.node_pos_new[idx3], self.node_pos_new[idx4]
            D_i = ti.Matrix.cols([b - a, c - a, d - a])
            F_i = ti.cast(D_i @ self.B[ele_idx], ti.f64)
            self.F[ele_idx] = F_i

            U, sig, V = ti.svd(F_i, ti.f64)
            self.Bp[ele_idx] = U @ V.transpose()        # Strain contraint

            D, max_it, tol = ti.Vector([5., 5., 5.]), 50, 1.e-6
            for itr in range(max_it):
                aa, bb, cc = D[0]+sig[0,0], D[1]+sig[1,1], D[2]+sig[2,2]
                C = aa*bb*cc - 1
                partial_C = ti.Vector([bb*cc, aa*cc, aa*bb])

                D_tmp = (partial_C.dot(D)-C) / partial_C.dot(partial_C) * partial_C
                D_error = (D - D_tmp).norm()
                D = D_tmp
                if D_error < tol:
                    break

            PP = ti.Matrix([[D[0]+sig[0,0], 0, 0], [0, D[1]+sig[1,1], 0], [0, 0, D[2]+sig[2,2]]])
            self.Bp[ele_idx+self.ELEMENT_NUM] = U @ PP @ V.transpose()        # Volume contraint


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        for i in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.rhs[i*dim + d] = self.node_mass[i] / self.dt**2 * self.sn[i*dim + d]
            
        for ele_idx in range(self.ELEMENT_NUM):
            idx1, idx2, idx3, idx4 = self.element[ele_idx]
            idx1_x, idx1_y, idx1_z = idx1 * self.dim, idx1 * self.dim + 1, idx1 * self.dim + 2
            idx2_x, idx2_y, idx2_z = idx2 * self.dim, idx2 * self.dim + 1, idx2 * self.dim + 2
            idx3_x, idx3_y, idx3_z = idx3 * self.dim, idx3 * self.dim + 1, idx3 * self.dim + 2
            idx4_x, idx4_y, idx4_z = idx4 * self.dim, idx4 * self.dim + 1, idx4 * self.dim + 2
            q_idx_vec = ti.Vector([idx1_x, idx1_y, idx1_z, idx2_x, idx2_y, idx2_z,
                                   idx3_x, idx3_y, idx3_z, idx4_x, idx4_y, idx4_z])

            A_i = self.A[ele_idx]
            AT_Bp_all_i = ti.Vector.zero(ti.f64, self.dim*(self.dim+1))
            for t in range(2):
                Bp_i = self.Bp[t*self.ELEMENT_NUM + ele_idx]
                Bp_i_vec = ti.Vector([Bp_i[0,0], Bp_i[0,1], Bp_i[0,2], 
                                      Bp_i[1,0], Bp_i[1,1], Bp_i[1,2], 
                                      Bp_i[2,0], Bp_i[2,1], Bp_i[2,2]])
                weight = 0.
                if t == 0:
                    weight = self.strain_weight[ele_idx]
                else:
                    weight = self.volume_weight[ele_idx]
                AT_Bp_all_i += weight * A_i.transpose() @ Bp_i_vec

            for j in range(self.dim*(self.dim+1)):
                self.rhs[q_idx_vec[j]] += AT_Bp_all_i[j]

        for q_idx in ti.static(self.fix_particle_list):
            weight = self.positional_weight
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*self.dim + d] += weight * self.node_init_pos[q_idx][d]

        for i in ti.static(range(self.contact_num)):
            q_idx = self.contact_particles_list[i]
            weight = self.positional_weight
            for d in ti.static(range(self.dim)):
                self.rhs[q_idx*self.dim + d] += weight * self.node_desired_pos[i][d]


    @ti.kernel
    def update_pos_new(self, sol:ti.types.ndarray()):
        for i in range(self.PARTICLE_NUM):
            for d in ti.static(range(self.dim)):
                self.node_pos_new[i][d] = sol[i*self.dim+d]


    @ti.kernel
    def update_vel_pos(self):
        for i in range(self.PARTICLE_NUM):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]

    
    def extract_bottom_particles(self):
        bottom_flag = ti.field(dtype=ti.i32, shape=self.PARTICLE_NUM)

        @ti.kernel
        def cal_bottom_glag(particle_num:float, seed_size:float):
            for q_idx in range(particle_num):
                if self.node_pos[q_idx].y < 0. + seed_size/3:
                    bottom_flag[q_idx] = True

        cal_bottom_glag(self.PARTICLE_NUM, self.seed_size)
        bottom_particles_set = set()
        for i in range(self.PARTICLE_NUM):
            if bottom_flag[i]:
                bottom_particles_set.add(i)
        bottom_particles_list = list(bottom_particles_set)
        bottom_num = len(bottom_particles_list)

        return bottom_particles_list, bottom_num
        

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
        # self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_NUM)
        self.node_show = ti.Vector.field(3, dtype=ti.f32, shape=self.BOTTOM_NUM)
        self.edge_show = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_NUM)
        self.edge_show.from_numpy(self.edge_np)


    def gui_show(self, window, canvas, scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0):
        """
        Show the GGUI
        """
        if SHOW_FLAG is False:
            return
        scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        scene.ambient_light((0.8, 0.8, 0.8))
        # the conversion of object particles, etc. the ggui of the taichi only support float32
        # 只取了最下部的粒子进行展示
        self.node_show.from_numpy(self.node_pos.to_numpy(dtype=np.float32)[self.bottom_particles_list])
        # self.node_show.from_numpy(self.node_pos.to_numpy(dtype=np.float32))

        scene.particles(self.node_show, radius=0.001, color=(0., 0., 0.))
        # scene.lines(self.node_show, width=1., indices=self.edge_show, color=(0., 0., 0.),
        #             vertex_count=0)
        canvas.scene(scene)
        canvas.set_background_color((1.0, 1.0, 1.0))
        # if WRITE_FLAG is True and itr_num % 10 == 0:
        if WRITE_FLAG is True:
            window.save_image(f'FigureWrite/{itr_num}.png')
        window.show()

    
    def preset_gui(self, camera_pos:list, camera_target:list):
        """
        Define the camera position & target
        """
        # self.window, self.camera, self.scene = self.gui_set(pos=[0.1, 0.2, 0.], target=[0.1, 0., 0.])
        self.window, self.camera, self.scene = self.gui_set(pos=camera_pos, target=camera_target)
        self.canvas = self.window.get_canvas()
        self.show_preset()

    
    @ti.kernel
    def init_vel(self):
        for i in range(self.PARTICLE_NUM):
            if self.node_init_pos[i].x > self.shape[0] - self.seed_size/3:
                self.node_vel[i].x = 5.
            else:
                self.node_vel[i].x = 0.


    def substep(self, step_num):
        self.construct_desired_pos()
        self.construct_sn()
        self.warm_start()
        # Local sovle needs iteration
        for itr in ti.static(range(self.solve_iteration)):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        # np.savetxt(f'DataWrite/rhs_{step_num}.csv', rhs_np, fmt='%f', delimiter=',')
        # np.savetxt(f'DataWrite/pos_{step_num}.csv', self.node_pos.to_numpy(), fmt='%f', delimiter=',')
        # np.savetxt(f'DataWrite/vel_{step_num}.csv', self.node_vel.to_numpy(), fmt='%f', delimiter=',')
        self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False,
                      itr_num=step_num)


def main():
    cube_shape = [0.1, 0.02, 0.1]
    mesh_file = 'Mesh/cube.msh'
    generate_msh(cube_shape, 0.01, mesh_file)
    class MyObect(SoftObject):
        def __init__(self, shape, seed_size, file=mesh_file):
            super().__init__(shape, seed_size, file)

    soft_obj = MyObect(shape=cube_shape, seed_size=0.01)
    soft_obj.preset_gui([0.1, 0.2, 0.05], [0.1, 0., 0.05])

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)
    # soft_obj.init_vel()
    np.savetxt('DataWrite/node_mass.csv', soft_obj.node_mass.to_numpy(), fmt='%.8f', delimiter=',')
    # np.savetxt('DataWrite/vel_init.csv', soft_obj.node_vel.to_numpy(), fmt='%f', delimiter=',')
    np.savetxt('DataWrite/lhs.csv', lhs_np, fmt='%f', delimiter=',')
    # exit(0)

    sample_particles_pos_list = []
    for i in range(500):
        soft_obj.substep(i)
        sample_particles_pos_list.append(list(soft_obj.node_pos[132].to_numpy()))
    np.savetxt(f'DataWrite/pos_before.csv', soft_obj.node_pos.to_numpy(), fmt='%f', delimiter=',')

    soft_obj.contact_vel.fill(0.)
    for i in range(1001):
        soft_obj.substep(i)
        sample_particles_pos_list.append(list(soft_obj.node_pos[132].to_numpy()))
        if i % 50 == 0:
            np.savetxt(f'DataWrite/pos_after_{i}.csv', soft_obj.node_pos.to_numpy(), fmt='%f', delimiter=',')
    
    np.savetxt(f'DataWrite/pos_init.csv', soft_obj.node_init_pos.to_numpy(), fmt='%f', delimiter=',')
    np.savetxt(f'DataWrite/sample_pos.csv', np.array(sample_particles_pos_list), fmt='%f', delimiter=',')

if __name__ == '__main__':
    main()