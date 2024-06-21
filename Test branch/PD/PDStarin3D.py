"""
This file simulates deformation by PD method in 3D
参考"_PDStrain.py"更改的三维版本,其中`strain_weight`和`volume_weight`和
之前的文件都不同
"""

import taichi as ti
import taichi.math as tm
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)
import numpy as np
from scipy import sparse


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
    def __init__(self, shape, seed_size):
        self.shape = shape
        self.seed_size = seed_size
        self.dt = 1./120
        self.rho = 1.e1
        self.volume_sum = ti.field(ti.f64, shape=())
        self.positional_weight = 1.e4
        self.E, self.nu = 5.e2, 0.1
        self.dim = len(shape)
        self.mu , self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        node_np, element_np, volume_np = self.load_msh('Mesh/liver.msh')
        self.edge_np = get_tetrahedron_edges(element_np)
        np.savetxt('node_np.csv', node_np, fmt='%f', delimiter=',')
        np.savetxt('element_np.csv', element_np, fmt='%d', delimiter=',')
        np.savetxt('volume_np.csv', volume_np, fmt='%f', delimiter=',')

        self.PARTICLE_NUM = node_np.shape[0]
        self.EDGE_NUM = edge_np.shape[0]
        self.ELEMENT_NUM = element_np.shape[0]

        # node_pos: 3D position of each node in time step
        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_init_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        # For local solver
        self.node_pos_new = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_mass = ti.field(dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.node_vel = ti.Vector.field(2, dtype=ti.f64, shape=self.PARTICLE_NUM)
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

        self.fix_particle_list = []

        self.construct_B()
        self.construct_weight()
        # self.construct_mass(self.volume_sum[None])

        # Print the information
        print('Particle number:', self.PARTICLE_NUM)
        print('Element number:', self.ELEMENT_NUM)


    @staticmethod
    def load_msh(file_path):
        nodes_array, elements_array = read_msh_file(file_path)
        # 打印节点和单元信息
        # print(f"节点信息：{nodes_array.shape[0]}")
        # print(f"单元信息：{elements_array.shape[0]}")

        volumes_list = []
        for ele in elements_array:
            node_indices = ele - 1          # 转换为从0开始的索引
            tetra_nodes = nodes_array[node_indices]
            volume_tmp = cal_tet_volume(tetra_nodes)
            volumes_list.append(volume_tmp)

        return nodes_array, elements_array, np.array(volumes_list)


    @ti.kernel
    def construct_mass(self):
        mass_tmp = self.rho * self.volume_sum / self.PARTICLE_NUM
        self.node_mass.fill(mass_tmp)


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
            for d in ti.static(range(dim)):
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
            for dim_idx in ti.static(dim):
                lhs_idx = q_idx_vec[dim_idx]
                self.lhs[lhs_idx, lhs_idx] += weight * A_i_eye[dim_idx, dim_idx]


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

            PP = ti.Matrix([[aa, 0, 0], [0, bb, 0], [0, 0, cc]])
            self.Bp[ele_idx+self.ELEMENT_NUM] = U @ PP @ V.transpose()        # Volume contraint


    @ti.kernel
    def construct_rhs(self):
        self.rhs.fill(0.)
        dim = self.dim
        for i in range(self.PARTICLE_NUM):
            for d in ti.static(range(dim)):
                self.rhs[i*dim + d] = self.node_mass[i] / self.dt**2 * self.sn[i*dim + d]
            
        for i in range(self.ELEMENT_NUM):
            idx1, idx2, idx3, idx4 = self.element[i]
            idx1_x, idx1_y, idx1_z = idx1 * self.dim, idx1 * self.dim + 1, idx1 * self.dim + 2
            idx2_x, idx2_y, idx2_z = idx2 * self.dim, idx2 * self.dim + 1, idx2 * self.dim + 2
            idx3_x, idx3_y, idx3_z = idx3 * self.dim, idx3 * self.dim + 1, idx3 * self.dim + 2
            idx4_x, idx4_y, idx4_z = idx4 * self.dim, idx4 * self.dim + 1, idx4 * self.dim + 2
            q_idx_vec = ti.Vector([idx1_x, idx1_y, idx1_z, idx2_x, idx2_y, idx2_z,
                                   idx3_x, idx3_y, idx3_z, idx4_x, idx4_y, idx4_z])

            A_i = self.A[i]
            AT_Bp_all_i = ti.Vector.zero(ti.f64, self.dim*(self.dim+1))
            for t in range(2):
                Bp_i = self.Bp[t*self.ELEMENT_NUM+i]
                Bp_i_vec = ti.Vector([Bp_i[0,0], Bp_i[0,1], Bp_i[0,2], 
                                      Bp_i[1,0], Bp_i[1,1], Bp_i[1,2], 
                                      Bp_i[2,0], Bp_i[2,1], Bp_i[2,2]])
                if t == 0:
                    weight = self.strain_weight[i]
                else:
                    weight = self.volume_weight[i]
                AT_Bp_all_i += weight * A_i.transpose() @ Bp_i_vec

            for i in range(self.dim*(self.dim+1)):
                self.rhs[q_idx_vec[i]] += AT_Bp_all_i[i]

        for i in ti.static(self.fix_particle_list):
            weight = self.positional_weight
            for d in ti.static(range(self.dim)):
                self.rhs[i*self.dim + d] += weight * self.node_init_pos[i][d]

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
        self.node_show.from_numpy(np.insert(self.node_pos.to_numpy(dtype=np.float32), 1,
                                            np.zeros(self.PARTICLE_NUM), axis=1))

        scene.particles(self.node_show, radius=0.001, color=(0., 0., 0.))
        scene.lines(self.node_show, width=1., indices=self.edge_show, color=(0., 0., 0.),
                    vertex_count=0)
        canvas.scene(scene)
        canvas.set_background_color((1.0, 1.0, 1.0))
        # if WRITE_FLAG is True and itr_num % 10 == 0:
        if WRITE_FLAG is True:
            window.save_image(f'FigureWrite/{itr_num}.png')
        window.show()

    
    @ti.kernel
    def init_vel(self):
        for i in range(self.PARTICLE_NUM):
            if self.node_init_pos[i].x > self.shape[0] - self.seed_size/3:
                self.node_vel[i].x = 5.
            else:
                self.node_vel[i].x = 0.


    def substep(self, step_num):
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
        self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False,
                      itr_num=step_num)


def main():
    class MyObect(SoftObject):
        def __init__(self, shape, seed_size):
            super().__init__(shape, seed_size)


    soft_obj = MyObect(shape=[0.1, 0.1, 0.1], seed_size=0.1/2)



if __name__ == '__main__':
    main()