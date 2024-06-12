"""
This file simulates deformation by PD method in 3D
"""

import taichi as ti
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
        np.savetxt('node_np.csv', node_np, fmt='%f', delimiter=',')
        np.savetxt('element_np.csv', element_np, fmt='%d', delimiter=',')
        np.savetxt('volume_np.csv', volume_np, fmt='%f', delimiter=',')

        self.PARTICLE_NUM = node_np.shape[0]
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
        self.element.from_numpy(element_np.astype(np.int32))
        self.element_volume.from_numpy(volume_np.astype(np.float64))

        self.B = ti.Matrix.field(3, 3, dtype=ti.f64, shape=self.ELEMENT_NUM)

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
            a, b, c, d = self.node_init_pos[idx1], self.node_init_pos[idx2], self.node_init_pos[idx2], self.node_init_pos[idx2]
            B_i_inv = ti.Matrix.cols([a - d, b - d, c - d])
            self.B[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_weight(self):
        for i in range(self.ELEMENT_NUM):
            self.strain_weight[i] = self.mu * 2 * self.element_volume[i]



def main():
    class MyObect(SoftObject):
        def __init__(self, shape, seed_size):
            super().__init__(shape, seed_size)


    soft_obj = MyObect(shape=[0.1, 0.1, 0.1], seed_size=0.1/2)



if __name__ == '__main__':
    main()