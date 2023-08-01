"""
This file is used to run auto differentiation test of PD simulation. The sparse matrix is solved
by Taichi.
"""

import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f32, debug=True)
import taichi.math as tm
import numpy as np


@ti.data_oriented
class PDTest():
    def __init__(self):
        self.dim = 2
        self.dt = 1./60
        self.rho = 1.0
        self.E = 5.e5
        self.nu = 0.4
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        self.positional_mass = 1.e6
        self.grasp_mass = 1.e6
        self.solve_itr = 10
        self.GRASP_VEL = ti.Vector.field(2, dtype=ti.f64, shape=1)
        self.GRASP_VEL[0] = ti.Vector([0.0, 0.0])

        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=4)
        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=4, needs_grad=True)
        self.node_pos_new = ti.Vector.field(2, dtype=ti.f64, shape=4)
        self.node_mass = ti.field(dtype=ti.f64, shape=4)
        self.node_vel = ti.Vector.field(2, dtype=ti.f64, shape=4)
        self.NODE_NUM = 4

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=5)
        self.EDGE_NUM = 5

        self.element = ti.Vector.field(3, dtype=ti.i32, shape=2)
        self.ele_vol = ti.field(dtype=ti.f64, shape=2)
        self.strain_weight = ti.field(dtype=ti.f64, shape=2)
        self.vol_weight = ti.field(dtype=ti.f64, shape=2)
        self.ELE_NUM = 2

        self.B = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELE_NUM)
        self.F = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELE_NUM)
        self.A = ti.Matrix.field(4, 6, dtype=ti.f64, shape=2*self.ELE_NUM)
        self.Bp = ti.Matrix.field(2, 2, dtype=ti.f64, shape=2*self.ELE_NUM)

        self.sn = ti.field(dtype=ti.f64, shape=self.NODE_NUM*2)
        self.lhs = ti.field(ti.f64, shape=(2*self.NODE_NUM, 2*self.NODE_NUM))
        self.rhs = ti.field(ti.f64, shape=2*self.NODE_NUM)

        self.lhs_t_builder = ti.linalg.SparseMatrixBuilder(2*self.NODE_NUM, 2*self.NODE_NUM, max_num_triplets=100)
        self.rhs_t = ti.ndarray(dtype=ti.f32, shape=2*self.NODE_NUM)
        self.sn_t = ti.ndarray(dtype=ti.f32, shape=2*self.NODE_NUM)

        self.assign_variable()
        self.construct_B()
        self.construct_volume()


    def assign_variable(self):
        node_pos_np = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
        node_mass_np = np.array([1.0, 1.0, 1.0, 1.0])
        self.node_pos_init.from_numpy(node_pos_np)
        self.node_pos.from_numpy(node_pos_np)
        self.node_mass.from_numpy(node_mass_np)

        edge_np = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
        self.edge.from_numpy(edge_np)
        self.edge_np = edge_np

        element_np = np.array([[0, 1, 2], [0, 2, 3]])
        self.element.from_numpy(element_np)

        self.fix_node_list = [0, 1]
        self.grasp_node_list = [2]


    @ti.kernel
    def construct_B(self):
        for i in range(self.ELE_NUM):
            ia, ib, ic = self.element[i]
            pa, pb, pc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            B_i_inv = ti.Matrix.cols([pb - pa, pc - pa])
            self.B[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_volume(self):
        for i in range(self.ELE_NUM):
            ia, ib, ic = self.element[i]
            pa, pb, pc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            ele_volume = abs((pb - pa).cross(pc - pa)) / 2.0
            self.ele_vol[i] = ele_volume
            self.strain_weight[i] = self.mu * 2 * ele_volume
            self.vol_weight[i] = self.lam * self.dim * ele_volume


    @ti.kernel
    def precomputation(self, lhs_t:ti.types.sparse_matrix_builder()):
        ELE_NUM = self.ELE_NUM
        dim = self.dim

        for i in range(self.NODE_NUM):
            for d in ti.static(range(2)):
                lhs_t[i*dim + d, i*dim + d] += self.node_mass[i]/self.dt**2

        for i in range(self.ELE_NUM):
            B_i = self.B[i]
            a = B_i[0, 0]
            b = B_i[0, 1]
            c = B_i[1, 0]
            d = B_i[1, 1]

            for t in range(2):
                self.A[t*ELE_NUM + i][0, 0] = -a - c
                self.A[t*ELE_NUM + i][0, 2] = a
                self.A[t*ELE_NUM + i][0, 4] = c
                self.A[t*ELE_NUM + i][1, 0] = -b - d
                self.A[t*ELE_NUM + i][1, 2] = b
                self.A[t*ELE_NUM + i][1, 4] = d
                self.A[t*ELE_NUM + i][2, 1] = -a - c
                self.A[t*ELE_NUM + i][2, 3] = a
                self.A[t*ELE_NUM + i][2, 5] = c
                self.A[t*ELE_NUM + i][3, 1] = -b - d
                self.A[t*ELE_NUM + i][3, 3] = b
                self.A[t*ELE_NUM + i][3, 5] = d

        for ele_idx in range(self.ELE_NUM):
            ia, ib, ic = self.element[ele_idx]
            ia_x, ia_y = ia * dim, ia * dim + 1
            ib_x, ib_y = ib * dim, ib * dim + 1
            ic_x, ic_y = ic * dim, ic * dim + 1
            q_idx_vec = ti.Vector([ia_x, ia_y, ib_x, ib_y, ic_x, ic_y])
            for t in range(2):
                A_i = self.A[t*ELE_NUM + ele_idx]
                # for A_row_idx, A_col_idx in ti.static(ti.ndrange(6,6)):
                for A_row_idx, A_col_idx in ti.ndrange(6, 6):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    matrix_temp = ti.f64(0.)
                    for idx in range(dim**2):
                        weight = ti.f64(0.)
                        if t == 0:
                            weight = self.strain_weight[ele_idx]
                        else:
                            weight = self.vol_weight[ele_idx]
                        matrix_temp += weight * A_i[idx, A_row_idx] * A_i[idx, A_col_idx]
                        lhs_t[lhs_row_idx, lhs_col_idx] += \
                            weight * A_i[idx, A_row_idx] * A_i[idx, A_col_idx]
                    # print('Matrix Index:', lhs_row_idx, lhs_col_idx)
                    # lhs_t[lhs_row_idx, lhs_col_idx] += matrix_temp


def main():
    class AutoDiffPD(PDTest):
        def __init__(self):
            super().__init__()
            self.my_loss = ti.field(dtype=ti.f64, shape=(), needs_grad=True)

    test = AutoDiffPD()

    test.precomputation(test.lhs_t_builder)

    test.lhs_t_builder.print_triplets()
    test.lhs_t = test.lhs_t_builder.build()
    test.solver = ti.linalg.SparseSolver(solver_type="LLT")
    test.solver.analyze_pattern(test.lhs_t)
    test.solver.factorize(test.lhs_t)


if __name__ == '__main__':
    main()