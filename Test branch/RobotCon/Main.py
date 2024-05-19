"""
Obtain the position of the black on red soft object.
"""


import numpy as np
from scipy.spatial import Delaunay
from ControlSimulation import *


dot_pixel = np.array([456, 823], dtype=int)

orign_pixel = np.array([456, 823], dtype=int)

dot_pos = np.array([0.05, 0.01])


"""标记点检测"""


def feature_barycentric_coordinates(p, mesh_nodes):
    """
    Compute the barycentric coordinates of a point p with respect to the triangle p0, p1, p2
    """
    p0, p1, p2 = mesh_nodes
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])


def find_element(tri, dot_pos):
    """
    Find the element which contains the dot
    :param tri:
    :param dot_pos:
    :return: element index
    """
    # 查找包含点的三角形
    simplex = tri.find_simplex(dot_pos)

    if simplex != -1:
        # 返回包含点的三角形的顶点索引
        return tri.simplices[simplex]
    else:
        return None


def main():
    obj_shape = [0.1, 0.1]
    obj_seed_size = 0.01

    class MyObject(SoftObject):
        def __init__(self, shape, seed_size, contact_idx):
            super().__init__(shape, seed_size, contact_idx)
            self.marker_element = None
            self.barycentric = None
            self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=1)
            self.dot_pos[0] = dot_pos
            self.dot_pos_desired = ti.Vector.field(2, dtype=ti.f64, shape=1)
            self.dot_pos_desired[0] = self.dot_pos[0] + ti.Vector([0.002, 0.])

            self.marker_element_get()


        def marker_element_get(self):
            mesh_nodes = self.tri.points
            element_np = find_element(self.tri, dot_pos)
            if element_np is not None:
                self.marker_element = list(element_np)
                barycentric_init = feature_barycentric_coordinates(dot_pos, mesh_nodes[element_np])
                self.barycentric = barycentric_init
                print("The dot is in element: ", element_np)
                print("The barycentric coordinates are: ", barycentric_init)
            else:
                print("The dot is not in the mesh object.")


        def construct_L_mrker(self):
            """
            Construct the L with marker that doesn't position on the node.
            :return:
            """
            dim = self.dim
            barycentric = self.barycentric
            desired_pos = self.dot_pos_desired[0]
            current_pos = self.dot_pos[0]
            error = current_pos - desired_pos
            L = error.norm() ** 2
            for idx, ele_idx in enumerate(self.marker_element):
                self.dL[ele_idx * 2] = 2 * (current_pos[0] - desired_pos[0]) * barycentric[idx]
                self.dL[ele_idx * 2 + 1] = 2 * (current_pos[1] - desired_pos[1]) * barycentric[idx]

            return error, L


        def diff_pd(self, itr_num):
            # compute Jacobian matrix by DiffPD
            self.partial_p()
            dA = self.rhs_dA.to_numpy()
            par_L = self.dL.to_numpy()
            z_np = self.z.to_numpy()
            for itr in ti.static(range(itr_num)):
                rhs_diff_np = dA @ z_np + par_L
                z_new_np = self.pre_fact_lhs_solve(rhs_diff_np)
                z_np = z_new_np
            self.z.from_numpy(z_np)


        @ti.kernel
        def compute_grad_y(self):
            for i in range(self.PARTICLE_NUM):
                idx0, idx1 = i*self.dim, i*self.dim+1
                self.grad_y[i].x = self.z[idx0]*self.node_mass[i]/self.dt**2
                self.grad_y[i].y = self.z[idx1]*self.node_mass[i]/self.dt**2


        def substep(self, step_num):
            # PD forward simulation
            self.construct_sn()
            self.warm_up()
            for itr in ti.static(range(self.solve_iteration)):
                self.local_solve()
                self.construct_rhs()
                rhs_np = self.rhs.to_numpy()
                node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
                self.update_pos_new(node_pos_new_np)

            self.update_vel_pos()


        def compute_gradient(self):
            error, loss_tmp = self.construct_L_mrker()
            self.loss = loss_tmp
            self.diff_pd(10)
            self.compute_grad_y()


    soft_obj = MyObject(obj_shape, obj_seed_size, [10])

    soft_obj.preset()

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    soft_obj.substep(1)
    soft_obj.compute_gradient()

    # np.savetxt('dL.txt', soft_obj.dL.to_numpy())
    # np.savetxt('grad_y.txt', soft_obj.grad_y.to_numpy())

    print('The gradient of the action:', soft_obj.grad_y[soft_obj.grasp_particle_list[0]].to_numpy())


    """机器人控制部分"""




if __name__ == '__main__':
    main()