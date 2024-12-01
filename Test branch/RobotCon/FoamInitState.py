"""
指定接触状态下的变形初始状态（重力导致）
created at 2024-10-27 by hsy
"""
import time
import taichi as ti
ti.init(arch=ti.gpu, device_memory_GB=6.0,  default_fp=ti.f64, debug=True)
import taichi.math as tm
import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from scipy import sparse
from scipy.sparse.linalg import spsolve
from ControlSim3d import SoftObject
np.set_printoptions(linewidth=200)



class Foam(SoftObject):
    def __init__(self, shape:List[float], size:float, file:str, contact:List[int], fix:List[int]):
        super().__init__(shape, size, file, contact, fix)
        self.gravity = 9.8


    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        for idx in range(self.PARTICLE_NUM):
            self.sn[idx*dim+0] = self.node_pos[idx].x + self.dt * self.node_vel[idx].x
            self.sn[idx*dim+1] = self.node_pos[idx].y + self.dt * self.node_vel[idx].y
            self.sn[idx*dim+2] = self.node_pos[idx].z + self.dt * self.node_vel[idx].z - self.gravity * self.dt**2

        # ti.loop_config(serialize=True)
        if self.contact_num > 0:
            for idx in range(self.contact_num):
                idx_value = self.contact_particles_ti[idx]
                self.sn[idx_value*dim+0] = self.node_pos[idx_value].x +  self.contact_vel[idx].x * self.dt
                self.sn[idx_value*dim+1] = self.node_pos[idx_value].y +  self.contact_vel[idx].y * self.dt
                self.sn[idx_value*dim+2] = self.node_pos[idx_value].z +  self.contact_vel[idx].z * self.dt

        if self.fix_num > 0:
            for idx in range(self.fix_num):
                idx_value = self.fix_particles_ti[idx]
                self.sn[idx_value*dim+0] = self.node_init_pos[idx_value].x
                self.sn[idx_value*dim+1] = self.node_init_pos[idx_value].y
                self.sn[idx_value*dim+2] = self.node_init_pos[idx_value].z


    @ti.kernel
    def update_vel_pos(self):
        # ti.loop_config(serialize=True)
        for idx in range(self.contact_num):
            idx_value = self.contact_particles_ti[idx]
            self.node_pos_new[idx_value] = self.node_pos[idx_value] + self.contact_vel[idx] * self.dt

        for i in range(self.PARTICLE_NUM):
            self.node_vel[i] = (self.node_pos_new[i] - self.node_pos[i]) / self.dt
            self.node_pos[i] = self.node_pos_new[i]

        for idx in ti.static(self.contact_particles_list):
            self.node_vel[idx] = ti.Vector([0., 0., 0.])

        for idx in ti.static(self.fix_particle_list):
            self.node_vel[idx] = ti.Vector([0., 0., 0.])


    def substep(self, step_num:ti.i32):
        self.construct_sn()
        self.warm_start()
        for itr in range(self.solve_iteration):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=step_num)


def main():
    cube_shape = [0.5, 0.5, 0.03]
    mesh_size = 0.03
    # X-Y平面放置，Z方向只有一个单元
    mesh_file = 'Mesh/foam_small.msh'
    contact_list = []
    # contact_list = [8, 9, 10, 11, 32, 33, 34, 35]
    contact_list = [14, 15, 16, 17, 38, 39, 40, 41]
    # contact_list = [10, 11, 12, 13] + [32, 33, 34, 35]
    fix_list = [250, 251, 252, 253, 274, 275, 276, 277]
    # fix_list = [206, 207, 208, 209] + [228, 229, 230, 231]
    soft = Foam(cube_shape, mesh_size, mesh_file, contact_list, fix_list)
    soft.preset_gui(camera_pos=[0.25, 0.25, 0.8], camera_target=[0.25, 0.25, 0.])

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    delta_node_list = []

    for itr in range(2000):
        print(f"Step: {itr}-----------------------------------------------------")
        soft.substep(itr)

        delta_tmp = []
        delta_tmp.append(soft.node_init_pos[1].z - soft.node_pos[1].z)
        delta_tmp.append(soft.node_init_pos[23].z - soft.node_pos[23].z)
        delta_tmp.append(soft.node_init_pos[265].z - soft.node_pos[265].z)
        delta_tmp.append(soft.node_init_pos[287].z - soft.node_pos[287].z)
        delta_node_list.append(delta_tmp)

        print(f"Fixed node {1} pos: {soft.node_init_pos[1].z - soft.node_pos[1].z:.6f}")
        print(f"Fixed node {23} pos: {soft.node_init_pos[23].z - soft.node_pos[23].z:.6f}")
        print(f"Fixed node {265} pos: {soft.node_init_pos[265].z - soft.node_pos[265].z:.6f}")
        print(f"Fixed node {287} pos: {soft.node_init_pos[287].z - soft.node_pos[287].z:.6f}")

    np.savetxt(f"delta_node_pos.csv", np.array(delta_node_list), fmt="%.8f", delimiter=",")
    np.savetxt(f"foam_init_state.csv", soft.node_pos.to_numpy(), fmt="%.8f", delimiter=",")


if __name__ == '__main__':
    main()