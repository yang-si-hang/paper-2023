""" 控制曲面的边界，完成对遮挡器官的暴露
created by hsy on 2025-07-28
"""
from pathlib import Path
import time
import numpy as np
import taichi as ti
from scipy import sparse
from scipy.sparse import linalg as spla
import numpy.typing as npt
import asyncio
import pkg_resources
import qtm_rt
from RobAction import URROb
ti.init(arch=ti.cpu, default_fp=ti.f64, debug=True)

from _DiffPDBend import SoftBend2D
from Utilize.GenMsh import save_vtu
from Utilize.MathNp import compress_vectors, find_triangle
from Utilize.Qualisys import init_marker, receive_qualysis
from CoordinateTransform import find_element, feature_barycentric_coordinates

dir_path = Path(__file__).parent

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip:str = '192.168.253.17'
qualysis_password:str = ''

selected_index = [145, 149, 148, 140]  # 选择的标记点索引
POINTS_NUM = len(selected_index)

# ----- 获取期望形状 -----
def genereate_transform():
    # 用来生成软体坐标系相对于世界坐标系的变换矩阵
    origin_pos  = np.array([73.8, 223.6, 174.1]) * 1.e-3
    x_deviation = np.array([73.5, 91.9, 174.6]) * 1.e-3
    y_deviation = np.array([195.8, 223.4, 177.9]) * 1.e-3

    x_axis = x_deviation - origin_pos
    y_axis = y_deviation - origin_pos
    z_axis = np.cross(x_axis, y_axis)

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3]  = origin_pos - 0.015 / 2 * x_axis - 0.015 / 2 * y_axis - 0.0096 * z_axis  # 平移到软体坐标系的原点
    print(f"Transformation matrix:\n{transformation_matrix}")
    np.savetxt(f'{dir_path}/Data/transformation_matrix.csv', transformation_matrix, fmt='%.10f', delimiter=',')

# genereate_transform()
# exit()

# 实验之前需要标定软体坐标系，确定Qualisys到软体坐标系的变换矩阵
trans_matrix = np.loadtxt(f'{dir_path}/Data/transformation_matrix.csv', delimiter=',')

def pos_in_soft(pos:npt.NDArray, trans_matrix:npt.NDArray)->npt.NDArray:
    """ 将世界坐标系的位置转换到软体坐标系 """
    pos_soft = np.linalg.inv(trans_matrix) @ np.c_[pos, np.ones(POINTS_NUM)].T
    pos_soft = pos_soft[:3].T
    return pos_soft

class SoftSurface(SoftBend2D):
    def __init__(self, shape, fix, contact, marker_pos_init, E, nu, dt, density, **kwargs):
        super().__init__(shape, fix, contact, E, nu, dt, density, **kwargs)
        self.loss = 0.
        self.marker_elements = ti.field(dtype=ti.i32, shape=POINTS_NUM)
        self.barycentrics    = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        # dot_pos: 是模型的粒子状态; dot_pos_soft: 是软体坐标系下的粒子状态
        self.dot_pos      = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_soft = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos.from_numpy(marker_pos_init)
        self.dot_pos_init.from_numpy(marker_pos_init)

        self.get_marker_element()
        print(f"Barycentric coordinates: {self.barycentrics}")

    def get_marker_element(self):
        triangle_idx, bary = find_triangle(self.dot_pos_init.to_numpy(), self.mesh.verts.pos_init.to_numpy(), self.ele.to_numpy())
        for i in range(POINTS_NUM):
            if triangle_idx[i] == -1:
                raise ValueError("The dot is not in the mesh object.")
            else:
                self.marker_elements[i] = triangle_idx[i]
                self.barycentrics[i] = ti.Vector([bary[i][0], bary[i][1], bary[i][2]], dt=ti.f64)

    @ti.kernel
    def get_marker_pos(self):
        for i in range(POINTS_NUM):
            ele_id = self.marker_elements[i]
            element = self.ele[ele_id]
            barycentric = self.barycentrics[i]
            pos1, pos2, pos3 = self.mesh.verts.pos[element[0]], self.mesh.verts.pos[element[1]], self.mesh.verts.pos[element[2]]

            dot_pos = barycentric[0] * pos1 + barycentric[1] * pos2 + barycentric[2] * pos3
            self.dot_pos[i] = dot_pos

    def construct_L_soft(self, dot_soft:npt.NDArray):
        """ 控制边界上的多个标记点到期望位置
        """
        self.dL_dq_y.fill(0.)
        loss = 0.
        for i in range(self.marker_N):
            q_i = self.marker_ti[i]
            desired_pos = self.marker_pos_desired[i]
            current_pos = dot_soft[i, :]
            self.error[i] = current_pos - desired_pos

            self.dL_dq_y[q_i*3]     = 2 * self.error[i].x
            self.dL_dq_y[q_i*3 + 1] = 2 * self.error[i].y
            self.dL_dq_y[q_i*3 + 2] = 2 * self.error[i].z

            loss += self.error[i].norm_sqr()
        return loss

    def compute_dcontact(self, dot_soft:npt.NDArray):
        loss = self.construct_L_soft(dot_soft)
        self.construct_g_hessian()
        self.compute_z_act(10)

        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())
        return loss

async def main():
    # ----- 连接Qualisys -----
    connection = await qtm_rt.connect(qualysis_ip)
    if connection is None:
        print("Failed to connect")
        return
    async with qtm_rt.TakeControl(connection, qualysis_password):
        await connection.new()

    # ----- 获得期望形状下的marker点 -----
    # _, marker_desired = await init_marker(connection, selected_index, step_num=50)
    # marker_desired_soft = pos_in_soft(marker_desired, trans_matrix)
    # np.savez(f"{dir_path}/Data/marker_desired.npz", marker_desired_soft=marker_desired_soft)
    # print(f"Desired marker positions: \n{marker_desired_soft}")
    # exit()

    # ----- 初始化marker点 -----
    dots_pos_init_dict, dots_pos_init_np = await init_marker(connection, selected_index)
    print(f"The initial position of dots in Qualisys frame: \n{dots_pos_init_np}")
    # Qualisys得到的marker点在soft坐标系下的三维位置
    dots_pos_soft = pos_in_soft(dots_pos_init_np, trans_matrix)
    dots_pos_soft_refined = np.zeros((POINTS_NUM, 3))
    dots_pos_soft_refined[:, 0] = 1.e-4
    dots_pos_soft_refined[:, 1] = dots_pos_soft[:, 1]  # 只保留y坐标，x, z坐标为0
    print(f'The initial position of dots in soft frame: \n{dots_pos_soft}')
    if dots_pos_init_np.shape[0] != POINTS_NUM:
        raise ValueError('Error dots number!')

    # ----- 初始化变形模型 -----
    obj_shape =  [0.15, 0.17] # dir_path / "Mesh/.msh"
    fix, contact = list(range(9)) + [89], [81] #[210, 224]
    gain = 5.e1
    params = {"E": 1.e5, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = SoftSurface(obj_shape, fix, contact, dots_pos_soft_refined, **params)
    marker_desired_load = np.load(f"{dir_path}/Data/marker_desired.npz")['marker_desired_soft']
    soft.marker_pos_desired.from_numpy(marker_desired_load)
    print(f"Desired marker positions: \n{marker_desired_load}")

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = spla.factorized(s_lhs_np)

    # ------ 模型和机器人都运动到初始状态 -----
    right_rob = URROb(500, "192.168.253.102")
    mov = np.array([0, -9., 27.]) / 1000
    rob_mov = np.hstack((mov, np.zeros(3)))
    right_rob.move_add_movel(pose_add=rob_mov, a=0.1, v=0.1)
    mov_time = 2.
    soft.contact_vel.from_numpy(mov.reshape(1, 3)/mov_time)
    for step in range(int(mov_time / soft.dt)):
        soft.substep(step)

    # save_vtu(f"{dir_path}/Mesh/plane.msh", soft.mesh.verts.pos.to_numpy(), f"{dir_path}/Mesh/plane_init.vtu")
    # exit()

    # ----- 设置机器人参数 -----
    right_rob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    right_rob.start_record_data(f'{dir_path}/Data/right_robot_data.csv')

    loss_list = []

    try:
        for step in range(400):
            print(f'Step: {step*soft.dt:.2f} ===================')
            dots_pos = await receive_qualysis(connection, selected_index)
            dots_pos_soft = pos_in_soft(dots_pos, trans_matrix)
            print(f'Detected points position: \n{dots_pos_soft}')

            soft.substep(step)
            soft.get_marker_pos()
            loss_tmp = soft.compute_dcontact(dots_pos_soft)
            print(f"Loss: {loss_tmp}")
            print(f"Error: {soft.error.to_numpy()}")

            dy_dcontact = soft.dy_contact.reshape(-1, 3)
            end_speed = -gain * dy_dcontact[soft.contact_particle_list]
            end_speed_compress = compress_vectors(end_speed, 0.08)
            soft.contact_vel.from_numpy(end_speed_compress)
            print(f"End speed: {end_speed_compress.flatten()}. Norm: {np.linalg.norm(end_speed_compress, axis=1)}")
            
            # ----- 机器人控制 -----
            rob_mov = end_speed_compress * soft.dt
            print(f"Right rob mov: {rob_mov[0,0], rob_mov[0,1], rob_mov[0,2]}") 
            # 确保上一步的运动执行完
            while True:
                if right_rob.rtde_c.getAsyncOperationProgress() < 0:
                    break
                else:
                    time.sleep(0.001)
            right_rob.move_add_movel_async([rob_mov[0,0], rob_mov[0,1], rob_mov[0,2], 0., 0., 0.], a=0.3, v=0.3)

            loss_list.append(loss_tmp)

    except KeyboardInterrupt:
        print("Stopping the data stream...")

    finally:
        right_rob.stop_movel()
        right_rob.exit_script()
        right_rob.stop_record_data()

        await connection.stream_frames_stop()
        print("Stop streaming...")

        np.savetxt(f'{dir_path}/Data/loss_list.csv', np.array(loss_list), fmt='%e', delimiter=',')


if __name__ == '__main__':
    asyncio.run(main())