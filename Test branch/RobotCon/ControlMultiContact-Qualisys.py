"""
实物实验,使用多接触点进行控制,使用Qualisys进行实时跟踪,基于Strain Constraint & Volume Constraint的DiffPD
created at 2024-10-09 by hsy
"""

import time
import os
import cv2
import numpy as np
import numpy.typing as npt
import asyncio
import pkg_resources
import qtm_rt
from scipy.spatial import KDTree
from scipy.stats import zscore
from _ControlMultiPoints import *
from RobAction import URROb
from ZedUtilize import *
from CoordinateTransform import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# pattern中有几个点，重新赋值
POINTS_NUM: int = 1

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip: str = '192.168.253.1'
qualysis_password: str = ''

trans_matrix = np.loadtxt('data/transformation_matrix.csv', delimiter=',')

def remove_outliers_and_get_center(positions):
    # 将位置转换为 NumPy 数组以便进行计算
    positions_array = np.array(positions)

    # 计算每个维度的 Z-score
    z_scores = np.abs(zscore(positions_array, axis=0))

    # 设置 Z-score 的阈值，通常取 3
    threshold = 3
    # 找到所有 Z-score 小于阈值的索引，即正常数据的索引
    filtered_indices = np.all(z_scores < threshold, axis=1)

    # 筛选出正常数据
    filtered_positions = positions_array[filtered_indices]

    # 如果有足够的正常数据点，计算中心，否则返回空
    if len(filtered_positions) > 0:
        # 计算去除异常点后的中心点（平均值）
        center = np.mean(filtered_positions, axis=0)
    else:
        center = None

    return filtered_positions, center


async def init_maker(connection)->npt.NDArray:
    step_num:int = 50
    dots_dict = {}
    for step in range(step_num):
        dot_data = await receive_qualysis(connection)
        idxs = dot_data['idx']
        for idx, idx_value in enumerate(idxs):
            if idx_value not in dots_dict:
                dots_dict[idx_value] = []
            dots_dict[idx_value].append(dot_data['pos'][idx])

    if not dots_dict:
        raise ValueError('No data captured!')

    centers = {}
    for index, positions in dots_dict.items():
        filtered_positions, center = remove_outliers_and_get_center(positions)
        centers[index] = center

    sorted_dots_dict = sorted(centers.items())
    dots_center = np.array([pos for index, pos in sorted_dots_dict])

    return dots_center


def pos_in_soft(pos:npt.NDArray)->npt.NDArray:
    # 将世界坐标系的位置转换到软体坐标系
    pos_soft = np.linalg.inv(trans_matrix) @ np.c_[pos, np.ones(POINTS_NUM)].T
    pos_soft = pos_soft[:3].T
    return pos_soft


def action_compress(vec:npt.NDArray, max_length:float=3.e-4)->npt.NDArray:
    """
    Compress action vector in a safe range
    """
    length = np.linalg.norm(vec)

    if length > max_length:
        factor = max_length / length
        return vec * factor
    else:
        return vec


async def receive_qualysis(connection):
    captured_data = {}

    # Define the callback to capture data
    def on_packet(packet):
        nonlocal captured_data
        header, markers = packet.get_3d_markers_no_label()
        if header.marker_count != POINTS_NUM:
            return

        markers_pos = []
        markers_idx = []
        for idx, marker in enumerate(markers):
            # 转换到单位米
            markers_pos.append([marker.x/1000., marker.y/1000., marker.z/1000.])
            markers_idx.append(marker.id)
            # print(f"Marker: {marker.id}. Position: ({marker.x/1000.}, {marker.y/1000.}, {marker.z/1000.})")
        captured_data['pos'] = markers_pos
        captured_data['idx'] = markers_idx

    await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)

    return captured_data


class MyObject(SoftObject):
    def __init__(self, shape, seed_size, marker_pos_init, contact_idx: list):
        super().__init__(shape, seed_size, contact_idx)
        self.dt = 1. / 100
        self.loss = np.zeros(POINTS_NUM)
        self.marker_elements = ti.Vector.field(3, dtype=ti.i32, shape=POINTS_NUM)
        self.barycentrics = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        # dot_pos: 是模型的粒子状态; dot_pos_soft: 是软体坐标系下的粒子状态
        self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_soft = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_desired = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos.from_numpy(marker_pos_init)
        self.dot_pos_init.from_numpy(marker_pos_init)


    def get_marker_element(self):
        mesh_nodes = self.tri.points
        for i in range(POINTS_NUM):
            element_np = find_element(self.tri, self.dot_pos_init[i])
            if element_np is not None:
                self.marker_elements[i] = element_np
                self.barycentrics[i] = feature_barycentric_coordinates(self.dot_pos_init[i], mesh_nodes[element_np])
                print("The dot is in element: ", element_np)
                print("The barycentric coordinates are: ", self.barycentrics[i])
            else:
                print("The dot is not in the mesh object.")


    def read_desired_pos(self, user_defined:bool):
        if user_defined:
            self.dot_pos_desired[0] = self.dot_pos_init[0] + ti.Vector([0.01, 0.01])
        else:
            pass


    @ti.kernel
    def get_marker_pos(self):
        for i in range(POINTS_NUM):
            element = self.marker_elements[i]
            barycentric = self.barycentrics[i]
            point1, point2, point3 = self.node_pos[element[0]], self.node_pos[element[1]], self.node_pos[element[2]]

            dot_pos = barycentric[0] * point1 + barycentric[1] * point2 + barycentric[2] * point3
            self.dot_pos[i] = dot_pos


    def construct_L_soft(self, marker_pos_soft:npt.NDArray):
        """
        在Soft的二维坐标系上计算loss
        :param marker_pos_soft: shape: (POINTS_NUM, 2)
        """
        dim = self.dim
        error = np.zeros((POINTS_NUM, 2), dtype=np.float64)
        self.dL.fill(0.)
        for marker_i in range(POINTS_NUM):
            barycentric = self.barycentrics[marker_i]
            desired_pos = self.dot_pos_desired[marker_i]
            current_pos = marker_pos_soft[marker_i]
            error[marker_i] = (current_pos - desired_pos).to_numpy()
            # print(f'Soft Coordinate Error： {error[marker_i]}')
            for idx, ele_idx in enumerate(self.marker_elements[marker_i]):
                self.dL[ele_idx * dim] += 2 * (current_pos[0] - desired_pos[0]) * barycentric[idx]
                self.dL[ele_idx * dim + 1] += 2 * (current_pos[1] - desired_pos[1]) * barycentric[idx]

        return error, np.linalg.norm(error, axis=1) ** 2


    def diff_pd(self, itr_num:int):
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
            idx0, idx1 = i * self.dim, i * self.dim + 1
            self.grad_y[i].x = self.z[idx0] * self.node_mass[i] / self.dt ** 2
            self.grad_y[i].y = self.z[idx1] * self.node_mass[i] / self.dt ** 2


    def substep(self, step_num:int):
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
        self.get_marker_pos()

    def actuate_action(self, contact_speed:npt.NDArray):
        # for idx in range(self.GRASP_N):
        #     self.GRASP_VEL[idx] = contact_speed
        self.GRASP_VEL.from_numpy(contact_speed)


    def compute_gradient(self, dot_soft):
        error, loss_tmp = self.construct_L_soft(dot_soft)
        self.loss = loss_tmp
        self.diff_pd(10)
        self.compute_grad_y()

        return loss_tmp


async def main():
    obj_shape = [0.14, 0.14]
    obj_seed_size = 0.01
    learning_rate = 1.e1

    camera_id, image = init_camera(1080, 30)
    window_name = 'Zed Camera Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)

    connection = await qtm_rt.connect(qualysis_ip)
    if connection is None:
        print("Failed to connect")
        return
    async with qtm_rt.TakeControl(connection, qualysis_password):
        await connection.new()

    dots_pos_init = await init_maker(connection)
    # Qualisys得到的marker点在soft坐标系下的三维位置
    dots_pos_soft = pos_in_soft(dots_pos_init)
    dots_pos_soft_2d = dots_pos_soft[:,:2]
    print(f'The initial position of dots in soft frame: \n{dots_pos_soft}')
    if dots_pos_init.shape[0] != POINTS_NUM:
        raise ValueError('Error dots number!')

    left_rob = URROb(500, "192.168.253.101")
    right_rob = URROb(500, "192.168.253.102")
    left_rob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    right_rob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    left_rob.start_record_data('left_robot_data.csv')
    right_rob.start_record_data('right_robot_data.csv')

    soft_obj = MyObject(obj_shape, obj_seed_size, dots_pos_soft_2d, [15, 255])
    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    dots_pos_soft = dots_pos_soft_2d
    dots_pos_model = soft_obj.dot_pos.to_numpy()
    contact_speed_np = np.zeros((POINTS_NUM, 2), dtype=np.float64)
    dots_soft_list = []
    delta_pos_list = []
    delta_pos_model_list = []
    loss_list = []
    rob_movement_list = []
    contact_pos_list = []
    strain_sum_list = []
    frame_name_list = []

    try:
        for step in range(200):
            print(f'Step: {step} ------------------------------------')
            dots_data = await receive_qualysis(connection)
            dots_pos = [point for index, point in sorted(zip(dots_data['idx'], dots_data['pos']))]
            dots_pos = np.array(dots_pos)
            dots_pos_soft_new = pos_in_soft(dots_pos)[:,:2]
            print(f'Detected points position: \n{dots_pos_soft_new}')

            delta_pos = dots_pos_soft_new - dots_pos_soft
            dots_pos_soft = dots_pos_soft_new

            soft_obj.substep(1)
            dots_pos_model_new = soft_obj.dot_pos.to_numpy()
            delta_pos_model = dots_pos_model_new - dots_pos_model
            dots_pos_model = dots_pos_model_new
            loss_tmp = soft_obj.compute_gradient(dots_pos_soft)

            for idx in range(soft_obj.GRASP_N):
                contact_speed_tmp = -learning_rate * soft_obj.grad_y[soft_obj.grasp_particle_list[idx]].to_numpy()
                contact_speed_np[idx, :] = action_compress(contact_speed_tmp, 8.e-4)
            end_move = contact_speed_np.tolist()

            soft_obj.actuate_action(contact_speed_np / soft_obj.dt)
            print(f'Loss items: {loss_tmp}; Loss sum: {np.sum(loss_tmp)}')
            print(f'The tool movement: Left: {end_move[0]}; Right: {end_move[1]}')

            # 机器人控制
            left_rob.move_add_movel([end_move[0][1], end_move[0][0], 0., 0., 0., 0.], a=0.1, v=0.1)
            right_rob.move_add_movel([end_move[1][1], end_move[1][0], 0., 0., 0., 0.], a=0.1, v=0.1)

            # 写入数据
            dots_soft_list.append(dots_pos_soft.flatten())
            delta_pos_list.append(delta_pos.flatten())
            delta_pos_model_list.append(delta_pos_model.flatten())
            loss_list.append(loss_tmp)
            rob_movement_list.append(end_move[0,:]+end_move[1,:])
            contact_pos_list.append(soft_obj.node_pos[15].to_numpy().tolist()+soft_obj.node_pos[255].to_numpy().tolist())
            strain_sum_list.append(np.sum(soft_obj.elemnt_strain.to_numpy()))

            color_image = get_image(camera_id, image)
            cv2.imshow(window_name, color_image)
            frame_name_list = image_save(color_image, step, frame_name_list, output_folder)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping the data stream...")

    finally:
        left_rob.stop_movel()
        right_rob.stop_movel()
        left_rob.exit_script()
        right_rob.exit_script()
        left_rob.stop_record_data()
        right_rob.stop_record_data()

        cv2.imwrite('task_complete.png', color_image)
        camera_id.close()
        cv2.destroyAllWindows()

        np.savetxt('dots_soft_list.csv', np.array(dots_soft_list), fmt='%.10f', delimiter=',')
        np.savetxt('delta_pos_list.csv', np.array(delta_pos_list), fmt='%.10f', delimiter=',')
        np.savetxt('delta_pos_model_list.csv', np.array(delta_pos_model_list), fmt='%.10f', delimiter=',')
        np.savetxt('strain_sum_list.csv', np.array(strain_sum_list), fmt='%.10f', delimiter=',')
        np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
        np.savetxt('rob_movement_list.csv', np.array(rob_movement_list), fmt='%.10f', delimiter=',')
        np.savetxt('contact_pos_list.csv', np.array(contact_pos_list), fmt='%.10f', delimiter=',')

        np.savetxt('final_strain.csv', soft_obj.elemnt_strain.to_numpy(), fmt='%.10f', delimiter=',')

        # 将保存的图像转换为视频
        image_to_video(frame_name_list, 'output_video.mp4')

    # 停止数据流
    await connection.stream_frames_stop()

if __name__ == '__main__':
    asyncio.run(main())