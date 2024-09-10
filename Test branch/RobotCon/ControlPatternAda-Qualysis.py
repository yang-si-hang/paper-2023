"""
使用Adaptive Controller的方法控制软体上的图案变形
created at 2024-09-02 by hsy
"""

import os
import time
import asyncio
import pkg_resources
import qtm_rt
import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from scipy.stats import zscore
from RobAction import URROb
from ZedUtilize import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

POINTS_NUM:int = 4
# points_pos = np.zeros((POINTS_NUM, 3))

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip:str = '192.168.253.1'
qualysis_password:str = ''

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


def read_desired_pos(dot_pos_init):
    data = np.load('data/desired_pos.npz')
    pos_desired = data['desired_pos']
    pos_soft_desired = data['desired_pos_soft']
    tree = KDTree(dot_pos_init)
    _, indices = tree.query(pos_desired)
    ordered_pos_desired = pos_desired[indices]
    ordered_pos_soft_desired = pos_soft_desired[indices]
    print("The desired position in World frame: \n", np.array_str(ordered_pos_desired, precision=6))
    print('The desired position in Soft frame: \n', np.array_str(ordered_pos_soft_desired, precision=6))

    return ordered_pos_desired, ordered_pos_soft_desired


def pos_in_soft(pos:npt.NDArray)->npt.NDArray:
    # 将世界坐标系的位置转换到软体坐标系
    pos_soft = np.linalg.inv(trans_matrix) @ np.c_[pos, np.ones(POINTS_NUM)].T
    pos_soft = pos_soft[:3].T
    return pos_soft


def action_compress(vec:npt.NDArray, max_length:float=3.e-4):
    """
    Compress action vector in a safe range
    """
    length = np.linalg.norm(vec)

    if length > max_length:
        factor = max_length / length
        return vec * factor
    else:
        return vec


def cal_loss(dot_pos_soft:npt.NDArray, dot_pos_desired:npt.NDArray, Ja:npt.NDArray):
    # 在软体坐标系中计算损失，只取X-Y平面的误差
    error = dot_pos_soft - dot_pos_desired
    L = np.linalg.norm(error, axis=1) ** 2
    dL = 2*error.flatten()
    dL_dx = dL @ Ja
    # for idx, ele_idx in enumerate(self.marker_element):
    #     self.dL[ele_idx * dim] = 2 * error[0] * barycentric[idx]
    #     self.dL[ele_idx * dim + 1] = 2 * error[1] * barycentric[idx]

    return error, L, dL_dx


def update_jacobian(factor:float, delta_action, delta_pos, Ja:npt.NDArray):
    # factor = 1.e1

    a = Ja.flatten()
    W = np.zeros((2*POINTS_NUM, 4*POINTS_NUM))
    for idx in range(2*POINTS_NUM):
        W[idx, 2*idx] = delta_action[0]
        W[idx, 2*idx+1] = delta_action[1]

    e = W @ a - delta_pos.flatten()
    a -= factor * W.transpose() @ e

    # a: shape:(N*2, 2), e: shape:(N*4,)
    return a.reshape((2*POINTS_NUM, 2)), e, W @ a


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


async def main():
    learning_rate = 1.e1
    dt = 1./100
    # ja = np.tile(np.eye(2, 2), (POINTS_NUM, 1))
    ja = np.tile(np.ones((2, 2)), (POINTS_NUM, 1))

    # camera_id, image_init, window_name = init_camera()
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
    dots_pos_soft = pos_in_soft(dots_pos_init)
    print(f'The initial position of dots in soft frame: \n{dots_pos_soft}')
    if dots_pos_init.shape[0] != POINTS_NUM:
        raise ValueError('Error dots number!')
    dots_pos_desired, dots_pos_soft_desired = read_desired_pos(dots_pos_init)
    dots_initial_error = dots_pos_soft_desired[:, :2] - dots_pos_soft[:, :2]
    print(f'Dots Movement Error & its lenght:\n')
    for idx, error in enumerate(dots_initial_error):
        print(f'{error}; {np.linalg.norm(error)}')

    MyRob = URROb(500)
    MyRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    MyRob.start_record_data('experiment_data.csv')

    end_movement_np = np.zeros(2)

    dots_soft_list = []
    delta_pos_ada_list = []
    delta_pos_list = []
    ja_list = []
    ja_error_list = []
    loss_list = []
    rob_movement_list = []
    frame_name_list = []

    try:
        for step in range(200):
            print(f'Step {step} -------------------------------------')
            dots_data = await receive_qualysis(connection)
            # print(dots_data)
            dots_pos = [point for index, point in sorted(zip(dots_data['idx'], dots_data['pos']))]
            dots_pos = np.array(dots_pos)
            dots_pos_soft_new = pos_in_soft(dots_pos)
            print(f'Detected points position: \n{dots_pos_soft_new}')

            delta_pos = dots_pos_soft_new - dots_pos_soft
            dots_pos_soft = dots_pos_soft_new

            _, loss_tmp, dL_daction = cal_loss(dots_pos_soft[:,:2], dots_pos_soft_desired[:,:2], ja)
            # `end_movement_np` is i step, `delta_pos`= `i+1 step` - `i step`
            ja_new, ja_error, delta_pos_ada = update_jacobian(1.e5, end_movement_np, delta_pos[:,:2], ja)
            ja = ja_new

            end_speed_np = -learning_rate * dL_daction
            end_movement_np = end_speed_np * dt
            end_movement_np = action_compress(end_movement_np, 8.e-4)
            end_movement = end_movement_np.tolist()

            print(f'Loss items: {loss_tmp}; Loss sum: {np.sum(loss_tmp)}')
            print(f'The tool movement: {end_movement}')

            # 机器人控制
            MyRob.move_add_movel([end_movement[1], end_movement[0], 0, 0, 0, 0], a=0.1, v=0.1)            # 设置机械臂末端速度

            # 写入数据
            dots_soft_list.append(dots_pos_soft.flatten())
            delta_pos_ada_list.append(delta_pos_ada)
            delta_pos_list.append(delta_pos.flatten())
            ja_list.append(ja.flatten())
            ja_error_list.append(ja_error)
            loss_list.append(loss_tmp)
            rob_movement_list.append(end_movement)

            color_image = get_image(camera_id, image)
            cv2.imshow(window_name, color_image)
            frame_name_list = image_save(color_image, step, frame_name_list, output_folder)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping the data stream...")

    finally:
        MyRob.stop_movel()
        MyRob.exit_script()
        MyRob.stop_record_data()

        cv2.imwrite('task_complete.png', color_image)
        camera_id.close()
        cv2.destroyAllWindows()

        np.savetxt('dots_soft_list.csv', np.array(dots_soft_list), fmt='%.10f', delimiter=',')
        np.savetxt('delta_pos_ada_list.csv', np.array(delta_pos_ada_list), fmt='%.10f', delimiter=',')
        np.savetxt('delta_pos_list.csv', np.array(delta_pos_list), fmt='%.10f', delimiter=',')
        np.savetxt('ja_list.csv', np.array(ja_list), fmt='%.10f', delimiter=',')
        np.savetxt('ja_error_list.csv', np.array(ja_error_list), fmt='%.10f', delimiter=',')
        np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
        np.savetxt('rob_movement_list.csv', np.array(rob_movement_list), fmt='%.10f', delimiter=',')

        # 将保存的图像转换为视频
        image_to_video(frame_name_list, 'output_video.mp4')

    # 停止数据流
    await connection.stream_frames_stop()

if __name__ == '__main__':
    asyncio.run(main())