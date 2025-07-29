"""
实物实验,使用多接触点进行控制pretension,使用Qualisys进行实时跟踪,
基于Strain Constraint的DiffPD
created at 2025-03-17 by hsy
"""
import time
import os, sys
import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple
import asyncio
import pkg_resources
import qtm_rt
from scipy import sparse
from scipy.stats import zscore
import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

script_dir = os.path.dirname(os.path.abspath(__file__))
# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from RobotCon._DiffPD2D import SoftObject2D, compress_vectors, line_from_points_2d
from RobAction import URROb
from Utilize.OrbbecShowUtilize import initialize_orbbec_camera, get_color_frame
from CoordinateTransform import *
os.chdir(script_dir)  # 修改当前工作目录


output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

selected_index = [629, 630]
POINTS_NUM = len(selected_index)

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip: str = '192.168.253.17'
qualysis_password: str = ''

# 实验之前需要标定软体坐标系，确定Qualisys到软体坐标系的变换矩阵
trans_matrix = np.loadtxt(f'{script_dir}/data/transformation_matrix.csv', delimiter=',')


def remove_outliers_and_get_center(positions:list):
    """移除异常值并计算中心点
    """
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
        center = np.array([np.nan, np.nan, np.nan])

    return filtered_positions, center


async def init_maker(connection, step_num:int=50)->Tuple[dict, npt.NDArray]:
    dots_dict = {key: [] for key in selected_index}         # 必须是指定的marker点
    for step in range(step_num):
        dot_data = await receive_qualysis(connection)
        # print(f"dot data: {dot_data}")
        for i, pos in enumerate(dot_data):
            marker_id = selected_index[i]
            dots_dict[marker_id].append(pos)

    if not any(dots_dict.values()):
        raise ValueError('No data captured!')

    centers = {}
    centers_np = np.full((len(selected_index), 3), np.nan) # 用NaN填充，确保形状一致
    for i, index in enumerate(selected_index):
        positions = dots_dict[index]
        if positions: # 确保这个index对应的数据存在
            filtered_positions, center = remove_outliers_and_get_center(positions)
            centers[index] = center
            centers_np[i] = center
        else:
            centers[index] = None # 或者你可以选择其他方式来处理缺失的数据，例如填充NaN

    return centers, centers_np


def pos_in_soft(pos:npt.NDArray)->npt.NDArray:
    # 将世界坐标系的位置转换到软体坐标系
    pos_soft = np.linalg.inv(trans_matrix) @ np.c_[pos, np.ones(POINTS_NUM)].T
    pos_soft = pos_soft[:3].T
    return pos_soft


async def receive_qualysis(connection)->npt.NDArray:
    """
    selected_data: key[int]: marker's Qualisys index; pos[list]: marker's pos
    """
    captured_data = {}
    selected_positions = np.full((len(selected_index), 3), np.nan) # 用NaN填充，确保形状一致

    # Define the callback to capture data
    def on_packet(packet):
        nonlocal captured_data
        header, markers = packet.get_3d_markers_no_label()

        markers_pos = []
        markers_idx = []
        for i, marker in enumerate(markers):
            # 转换到单位米
            markers_pos.append([marker.x/1000., marker.y/1000., marker.z/1000.])
            markers_idx.append(marker.id)
            captured_data[f"{marker.id}"] = [marker.x/1000., marker.y/1000., marker.z/1000.]

    await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)

    # `capture_data`不包含重复的采样的数据，只有每个marker的一份数据
    for i, idx in enumerate(selected_index):
        if f"{idx}" in captured_data:
            selected_positions[i] = captured_data[f"{idx}"]

    return selected_positions


def point_in_triangle(points, triangle_nodes):
    """检查一个或多个点是否在一个三角形内部，并计算重心坐标。

    Args:
        points: (N, 2) NumPy 数组，包含 N 个点的坐标。
        triangle_nodes: (3, 2) NumPy 数组，包含三角形顶点的坐标。

    返回:
        (N,) NumPy 数组，包含布尔值，表示每个点是否在三角形内部。
        (N, 3) NumPy 数组，包含每个点的重心坐标。
    """
    x, y = points[:, 0], points[:, 1]
    x1, y1 = triangle_nodes[0, 0], triangle_nodes[0, 1]
    x2, y2 = triangle_nodes[1, 0], triangle_nodes[1, 1]
    x3, y3 = triangle_nodes[2, 0], triangle_nodes[2, 1]

    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    
    if denominator == 0:
      return np.full(x.shape, False), np.full((x.shape[0],3), np.nan)

    u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    w = 1 - u - v

    is_inside = (u >= 0) & (v >= 0) & (w >= 0)
    barycentric_coords = np.stack((u, v, w), axis=-1)

    return is_inside, barycentric_coords


def find_triangle(points, nodes, triangles):
    """查找一个或多个点所在的三角形，并计算重心坐标。

    Args:
        points: (N, 2) NumPy 数组，包含 N 个点的坐标。
        nodes: (M, 2) NumPy 数组，包含所有节点坐标。
        triangles: (T, 3) NumPy 数组，包含三角形顶点索引。

    Returns:
        (N,) NumPy 数组，包含每个点所在的三角形索引，如果点不在任何三角形内部，则返回 -1。
        (N, 3) NumPy 数组，包含每个点的重心坐标，如果点不在任何三角形内部，则返回 NaN。
    """
    num_points = points.shape[0]
    num_triangles = triangles.shape[0]
    results = np.full(num_points, -1, dtype=int)
    barycentric_results = np.full((num_points, 3), np.nan)

    for i in range(num_triangles):
        triangle = triangles[i]
        triangle_nodes = nodes[triangle]
        is_inside, barycentric_coords = point_in_triangle(points, triangle_nodes)
        results[is_inside] = i
        barycentric_results[is_inside] = barycentric_coords[is_inside]

    return results, barycentric_results


class MyObject(SoftObject2D):
    def __init__(self, shape, fix, contact, marker_pos_init, E, nu, dt, density, **kwargs):
        super().__init__(shape, fix, contact, E, nu, dt, density, **kwargs)
        self.loss = 0.
        self.marker_elements = ti.field(dtype=ti.i32, shape=POINTS_NUM)
        self.barycentrics    = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        # dot_pos: 是模型的粒子状态; dot_pos_soft: 是软体坐标系下的粒子状态
        self.dot_pos      = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_soft = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos.from_numpy(marker_pos_init)
        self.dot_pos_init.from_numpy(marker_pos_init)

        self.get_marker_element()


    def get_marker_element(self):
        triangle_idx, bary = find_triangle(self.dot_pos_init.to_numpy(), self.node_pos_init.to_numpy(), self.ele.to_numpy())
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
            point1, point2, point3 = self.node_pos[element[0]], self.node_pos[element[1]], self.node_pos[element[2]]

            dot_pos = barycentric[0] * point1 + barycentric[1] * point2 + barycentric[2] * point3
            self.dot_pos[i] = dot_pos


    def construct_L_soft(self, marker_pos_soft:npt.NDArray, desired_gain:float=1.1):
        """将两个Marker点的距离拉伸到指定的比例
        """
        if marker_pos_soft.shape[0] != POINTS_NUM:
            raise ValueError('Error marker number!')
        
        self.dL_dq_contact.fill(0.)
        pos1, pos2 = marker_pos_soft[0,:], marker_pos_soft[1,:]

        feature_tmp = pos1 - pos2
        feature_desired = self.dot_pos_init[0].to_numpy() - self.dot_pos_init[1].to_numpy()
        feature_desired *= desired_gain
        dist_tmp = np.linalg.norm(feature_tmp)
        dist_desired = np.linalg.norm(feature_desired)

        a, b, c = line_from_points_2d(self.dot_pos_init[0].to_numpy(), self.dot_pos_init[1].to_numpy())
        line_normal = np.array([a, b], dtype=np.float64)

        # 两个点到直线的距离
        line_distance1 = line_normal.dot(pos1) + c
        line_distance2 = line_normal.dot(pos2) + c

        con_w = 5.e-1            # 加大约束权重
        loss = (dist_tmp - dist_desired) ** 2
        loss_con = line_distance1 ** 2 + line_distance2 ** 2
        loss_con *= con_w
        dL_dpos1 = 2*(dist_tmp - dist_desired) * (pos1 - pos2) / dist_tmp + 2 * con_w * line_distance1 * line_normal
        dL_dpos2 = 2*(dist_tmp - dist_desired) * (pos2 - pos1) / dist_tmp + 2 * con_w * line_distance2 * line_normal

        dL_dpos = np.row_stack((dL_dpos1, dL_dpos2))

        self.dL_dq_contact.fill(0.)
        for m_i in range(POINTS_NUM):
            ele_idx = self.marker_elements[m_i]
            ele = self.ele[ele_idx]
            for i in range(3):
                self.dL_dq_contact[ele[i] * self.dim]     += dL_dpos[m_i, 0] * self.barycentrics[m_i][i]
                self.dL_dq_contact[ele[i] * self.dim + 1] += dL_dpos[m_i, 1] * self.barycentrics[m_i][i]

        return loss, loss_con
    

    def compute_dcontact(self, dot_soft:npt.NDArray):
        loss_dist, loss_con = self.construct_L_soft(dot_soft, 1.1)
        self.construct_g_hessian()
        self.compute_z(10)

        print(f'Loss: {loss_dist}; Loss constraint: {loss_con}')
        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())

        return loss_dist, loss_con


async def main():
    # ----- 连接Orbbec相机 -----
    camera_pipeline = initialize_orbbec_camera(2560, 1440, 25)
    if not camera_pipeline:
        raise RuntimeError("相机初始化失败，请检查连接和参数设置")
    window_name = 'Orbbec Camera Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)
    frame = None

    # ----- 连接Qualisys -----
    connection = await qtm_rt.connect(qualysis_ip)
    if connection is None:
        print("Failed to connect")
        return
    async with qtm_rt.TakeControl(connection, qualysis_password):
        await connection.new()

    # ----- 初始化marker点 -----
    dots_pos_init_dict, dots_pos_init_np = await init_maker(connection)
    print(f"The initial position of dots in Qualisys frame: \n{dots_pos_init_np}")
    # Qualisys得到的marker点在soft坐标系下的三维位置
    dots_pos_soft = pos_in_soft(dots_pos_init_np)
    dots_pos_soft_2d = dots_pos_soft[:,:2]
    print(f'The initial position of dots in soft frame: \n{dots_pos_soft}')
    if dots_pos_init_np.shape[0] != POINTS_NUM:
        raise ValueError('Error dots number!')

    # ----- 初始化变形模型 -----
    obj_shape = "Mesh/shape_cut4.yaml" #[0.14, 0.14]
    # obj_shape = [0.11, 0.13]
    fix, contact = range(12), [132, 143] #[210, 224]
    gain = 5.e2
    params = {"E": 1.e5, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft_obj = MyObject(obj_shape, fix, contact, dots_pos_soft_2d, **params)
    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    # ----- 连接UR机器人 -----
    left_rob = URROb(500, "192.168.253.101")
    right_rob = URROb(500, "192.168.253.102")
    left_rob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    right_rob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    left_rob.start_record_data('left_robot_data.csv')
    right_rob.start_record_data('right_robot_data.csv')

    dots_pos_soft = dots_pos_soft_2d
    # dots_pos_model = soft_obj.dot_pos.to_numpy()
    # contact_speed_np = np.zeros((soft_obj.GRASP_N, 2), dtype=np.float64)
    # dots_soft_list = []
    # delta_pos_list = []
    # delta_pos_model_list = []
    loss_list = []
    loss_con_list = []
    # rob_movement_list = []
    # contact_pos_list = []
    # strain_sum_list = []
    frame_name_list = []

    frame = get_color_frame(camera_pipeline)
    cv2.imwrite('task_initial.png', frame)

    try:
        for step in range(100):
            time_start = time.time()
            print(f'Step: {step} ------------------------------------')
            dots_pos = await receive_qualysis(connection)
            # dots_pos = [point for index, point in sorted(zip(dots_data['idx'], dots_data['pos']))]
            # dots_pos = np.array(dots_pos)
            dots_pos_soft_new = pos_in_soft(dots_pos)[:,:2]
            print(f'Detected points position: \n{dots_pos_soft_new}')

            # delta_pos = dots_pos_soft_new - dots_pos_soft
            dots_pos_soft = dots_pos_soft_new

            soft_obj.substep(step)
            soft_obj.get_marker_pos()
            dots_pos_model_new = soft_obj.dot_pos.to_numpy()
            # delta_pos_model = dots_pos_model_new - dots_pos_model
            dots_pos_model = dots_pos_model_new

            loss_tmp, loss_con_tmp = soft_obj.compute_dcontact(dots_pos_soft)

            dy_dcontact = soft_obj.dy_contact.reshape(-1, 2)        # reshape到与接触点个数相同
            end_speed = -gain * dy_dcontact[soft_obj.contact_particle_list]
            end_speed_compress = compress_vectors(end_speed, 0.02)
            soft_obj.contact_vel.from_numpy(end_speed_compress)
            # print(f"dy_contact: {soft_obj.dy_contact}")
            print(f"End speed: {end_speed.flatten()}")

            # 机器人控制
            rob_mov = end_speed_compress * soft_obj.dt
            print(f"left rob mov: {rob_mov[0,0], rob_mov[0,1]}; right rob mov: {[-rob_mov[1,1], rob_mov[1,0]]}")
            # 确保上一步的运动执行完
            while True:
                if left_rob.rtde_c.getAsyncOperationProgress() < 0 and right_rob.rtde_c.getAsyncOperationProgress() < 0:
                    break
                else:
                    time.sleep(0.001)
            left_rob.move_add_movel_async([rob_mov[0,0], rob_mov[0,1] + 0.e-5, 0., 0., 0., 0.], a=0.3, v=0.3)
            right_rob.move_add_movel_async([-rob_mov[1,1] - 0.e-5, rob_mov[1,0], 0., 0., 0., 0.], a=0.3, v=0.3)

            # contact_pos_tmp = []
            # for idx in soft_obj.grasp_particle_list:
            #     contact_pos_tmp + soft_obj.node_pos[idx].to_numpy().tolist()

            # 写入数据
            # dots_soft_list.append(dots_pos_soft.flatten())
            # # delta_pos_list.append(delta_pos.flatten())
            # # delta_pos_model_list.append(delta_pos_model.flatten())
            loss_list.append(loss_tmp)
            loss_con_list.append(loss_con_tmp)
            # # rob_movement_list.append(end_move[0]+end_move[1])
            # contact_pos_list.append(contact_pos_tmp)
            # strain_sum_list.append(np.sum(soft_obj.elemnt_strain.to_numpy()))

            time_write = time.time()
            # frame = get_color_frame(camera_pipeline)
            # cv2.imshow(window_name, frame)
            # image_path = os.path.join(output_folder, f'frame_{step:04d}.png')
            # cv2.imwrite(image_path, frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

            time_end = time.time()
            # print(f"Time for frame: {time_end - time_write:.4f} seconds")
            print(f"Time: {time_end - time_start:.4f} seconds")

    except KeyboardInterrupt:
        print("Stopping the data stream...")

    finally:
        left_rob.stop_movel()
        right_rob.stop_movel()
        left_rob.exit_script()
        right_rob.exit_script()
        left_rob.stop_record_data()
        right_rob.stop_record_data()

        # 停止Qualisys数据流
        await connection.stream_frames_stop()
        print("Stop streaming...")

        frame = get_color_frame(camera_pipeline)
        if frame is not None:
            cv2.imwrite('task_complete.png', frame)
        camera_pipeline.stop()
        cv2.destroyAllWindows()

        # np.savetxt('dots_soft_list.csv', np.array(dots_soft_list), fmt='%.10f', delimiter=',')
        # np.savetxt('delta_pos_list.csv', np.array(delta_pos_list), fmt='%.10f', delimiter=',')
        # np.savetxt('delta_pos_model_list.csv', np.array(delta_pos_model_list), fmt='%.10f', delimiter=',')
        # np.savetxt('strain_sum_list.csv', np.array(strain_sum_list), fmt='%.10f', delimiter=',')
        np.savetxt('loss_list.csv', np.array(loss_list), fmt='%e', delimiter=',')
        np.savetxt('loss_con_list.csv', np.array(loss_con_list), fmt='%e', delimiter=',')
        # np.savetxt('rob_movement_list.csv', np.array(rob_movement_list), fmt='%.10f', delimiter=',')
        # np.savetxt('contact_pos_list.csv', np.array(contact_pos_list), fmt='%.10f', delimiter=',')
        # np.savetxt('final_strain_list.csv', soft_obj.elemnt_strain.to_numpy(), fmt='%.10f', delimiter=',')

        # 将保存的图像转换为视频
        # image_to_video(frame_name_list, 'output_video.mp4')


if __name__ == '__main__':
    asyncio.run(main())