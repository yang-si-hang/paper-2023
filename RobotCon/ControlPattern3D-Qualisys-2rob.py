"""
实物实验,使用Strain Constraint & Volume Constraint的DiffPD的3D版本进行控制
使用双机器人控制
created at 2024-10-24 by hsy
"""
import time
import os
from collections import defaultdict
# import cv2
import numpy as np
import numpy.typing as npt
from typing import List, Tuple
import json
import asyncio
import pkg_resources
import qtm_rt
import tifffile.geodb
from scipy.spatial import KDTree
from scipy.stats import zscore
import pyvista as pv
from ControlSim3d import *
from RobAction import URROb
from ZedUtilize import *
# from CVVideo import *
from CoordinateTransform import *
from Mesh.Visualize import mesh_visual

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# pattern中有几个点，重新赋值
POINTS_NUM:int = 4
selected_index = [36, 39, 41, 37]

QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip:str = '192.168.253.1'
qualysis_password:str = ''

trans_matrix = np.loadtxt('data/transformation_matrix.csv', delimiter=',')


def remove_outliers_and_get_center(positions:List)->Tuple[npt.NDArray, npt.NDArray]:
    # 将位置转换为 NumPy 数组以便进行计算
    positions_array = np.array(positions)
    # print(f"positions: {positions}")

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


async def init_maker(connection, step_num:int=50)->Tuple[dict, npt.NDArray]:
    # step_num:int = 50
    dots_dict = {key: [] for key in selected_index}         # 必须是指定的marker点
    for step in range(step_num):
        dot_data = await receive_qualysis(connection)
        # print(f"dot data: {dot_data}")
        for key, value in dot_data.items():
            dots_dict[key].append(value)
        # idxs = dot_data['idx']
        # for idx, idx_value in enumerate(idxs):
        #     if idx_value not in dots_dict:
        #         dots_dict[idx_value] = []
        #     dots_dict[idx_value].append(dot_data['pos'][idx])

    if not dots_dict:
        raise ValueError('No data captured!')

    centers = {}
    centers_list = []
    for index, positions in dots_dict.items():
        filtered_positions, center = remove_outliers_and_get_center(positions)
        centers[index] = center
        centers_list.append(center)

    # sorted_dots_dict = sorted(centers.items())
    # dots_center = np.array([pos for index, pos in sorted_dots_dict])

    return centers, np.array(centers_list)


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


# async def receive_qualysis(connection):
#     captured_data = {}
#
#     # Define the callback to capture data
#     def on_packet(packet):
#         nonlocal captured_data
#         header, markers = packet.get_3d_markers_no_label()
#         if header.marker_count != POINTS_NUM:
#             return
#
#         markers_pos = []
#         markers_idx = []
#         for idx, marker in enumerate(markers):
#             # 转换到单位米
#             markers_pos.append([marker.x/1000., marker.y/1000., marker.z/1000.])
#             markers_idx.append(marker.id)
#             # print(f"Marker: {marker.id}. Position: ({marker.x/1000.}, {marker.y/1000.}, {marker.z/1000.})")
#         captured_data['pos'] = markers_pos
#         captured_data['idx'] = markers_idx
#
#     await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)
#
#     return captured_data


async def receive_qualysis(connection)->dict:
    """
    selected_data: key[int]: marker's Qualisys index; pos[list]: marker's pos
    """
    captured_data = {}
    selected_data = {key: [] for key in selected_index}
    # selected_data = {'idx': [], 'pos': []}

    # Define the callback to capture data
    def on_packet(packet):
        nonlocal captured_data
        header, markers = packet.get_3d_markers_no_label()
        # if header.marker_count != POINTS_NUM:
        #     return

        markers_pos = []
        markers_idx = []
        for i, marker in enumerate(markers):
            # 转换到单位米
            markers_pos.append([marker.x/1000., marker.y/1000., marker.z/1000.])
            markers_idx.append(marker.id)
            # print(f"Marker: {marker.id}. Position: ({marker.x/1000.}, {marker.y/1000.}, {marker.z/1000.})")
        captured_data['pos'] = markers_pos
        captured_data['idx'] = markers_idx

    await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)

    # `capture_data`不包含重复的采样的数据，只有每个marker的一份数据
    for idx, pos in zip(captured_data['idx'], captured_data['pos']):
        if idx in selected_index:
            selected_data[idx] = pos
            # selected_data['idx'].append(idx)
            # selected_data['pos'].append(pos)

    return selected_data


def get_barycentrics(file_path:str, marker_pos_soft:npt.NDArray):
    assert marker_pos_soft.shape[0] == POINTS_NUM
    node_mesh_pos, element, _ = load_msh(file_path)
    cells = np.hstack((np.full((element.shape[0], 1), 4), element)).astype(np.int32)
    celltypes = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)
    mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, node_mesh_pos)

    marker_ele = np.zeros((POINTS_NUM, 4), dtype=int)
    barycentric = np.zeros((POINTS_NUM, 4), dtype=float)
    for i in range(POINTS_NUM):
        point = marker_pos_soft[i]
        print(f"The dot position in soft coordinate: {point}")
        # 从0开始编号
        cell_id = mesh.find_containing_cell(point)
        if cell_id >= 0:
            element_tmp = element[cell_id]
            cell_nodes = mesh.get_cell(cell_id).points
            marker_ele[i] = element_tmp
            barycentric[i] = feature_barycentric_coordinates_tet(point, cell_nodes)
            print("The dot is in element: ", element_tmp)
            print("The barycentric coordinates are: ", barycentric[i])
        else:
            print("The dot is not in the mesh object.")

    return marker_ele, barycentric


class MyObject(SoftObject):
    def __init__(self, shape:List[float], seed_size:float, mesh_file:str, marker_pos_init:npt.NDArray, contact_list:List[int], fixed_list:List[int]):
        super().__init__(shape, seed_size, mesh_file, contact_list, fixed_list)
        self.dt = 1. / 100
        self.gravity = 9.8
        self.loss = np.zeros(POINTS_NUM)
        self.node_init_pos_gravity = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_NUM)
        node_init_gravity_pos, _, _ = load_msh('Mesh/cube_new.msh')
        self.node_init_pos_gravity.from_numpy(node_init_gravity_pos)

        self.marker_elements = ti.Vector.field(4, dtype=ti.i32, shape=POINTS_NUM)
        self.barycentrics = ti.Vector.field(4, dtype=ti.f64, shape=POINTS_NUM)
        # dot_pos: 是模型的粒子状态; dot_pos_soft: 是软体坐标系下的粒子状态
        self.dot_pos = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        # self.dot_pos_soft = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_desired = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos.from_numpy(marker_pos_init)
        self.dot_pos_init.from_numpy(marker_pos_init)

        # self.get_marker_element()
        # self.read_desired_pos()
        self.read_desired_pattern()

        print(f"Deisred Marker pos: \n{self.dot_pos_desired.to_numpy()}")


    def get_marker_element(self, marker_pos_soft:npt.NDArray):
        cells = np.hstack((np.full((self.ELEMENT_NUM, 1), 4), self.element.to_numpy())).astype(np.int32)
        points = self.node_init_pos_gravity.to_numpy()
        celltypes = np.full(cells.shape[0], pv.CellType.TETRA, dtype=np.uint8)
        # np.savetxt(f"cell.csv", cells, fmt='%d', delimiter=',')
        # np.savetxt(f"points.csv", points, fmt='%.10f', delimiter=',')
        mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)

        for i in range(POINTS_NUM):
            point = marker_pos_soft[i]
            print(f"The dot position in soft coordinate: {point}")
            # 从0开始编号
            cell_id = mesh.find_containing_cell(point)
            if cell_id >= 0:
                element_tmp = self.element[cell_id]
                cell_nodes = mesh.get_cell(cell_id).points
                self.marker_elements[i] = element_tmp
                self.barycentrics[i] = feature_barycentric_coordinates_tet(point, cell_nodes)
                print("The dot is in element: ", element_tmp)
                print("The barycentric coordinates are: ", self.barycentrics[i])
            else:
                print("The dot is not in the mesh object.")


    def read_desired_pos(self):
        dot_pos_init = self.dot_pos_init.to_numpy()
        dot_pos_desired = dot_pos_init + np.array([-0.01, 0.0, -0.03])
        self.dot_pos_desired.from_numpy(dot_pos_desired)


    def read_desired_pattern(self):
        with open("data/desired_patrrern.json", "r") as json_file:
            data = json.load(json_file)
        dot_desired_pos = np.zeros((POINTS_NUM, 3), dtype=float)
        for i, pos in enumerate(data.values()):
            # print(f"Desired Marker pos: {pos}")
            dot_desired_pos[i] = pos

        self.dot_pos_desired.from_numpy(dot_desired_pos)
        # print(f"Desired Marker pos: \n{dot_desired_pos}")


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


    @ti.kernel
    def get_marker_model_pos(self):
        for i in range(POINTS_NUM):
            element = self.marker_elements[i]
            barycentric = self.barycentrics[i]
            point1, point2, point3 = self.node_pos[element[0]], self.node_pos[element[1]], self.node_pos[element[2]]

            dot_pos = barycentric[0] * point1 + barycentric[1] * point2 + barycentric[2] * point3 + barycentric[3] * self.node_pos[element[3]]
            self.dot_pos[i] = dot_pos


    def construct_L_soft(self, marker_pos_soft:npt.NDArray):
        """
        在Soft的二维坐标系上计算loss
        :param marker_pos_soft: shape: (POINTS_NUM, 3)
        """
        dim = self.dim
        error = np.zeros((POINTS_NUM, dim), dtype=np.float64)
        self.dL_dq.fill(0.)
        for marker_i in range(POINTS_NUM):
            barycentric = self.barycentrics[marker_i]
            desired_pos = self.dot_pos_desired[marker_i]
            current_pos = marker_pos_soft[marker_i]
            error[marker_i] = (current_pos - desired_pos).to_numpy()
            # print(f'Soft Coordinate Error： {error[marker_i]}')
            for idx, ele_idx in enumerate(self.marker_elements[marker_i]):
                self.dL_dq[ele_idx * dim] += 2 * (current_pos[0] - desired_pos[0]) * barycentric[idx]
                self.dL_dq[ele_idx * dim + 1] += 2 * (current_pos[1] - desired_pos[1]) * barycentric[idx]
                self.dL_dq[ele_idx * dim + 2] += 2 * (current_pos[2] - desired_pos[2]) * barycentric[idx]

        return error, np.linalg.norm(error, axis=1) ** 2


    def diff_pd(self, itr_num:int):
        self.partial_p()
        dA = self.rhs_dA.to_numpy()
        par_L = self.dL_dq.to_numpy()
        z_np = self.z.to_numpy()
        for itr in ti.static(range(itr_num)):
            rhs_diff_np = dA @ z_np + par_L
            z_new_np = self.pre_fact_lhs_solve(rhs_diff_np)
            z_np = z_new_np
        self.z.from_numpy(z_np)


    def substep(self, step_num:int=1):
        self.construct_sn()
        self.warm_start()
        for itr in range(self.solve_iteration):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        self.get_marker_model_pos()


    def compute_gradient(self, dot_sofa:npt.NDArray):
        error, loss_tmp = self.construct_L_soft(dot_sofa)
        # print(f"Error: {error}")
        # np.savetxt(f"dL_dq.csv", self.dL_dq.to_numpy(), fmt='%.10f', delimiter=',')
        self.loss = loss_tmp
        self.diff_pd(10)
        self.cal_ygrad()            # 得到dL/dy

        return error, loss_tmp


    def compute_action(self, contact_list:List=None):
        dim = self.dim
        contact_num = len(contact_list)
        contact_pos = np.zeros((contact_num, dim), dtype=np.float64)
        dL_dcnt = np.zeros((contact_num, dim), dtype=np.float64)
        for idx, idx_value in enumerate(contact_list):
            contact_pos[idx] = np.array([self.node_pos[idx_value][0], self.node_pos[idx_value][1], self.node_pos[idx_value][2]])
            dL_dcnt[idx] = np.array([self.dL_dy[idx_value*dim+0], self.dL_dy[idx_value*dim+1], self.dL_dy[idx_value*dim+2]])
        contact_center = np.mean(contact_pos, axis=0)
        print(f"dcnt:\n{dL_dcnt}")

        dL_dcnt_rel = dL_dcnt - np.mean(dL_dcnt, axis=0)
        dL_drob = np.zeros(6, dtype=np.float64)
        dL_drob[:3] = np.mean(dL_dcnt, axis=0)

        # 是否使用相对于中心点的相对速度，没有区别
        rot_tmp = np.zeros(3, dtype=np.float64)
        for i in range(self.contact_num):
            rot_tmp += np.cross(contact_pos[i] - contact_center, dL_dcnt_rel[i]) / np.dot(contact_pos[i] - contact_center, contact_pos[i] - contact_center) / self.contact_num

        dL_drob[3] = rot_tmp[0]
        dL_drob[4] = rot_tmp[1]
        dL_drob[5] = rot_tmp[2]

        return dL_drob


    def compute_action2(self)->Tuple[npt.NDArray, npt.NDArray]:
        # 双机器人的情况，前8个为左，后8个为右
        dim = self.dim
        num: int = 8
        contact_pos = np.zeros((self.contact_num, dim), dtype=np.float64)
        dL_dcnt = np.zeros((self.contact_num, dim), dtype=np.float64)
        for idx, idx_value in enumerate(self.contact_particles_list):
            contact_pos[idx] = np.array(
                [self.node_pos[idx_value][0], self.node_pos[idx_value][1], self.node_pos[idx_value][2]])
            dL_dcnt[idx] = np.array(
                [self.dL_dy[idx_value * dim + 0], self.dL_dy[idx_value * dim + 1], self.dL_dy[idx_value * dim + 2]])
        contact_center_left = np.mean(contact_pos[:8], axis=0)
        contact_center_right = np.mean(contact_pos[8:], axis=0)

        dL_dcnt_rel_left = dL_dcnt[:8] - contact_center_left
        dL_dcnt_rel_right = dL_dcnt[8:] - contact_center_right

        dL_drob_left = np.zeros(6, dtype=np.float64)
        dL_drob_right = np.zeros(6, dtype=np.float64)
        dL_drob_left[:3] = np.mean(dL_dcnt[:8], axis=0)
        dL_drob_right[:3] = np.mean(dL_dcnt[8:], axis=0)

        rot_tmp_left = np.zeros(3, dtype=np.float64)
        rot_tmp_right = np.zeros(3, dtype=np.float64)
        for i in range(num):
            rot_tmp_left += np.cross(contact_pos[i] - contact_center_left, dL_dcnt_rel_left[i]) / np.dot(
                contact_pos[i] - contact_center_left, contact_pos[i] - contact_center_left) / num
            rot_tmp_right += np.cross(contact_pos[i + 8] - contact_center_right, dL_dcnt_rel_right[i]) / np.dot(
                contact_pos[i + 8] - contact_center_right, contact_pos[i + 8] - contact_center_right) / num

        dL_drob_left[3:] = rot_tmp_left
        dL_drob_right[3:] = rot_tmp_right

        return dL_drob_left*num, dL_drob_right*num


async def main():
    learning_rate = 5.e-1
    obj_shape = [0.5, 0.5, 0.05]
    obj_seed_size = 0.05
    mesh_file = 'Mesh/cube_new_gravity.msh'
    contact_list = [8, 9, 10, 11, 32, 33, 34, 35] + [250, 251, 252, 253, 274, 275, 276, 277]
    # fixed_list = [250, 251, 252, 253, 274, 275, 276, 277]
    fixed_list = []

    camera_id, image = init_camera(720, 30)
    window_name = 'Zed Camera Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)

    connection = await qtm_rt.connect(qualysis_ip)
    if connection is None:
        print("Failed to connect")
        return
    async with qtm_rt.TakeControl(connection, qualysis_password):
        await connection.new()

    dots_pos_init_dict, dots_pos_init_np = await init_maker(connection)
    # Qualisys得到的marker点在soft坐标系下的三维位置
    # dots_pos_soft = pos_in_soft(dots_pos_init)
    # 需要与全局变量`selected_index`的顺序一致
    dots_init_pos_soft = np.array([[0.13648599,0.14154273,0.02978576],
                                [0.19588400,0.14142359,0.02978576],
                                [0.21274857,0.09284105,0.02978576],
                                [0.26320771,0.12490529,0.02978576]], dtype=float)
    print(f'The initial position of dots in Qualisys frame: \n{dots_pos_init_dict}')
    if dots_pos_init_np.shape[0] != POINTS_NUM:
        raise ValueError('Error dots number!')

    LeftRob = URROb(500, "192.168.253.101")
    LeftRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    LeftRob.start_record_data('left_robot_data.csv')
    left_t = np.array([0., 0.011, 0.068])
    left_R = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    # left_tool_trans = np.vstack((np.hstack((left_R, left_t.reshape(3, 1))), np.array([0., 0., 0., 1.])))
    # left_tcp = LeftRob.rtde_c.getTCPOffset()
    # print(f'The left robot TCP offset: {left_tcp}')

    RightRob = URROb(500, "192.168.253.102")
    RightRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    RightRob.start_record_data('right_robot_data.csv')

    soft = MyObject(obj_shape, obj_seed_size, mesh_file, dots_pos_init_np, contact_list, fixed_list)
    marker_ele_np, barycentric_np = get_barycentrics("Mesh/cube_new.msh", dots_init_pos_soft)
    soft.marker_elements.from_numpy(marker_ele_np)
    soft.barycentrics.from_numpy(barycentric_np)

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    loss_list = []
    dot_pos_list = []
    left_action_list = []
    right_action_list = []
    frame_name_list = []

    try:
        for step in range(700):
            print(f'Step: {step} ------------------------------------')
            dots_data_dict = await receive_qualysis(connection)
            dots_pos = [pos for pos in dots_data_dict.values()]
            # dots_pos = [point for index, point in sorted(zip(dots_data['idx'], dots_data['pos']))]
            dots_pos = np.array(dots_pos)
            print(f'Detected points position: \n{dots_data_dict}')

            soft.substep(1)
            dots_pos_model = soft.dot_pos.to_numpy()
            error_tmp, loss_tmp = soft.compute_gradient(dots_pos)
            print(f"Marker model position: \n{dots_pos_model}")

            drob_left, drob_right = soft.compute_action2()
            # print(f"drob: {-drob}")
            compressed_action_left = action_compress(-learning_rate*drob_left, 2.e-3)
            compressed_action_right = action_compress(-learning_rate*drob_right, 2.e-3)
            soft.apply_action2(np.concatenate((compressed_action_left, compressed_action_right)))

            left_action = compressed_action_left.tolist()
            right_action = compressed_action_right.tolist()
            print(f"Error: {error_tmp}")
            print(f'Loss items: {loss_tmp}; Loss sum: {np.sum(loss_tmp)}')
            print(f'The left tool movement: {left_action}; Length: {np.linalg.norm(left_action)}')
            print(f'The right tool movement: {right_action}; Length: {np.linalg.norm(right_action)}')

            # 机器人的tool坐标被修改过
            LeftRob.move_add_movel(left_action, a=0.1, v=0.1)
            RightRob.move_add_movel(right_action, a=0.1, v=0.1)

            dot_pos_list.append(dots_pos.flatten())
            left_action_list.append(left_action)
            right_action_list.append(right_action)
            loss_list.append(loss_tmp)

            color_image = get_image(camera_id, image, 'RIGHT')
            cv2.imshow(window_name, color_image)
            frame_name_list = image_save(color_image, step, frame_name_list, output_folder)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Stopping the data stream...")
        camera_id.close()

    finally:
        LeftRob.stop_movel()
        RightRob.stop_movel()
        LeftRob.exit_script()
        RightRob.exit_script()
        LeftRob.stop_record_data()
        RightRob.stop_record_data()

        # 停止数据流
        await connection.stream_frames_stop()

        cv2.imwrite('task_complete.png', color_image)
        camera_id.close()
        print(f"Camera closed...")
        cv2.destroyAllWindows()

        np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
        np.savetxt('dot_pos_list.csv', np.array(dot_pos_list), fmt='%.10f', delimiter=',')
        np.savetxt('left_action_list.csv', np.array(left_action_list), fmt='%.10f', delimiter=',')
        np.savetxt("right_action_list.csv", np.array(right_action_list), fmt='%.10f', delimiter=',')

        np.savetxt(f"node_final_pos.csv", soft.node_pos.to_numpy(), fmt="%.8f", delimiter=",")

        mesh_visual(soft.node_pos.to_numpy(), soft.element.to_numpy(), contact_list, fixed_list)

        # 将保存的图像转换为视频
        image_to_video(frame_name_list, 'output_video.mp4')


if __name__ == '__main__':
    asyncio.run(main())