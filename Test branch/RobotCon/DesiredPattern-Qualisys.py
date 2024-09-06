"""
获得期望变形下的标记点位置,使用Qualisys捕捉标记球的位置作为期望位置
"""

import os
import asyncio

import cv2
import pkg_resources
import qtm_rt
import numpy as np
import numpy.typing as npt
from sklearn.cluster import KMeans
from scipy.stats import zscore
from collections import defaultdict
from ZedUtilize import *


QTM_FILE = pkg_resources.resource_filename("qtm_rt", "data/Demo.qtm")
qualysis_ip = '192.168.253.1'
qualysis_password = ''

POINTS_NUM:int = 4

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


async def receive_qualysis(connection)->list:
    captured_data = []

    # Define the callback to capture data
    def on_packet(packet):
        header, markers = packet.get_3d_markers_no_label()
        for idx, marker in enumerate(markers):
            pos = (marker.x, marker.y, marker.z)
            captured_data.append([marker.id, pos])

    await connection.stream_frames(components=["3dnolabels"], on_packet=on_packet)

    return captured_data


async def main():
    origin_pos = np.array([0.3815, 0.0117, 0.0810])
    x_deviation = np.array([0.3820, 0.1272, 0.0827])
    y_deviation = np.array([0.5201, 0.0114, 0.0784])

    camera_id, image = init_camera(1080, 30)
    window_name = 'CAPTURE DESIRED SHAPE'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)

    x_axis = x_deviation - origin_pos
    y_axis = y_deviation - origin_pos
    z_axis = np.cross(x_axis, y_axis)

    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T

    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = origin_pos + 0.0048 * z_axis - 0.015 / 2 * y_axis
    np.savetxt('data/transformation_matrix.csv', transformation_matrix, fmt='%.10f', delimiter=',')

    connection = await qtm_rt.connect(qualysis_ip)
    if connection is None:
        print("Failed to connect")
        return
    async with qtm_rt.TakeControl(connection, qualysis_password):
        await connection.new()

    step_num:int = 50
    dots_list = []
    for step in range(step_num):
        dot_pos = await receive_qualysis(connection)
        dots_list += dot_pos

    if not dots_list:
        raise ValueError('No data captured!')

    grouped_dots = defaultdict(list)
    for item in dots_list:
        index, pos = item
        grouped_dots[index].append(pos)

    centers = {}
    for index, positions in grouped_dots.items():
        filtered_positions, center = remove_outliers_and_get_center(positions)
        centers[index] = center / 1000.

    sorted_centers = sorted(centers.items())  # 按键（序号）排序
    print(f'Center positions: {sorted_centers}')
    dots_pos_center = np.array([center for index, center in sorted_centers])

    dots_pos_center_soft = transformation_matrix @ np.c_[dots_pos_center, np.ones(POINTS_NUM)].T
    dots_pos_center_soft = dots_pos_center_soft[:3].T

    print(f'Desired points position in Qualysis world frame: \n{np.array_str(dots_pos_center, precision=6, suppress_small=True)}')
    print(f'Desired points position in soft frame: \n{np.array_str(dots_pos_center_soft, precision=6, suppress_small=True)}')
    np.savez('data/desired_pos.npz', desired_pos=dots_pos_center, desired_pos_soft=dots_pos_center_soft)

    while True:
        color_image = get_image(camera_id, image)
        cv2.imshow(window_name, color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imwrite('data/desired_pos.png', color_image)
    camera_id.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    asyncio.run(main())