import numpy as np
import numpy.typing as npt
from scipy.stats import zscore
from typing import Tuple

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

async def receive_qualysis(connection, selected_index)->npt.NDArray:
    """
    Args:
        connection: Qualisys connection object
        selected_index (list[int]): The list of marker indices to be selected.
    Returns:
        selected_positions (np.ndarray): shape (len(selected_index), 3)    
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

async def init_marker(connection, selected_index, step_num:int=50)->Tuple[dict, npt.NDArray]:
    dots_dict = {key: [] for key in selected_index}         # 必须是指定的marker点
    for step in range(step_num):
        dot_data = await receive_qualysis(connection, selected_index)
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