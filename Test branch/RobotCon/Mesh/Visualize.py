"""
可视化网格划分
如果内部平面存在冲突(不是平面交叉,对角线交叉),会出现深色区域
created at 2024-10-17 by hsy
"""
import os
import numpy as np
import numpy.typing as npt
from typing import List
import pyvista as pv
import random
from Utilize.GenMsh import write_msh2


def mesh_visual(node:npt.NDArray, ele:npt.NDArray, contact_list:List[int], fix_list:List[int]):
    """
    使用 PyVista 进行网格可视化
    """
    # Marker Point
    marker_points = node[121]

    # contact_list = [8, 9, 10, 11, 32, 33, 34, 35]
    contact_points = node[contact_list]

    # fixed_list = [250, 251, 252, 253, 274, 275, 276, 277]
    fixed_points = node[fix_list]

    # 设置点和单元
    num_tetra = ele.shape[0]
    cells = np.hstack([np.full((num_tetra, 1), 4), ele]).flatten()  # 每个四面体包含4个节点
    grid = pv.UnstructuredGrid(cells, np.full(num_tetra, pv.CellType.TETRA), node)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(grid, opacity=1., color='skyblue', edge_color='black', show_edges=True)
    if contact_list:
        plotter.add_points(contact_points, color='blue', point_size=15, render_points_as_spheres=True)
    if fix_list:
        plotter.add_points(fixed_points, color='green', point_size=15, render_points_as_spheres=True)
    # plotter.add_points(marker_points, color=[255, 0, 0], point_size=15, render_points_as_spheres=True)
    plotter.add_axes(interactive=True, line_width=2, color='black')
    plotter.show()


def generate_random_tetrahedra(v0, v1, v2, v3, v4, v5, v6, v7):
    """
    定义多种长方体划分为四面体的方式
    """
    options = [
        # 划分方案 1，选择对角线 v1-v6，相对于 v0-v7
        [[4, v0, v1, v2, v6], [4, v1, v2, v3, v7], [4, v1, v2, v6, v7], [4, v0, v1, v5, v6], [4, v1, v5, v6, v7], [4, v0, v4, v5, v6]],
        # # 划分方案 2，选择对角线 v0-v7，相对于 v1-v6
        # [[4, v0, v1, v5, v7], [4, v0, v4, v6, v7], [4, v0, v4, v5, v7], [4, v0, v1, v3, v7], [4, v0, v3, v6, v7], [4, v0, v2, v3, v6]],
        # 划分方案 3，选择对角线 v2-v5，相对于 v3-v4
        [[4, v1, v2, v3, v5], [4, v0, v2, v4, v5], [4, v0, v1, v2, v5], [4, v2, v3, v5, v7], [4, v2, v4, v5, v6], [4, v2, v5, v6, v7]]
        # 划分方案 4，选择对角线 v3-v4，相对于 v2-v5
        # [[4, v0, v1, v3, v4], [4, v1, v3, v4, v7], [4, v1, v4, v5, v7], [4, v0, v2, v3, v4], [4, v3, v4, v6, v7], [4, v2, v3, v4, v6]]
    ]
    return random.choice(options)


if __name__ == '__main__':
    script_path = os.path.dirname(os.path.abspath(__file__))

    # 1. 生成均匀分布的采样节点
    x_gap, y_gap, z_gap = 0.05, 0.05, 0.03  # 每个方向的间隙
    x_range, y_range, z_range = (0, 0.395), (0, 0.27), (0, 0.03)  # 长方体范围

    x = np.linspace(x_range[0], x_range[1], 11+1)
    y = np.linspace(y_range[0], y_range[1], 11+1)
    z = np.linspace(z_range[0], z_range[1], 1+1)

    # x = np.arange(x_range[0], x_range[1] + x_gap, x_gap)
    # y = np.arange(y_range[0], y_range[1] + y_gap, y_gap)
    # z = np.arange(z_range[0], z_range[1] + z_gap, z_gap)

    # 先排最后一个维度，再排倒数第二个维度，最后排第一个维度
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    # 2. 生成规则的四面体网格
    cells = []
    num_x, num_y, num_z = len(x) - 1, len(y) - 1, len(z) - 1

    # 遍历每个小长方体，生成随机划分的四面体
    for i in range(num_x):
        for j in range(num_y):
            for k in range(num_z):
                # 获取小长方体的8个顶点索引
                # v0=[0,0,0]; v1=[0,0,1]; v2=[0,1,0]; v3=[0,1,1]; v4=[1,0,0]; v5=[1,0,1]; v6=[1,1,0]; v7=[1,1,1]
                v0 = i * (num_y + 1) * (num_z + 1) + j * (num_z + 1) + k
                v1 = v0 + 1
                v2 = v0 + (num_z + 1)
                v3 = v2 + 1
                v4 = v0 + (num_y + 1) * (num_z + 1)
                v5 = v4 + 1
                v6 = v4 + (num_z + 1)
                v7 = v6 + 1

                # 随机选择一种划分方式
                tetrahedra = generate_random_tetrahedra(v0, v1, v2, v3, v4, v5, v6, v7)
                cells.extend(tetrahedra)

    # 3. 使用 PyVista 进行可视化
    # 创建一个 PyVista 的 UnstructuredGrid 对象
    ugrid = pv.UnstructuredGrid(
        np.hstack(cells), np.array([10] * len(cells)), points
    )

    # 可视化四面体网格
    plotter = pv.Plotter()
    plotter.add_mesh(ugrid, show_edges=True, opacity=0.5)
    plotter.show_axes()
    plotter.show()

    cells_np = np.array(cells)
    np.savetxt(f"{script_path}/foam_small_ele.csv", cells_np[:,1:], fmt='%d', delimiter=",")
    write_msh2(f'{script_path}/foam_small.msh', points, cells_np[:,1:])