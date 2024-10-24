"""
使用pyvista可视化网格,并将变形过程渲染为视频
created at 2024-10-18 by hsy
"""

import numpy as np
import pyvista as pv
import os
import numpy.typing as npt
script_dir = os.path.dirname(os.path.abspath(__file__))


def mesh_visual(node:npt.NDArray, ele:npt.NDArray):
    # Marker Point
    marker_points = node_pos[121]

    contact_list = [10, 11, 12, 13] + [32, 33, 34, 35]
    contact_points = node_pos[contact_list]

    fixed_list = [206, 207, 208, 209] + [228, 229, 230, 231]
    fixed_points = node_pos[fixed_list]

    # 设置点和单元
    num_tetra = elements.shape[0]
    cells = np.hstack([np.full((num_tetra, 1), 4), elements]).flatten()  # 每个四面体包含4个节点
    grid = pv.UnstructuredGrid(cells, np.full(num_tetra, pv.CellType.TETRA), node_pos)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(grid, opacity=1., color='skyblue', edge_color='black', show_edges=True)
    plotter.add_points(marker_points, color=[255, 0, 0], point_size=15, render_points_as_spheres=True)
    plotter.add_axes(interactive=True, line_width=2, color='black')
    plotter.show()


if __name__ == '__main__':
    num:int = 4
    node_pos = np.loadtxt(f"{script_dir}/node_pos_final{num}.csv", delimiter=',')

    # elements 是 E x 4 的数组，表示 E 个四面体的节点索引
    elements = np.loadtxt(f'{script_dir}/element{num}.csv', delimiter=',', dtype=int)

    mesh_visual(node_pos, elements)
