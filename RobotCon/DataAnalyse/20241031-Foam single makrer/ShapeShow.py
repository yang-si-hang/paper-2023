"""
使用pyvista可视化网格,并将变形过程渲染为视频
created at 2024-10-18 by hsy
"""
import numpy as np
import pyvista as pv
import os
import numpy.typing as npt
from Utilize.GenMsh import write_msh2


def mesh_visual(node:npt.NDArray, ele:npt.NDArray):
    # Marker Point
    marker_points = node_pos[121]

    contact_list = [8, 9, 10, 11, 32, 33, 34, 35]
    contact_points = node_pos[contact_list]

    fixed_list = [250, 251, 252, 253, 274, 275, 276, 277]
    fixed_points = node_pos[fixed_list]

    # 设置点和单元
    num_tetra = elements.shape[0]
    cells = np.hstack([np.full((num_tetra, 1), 4), elements]).flatten()  # 每个四面体包含4个节点
    grid = pv.UnstructuredGrid(cells, np.full(num_tetra, pv.CellType.TETRA), node_pos)

    # 可视化
    plotter = pv.Plotter()
    plotter.add_mesh(grid, opacity=1., color='skyblue', edge_color='black', show_edges=True)
    plotter.add_points(contact_points, color='blue', point_size=15, render_points_as_spheres=True)
    plotter.add_points(fixed_points, color='green', point_size=15, render_points_as_spheres=True)
    # plotter.add_points(marker_points, color=[255, 0, 0], point_size=15, render_points_as_spheres=True)
    plotter.add_axes(interactive=True, line_width=2, color='black')
    plotter.show()


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    num:int = 2
    node_pos = np.loadtxt(f"node_final_pos{num}.csv", delimiter=',')

    # elements 是 E x 4 的数组，表示 E 个四面体的节点索引
    elements = np.loadtxt(f'cube_new_element.csv', delimiter=',', dtype=int)

    mesh_visual(node_pos, elements)


