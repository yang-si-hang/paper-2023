"""
使用pyvista可视化网格,并将变形过程渲染为视频
created at 2024-10-18 by hsy
"""

import numpy as np
import pyvista as pv
import os
import numpy.typing as npt
from typing import List
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
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


def compute_transformation_matrix(points:npt.NDArray) -> npt.NDArray:
    """_summary_
    通过8个接触点计算机器人的末端坐标系的变换矩阵
    Args:
        points (npt.NDArray): points is 2d array of 8 points, each defined as (x, y, z)

    Returns:
        npt.NDArray: transformation matrix of shape (4, 4)
    """
    # Calculate the center of the cube (mean of the 8 points)
    center = np.mean(points, axis=0)
    
    # Compute the basis vectors for the local coordinate system
    vec_x = points[4] - points[0]
    vec_y = points[2] - points[0]
    vec_z = points[1] - points[0]
    
    # Normalize the basis vectors
    vec_x /= np.linalg.norm(vec_x)
    vec_y /= np.linalg.norm(vec_y)
    vec_z /= np.linalg.norm(vec_z)
    
    # Construct the rotation matrix
    R = np.column_stack((vec_x, vec_y, vec_z))
    
    # Construct the homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    
    return T


def plot_coordinate_system(ax, T, length=1.0):
    origin = T[:3, 3]
    x_dir = T[:3, 0] * length
    y_dir = T[:3, 1] * length
    z_dir = T[:3, 2] * length
    
    ax.quiver(*origin, *x_dir, color='r', arrow_length_ratio=0.1)
    ax.quiver(*origin, *y_dir, color='g', arrow_length_ratio=0.1)
    ax.quiver(*origin, *z_dir, color='b', arrow_length_ratio=0.1)

    
def animate_coordinate_systems(transformations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    def update(num):
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Time step: {num}')
        plot_coordinate_system(ax, transformations[num])
    
    ani = animation.FuncAnimation(fig, update, frames=len(transformations), interval=200)
    plt.show()


if __name__ == '__main__':
    num:int = 8
    node_pos = np.loadtxt(f"{script_dir}/node_pos_final{num}.csv", delimiter=',')

    # elements 是 E x 4 的数组，表示 E 个四面体的节点索引
    elements = np.loadtxt(f'{script_dir}/element{num}.csv', delimiter=',', dtype=int)

    mesh_visual(node_pos, elements)

    # 可视化机器人末端的坐标系的运动过程
    # contact_pos_np = np.loadtxt(f"{script_dir}/contact_pos{num}.csv", delimiter=',')
    # contact_pos_reshape = contact_pos_np.reshape(-1, contact_pos_np.shape[1]//3, 3)

    # transformations = [compute_transformation_matrix(contact_pos_reshape[i, :, :])for i in range(200)]
    # animate_coordinate_systems(transformations)