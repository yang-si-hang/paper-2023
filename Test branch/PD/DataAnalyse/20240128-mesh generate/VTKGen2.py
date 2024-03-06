"""
This file is worked.
The indices of .vtk file are 0-offset.
The example .vtk files in websites: https://people.sc.fsu.edu/~jburkardt/data/vtk/vtk.html
"""


import numpy as np
from scipy.spatial import Delaunay
import os
script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


def mesh_data(shape, seed_size):
    L = shape[0]
    W = shape[1]
    # If the shape can be divided by seed_size, the remainder is 1, otherwise 0
    LN_remain = int(1) if np.mod(L, seed_size) < 1.e-8 else int(0)  # 1e-8 due to the precision problem
    WN_remain = int(1) if np.mod(W, seed_size) < 1.e-8 else int(0)
    LN = int(np.ceil(L / seed_size)) + LN_remain
    WN = int(np.ceil(W / seed_size)) + WN_remain

    xx, yy = np.meshgrid(np.linspace(0, L, LN), np.linspace(0, W, WN))
    xx_pad = xx.flatten()
    yy_pad = yy.flatten()
    node = np.array([xx_pad, yy_pad]).T

    tri = Delaunay(node)

    element = tri.simplices
    # element += 1

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_tmp = tuple(sorted([i, (i + 1) % 3]))
            edge_set.add(edge_tmp)

    edge = np.array(list(edge_set))

    data = {'v': node, 'e': edge, 'f': element}

    return data


# 定义一个函数来写入VTK文件
def write_vtk(file_path, points, polygons):
    with open(file_path, 'w') as file:
        # 写入VTK文件的头部
        file.write("# vtk DataFile Version 5.1\n")
        file.write("vtk output\n")
        file.write("ASCII\n")
        file.write("DATASET POLYDATA\n")

        # 写入点信息
        file.write(f"POINTS {len(points)} float\n")
        for point in points:
            file.write(f"{' '.join(map(str, point))} \n")

        # 写入多边形信息
        polygon_size = sum(len(polygon) + 1 for polygon in polygons)  # +1 因为每个多边形前都有一个表示顶点数的数字
        file.write(f"POLYGONS {len(polygons)} {polygon_size}\n")
        for polygon in polygons:
            file.write(f"{len(polygon)} {' '.join(map(str, polygon))}\n")


# 指定新VTK文件的路径
output_file_path = 'rect_triangular.vtk'

data = mesh_data(shape=[0.1, 0.1], seed_size=0.01)
vertices = data['v']
faces = data['f']

height = 0.
y_axis = np.ones(vertices.shape[0]) * height
vertices = np.column_stack((vertices[:, 0], vertices[:, 1], y_axis))
write_vtk(output_file_path, vertices, faces)
