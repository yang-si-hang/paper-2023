"""
# 使用pygmsh来划分网格,然后写入.msh文件
"""

import pygmsh
import numpy as np


def generate_msh(shape, mesh_size, file_path):
    # 创建立方体的几何形状
    with pygmsh.occ.Geometry() as geom:
        length, width, height = shape
        geom.add_box([0, 0, 0], [length, width, height], mesh_size=mesh_size)
        
        mesh = geom.generate_mesh()

    # 获取节点和单元信息
    points = mesh.points
    cells = mesh.cells_dict['tetra']

    # 创建msh格式的数据
    msh_data = "$NOD\n{num_nodes}\n".format(num_nodes=len(points))

    for i, point in enumerate(points):
        # msh_data += "{index} {x:.5f} {y:.5f} {z:.5f}\n".format(index=i+1, x=point[0], y=point[1], z=point[2])
        msh_data += f'{i+1} {point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n'

    msh_data += "$ENDNOD\n$ELM\n{num_elements}\n".format(num_elements=len(cells))

    for i, cell in enumerate(cells):
        msh_data += f'{i+1} 4 1 1 4 {int(cell[0]+1):d} {int(cell[1]+1):d} {int(cell[2]+1):d} {int(cell[3]+1):d}\n'

    msh_data += "$ENDELM\n"

    # 将数据写入文件
    with open(file_path, 'w') as f:
        f.write(msh_data)

    print(f"Mesh data has been written to {file_path}")


if __name__ == '__main__':
    shape = [0.1, 0.02, 0.1]
    file_path = 'Mesh/cube.msh'
    generate_msh(shape, file_path)