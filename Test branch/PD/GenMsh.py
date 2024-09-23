"""
使用pygmsh来划分网格,然后写入.msh文件
Gmsh软件使用指南：
1. Visibility的设置：`Tools` -> `Options` -> `Mesh` -> `Visibility`
2. 几何元素的颜色设置：`Tools` -> `Options` -> `Mesh` -> `Colors`
3. 全局颜色设置：`Tools` -> `Options` -> `General` -> `Colors`
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


def read_elements_from_msh(file_path):
    """
    读取 .msh 文件中的四面体元素。
    """
    nodes = []
    elements = []
    in_nodes_section = False
    in_elements_section = False

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('$NOD'):
                in_nodes_section = True
                continue
            elif line.startswith('$ENDNOD'):
                in_nodes_section = False
                continue
            elif line.startswith('$ELM'):
                in_elements_section = True
                continue
            elif line.startswith('$ENDELM'):
                in_elements_section = False
                continue

            if in_nodes_section:
                # 解析节点部分
                parts = line.strip().split()
                if len(parts) == 1:
                    # 这是节点的总数行，跳过
                    continue
                node_id = int(parts[0])
                x, y, z = map(float, parts[1:])
                nodes.append((node_id, x, y, z))

            if in_elements_section:
                # 解析四面体元素部分
                parts = line.strip().split()
                if len(parts) == 1:
                    # 这是单元的总数行，跳过
                    continue
                element_id = int(parts[0])
                node_ids = list(map(int, parts[-4:]))
                elements.append((element_id, node_ids))

    return nodes, elements


def generate_lines_and_surfaces(elements):
    """
    从四面体元素中生成线和面。
    """
    edges = set()
    faces = set()

    for _, (n1, n2, n3, n4) in elements:
        # 生成四面体的6条边
        edges.update([(min(n1, n2), max(n1, n2)),
                      (min(n1, n3), max(n1, n3)),
                      (min(n1, n4), max(n1, n4)),
                      (min(n2, n3), max(n2, n3)),
                      (min(n2, n4), max(n2, n4)),
                      (min(n3, n4), max(n3, n4))])

        # 生成四面体的4个三角形面
        face_list = [(n1, n2, n3), (n1, n2, n4), (n1, n3, n4), (n2, n3, n4)]
        for face in face_list:
            faces.add(tuple(sorted(face)))

    return list(edges), list(faces)


def write_new_msh(file_path, nodes, edges, faces, elements):
    """
    将节点、线和面信息写入新的 .msh 文件。
    """
    with open(file_path, 'w') as f:
        # 写入节点
        f.write("$NOD\n")
        f.write(f"{len(nodes)}\n")
        for node_id, x, y, z in nodes:
            f.write(f'{node_id} {x:.5f} {y:.5f} {z:.5f}\n')
        f.write("$ENDNOD\n")

        # 写入线
        f.write("$ELM\n")
        # f.write(f"{len(edges) + len(elements)}\n")
        f.write(f"{len(edges) + len(faces) + len(elements)}\n")

        for element_id, (n1, n2) in enumerate(edges, 1):
            f.write(f'{element_id} 1 1 1 2 {n1} {n2}\n')  # 1 表示线元素 2 似乎表示线的节点数
            
        # 写入面
        start_face_id = len(edges) + 1
        for element_id, (n1, n2, n3) in enumerate(faces, start_face_id):
            f.write(f'{element_id} 2 1 1 3 {n1} {n2} {n3}\n')  # 2 表示面元素 3 似乎表示面的节点数

        start_tet_id = len(edges) + len(faces)
        for element_id, (n1, n2, n3, n4) in elements:
            f.write(f'{element_id+start_tet_id} 4 1 1 4 {n1} {n2} {n3} {n4}\n') # 4 表示四面体元素 4 似乎表示四面体的节点数

        f.write("$ENDELM\n")

    print(f"New mesh with lines and surfaces written to {file_path}")


if __name__ == '__main__':
    # shape = [0.1, 0.02, 0.1]
    # file_path = 'Mesh/cube.msh'
    # generate_msh(shape, file_path)

    file_path = 'Mesh/liver.msh'
    nodes, elements = read_elements_from_msh(file_path)
    edges, faces = generate_lines_and_surfaces(elements)

    # 第四步：将节点、线和面信息写入新的 .msh 文件
    new_file_path = 'Mesh/cube_with_lines_and_surfaces.msh'
    write_new_msh(new_file_path, nodes, edges, faces, elements)