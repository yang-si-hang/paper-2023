"""
使用pygmsh来划分网格,然后写入.msh文件
Gmsh软件使用指南:
1. Visibility的设置:`Tools` -> `Options` -> `Mesh` -> `Visibility`
2. 几何元素的颜色设置：`Tools` -> `Options` -> `Mesh` -> `Colors`
3. 全局颜色设置：`Tools` -> `Options` -> `General` -> `Colors`
"""
import os
import pygmsh
import numpy as np
from typing import Tuple, List
import numpy.typing as npt
import yaml
import meshio
# import pyvista as pv
from scipy.spatial import Delaunay
import random


def mesh_obj_tri(obj_shape:List[float], seed_size:float)->Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """将二维对象生成三角形网格
    Args:
        obj_shape (List[float]): [length, width]
        seed_size (float): 网格尺寸
    Returns:
        Tuple[npt.NDArray[np.float64], npt.NDArray[np.int32], npt.NDArray[np.int32]]: 节点、边、单元
    """
    length, width = obj_shape

    length_n = int(length / seed_size)
    width_n = int(width / seed_size)

    length_n = length_n if abs(length - length_n * seed_size) < 1.e-6 else length_n + 1
    width_n = width_n if abs(width - width_n * seed_size) < 1.e-6 else width_n + 1

    xx, yy = np.meshgrid(np.linspace(0, length, length_n+1), np.linspace(0, width, width_n+1))
    xx_pad = xx.flatten('C')
    yy_pad = yy.flatten('C')
    node = np.array([xx_pad, yy_pad], dtype=float).T         # dim: N*2

    tri = Delaunay(node)
    element = np.sort(tri.simplices, axis=1)

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_temp = tuple(sorted(simplices[[i, (i + 1) % 3]]))
            edge_set.add(edge_temp)

    edge = np.array(list(edge_set), dtype=int)

    return node, edge, element


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


def generate_random_tetrahedra(v0, v1, v2, v3, v4, v5, v6, v7):
    # 定义多种长方体划分为四面体的方式,只适合`Z方向`仅有一个单位的情况
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


def generate_cube_msh(file_path:str, cube_shape:list, axis_seed:list):
    # 1. 生成均匀分布的采样节点
    x_gap, y_gap, z_gap = axis_seed  # 每个方向的间隙
    x_range, y_range, z_range = (0, cube_shape[0]), (0, cube_shape[1]), (0, cube_shape[2])  # 长方体范围

    x = np.arange(x_range[0], x_range[1] + x_gap, x_gap)
    y = np.arange(y_range[0], y_range[1] + y_gap, y_gap)
    z = np.arange(z_range[0], z_range[1] + z_gap, z_gap)

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

    # 3. 使用 PyVista 进行可视化，从0开始索引
    ugrid = pv.UnstructuredGrid(np.hstack(cells), np.array([10] * len(cells)), points)

    # 可视化四面体网格
    plotter = pv.Plotter()
    plotter.add_mesh(ugrid, show_edges=True, opacity=0.5)
    plotter.show_axes()
    plotter.show()

    cells_np = np.array(cells)
    # np.savetxt('node.csv', points, delimiter=',', fmt='%f')
    # np.savetxt('element.csv', cells_np[:,1:], delimiter=',', fmt='%d')

    write_msh2(file_path, points, cells_np[:,1:])


def read_mshv2_triangle(filename:str):
    """ read Gmsh file with version 2.2 and return

    Returns:
        nodes (npt.Ndarray): (N, 3)，存储节点坐标 \\
        triangles (npt.Ndarray): (T, 3)，存储三角形单元的节点索引(0-based)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # 去掉每行末尾的换行符，方便处理
    lines = [line.strip() for line in lines]

    # 1. 找到 "$Nodes" 段落并读取节点
    try:
        idx_nodes_start = lines.index('$Nodes')
    except ValueError:
        raise ValueError("未在文件中找到 $Nodes 段落，请检查文件格式。")

    # 读取节点数 N
    N = int(lines[idx_nodes_start + 1])
    # 读取 N 行节点数据
    node_data_lines = lines[idx_nodes_start + 2 : idx_nodes_start + 2 + N]

    # 将节点数据存储到一个列表中
    nodes = []
    for line in node_data_lines:
        parts = line.split()
        # parts[0] 是节点编号(1-based)，此处可忽略
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        nodes.append([x, y, z])
    nodes = np.array(nodes, dtype=float)

    # 2. 找到 "$Elements" 段落并读取单元
    try:
        idx_elements_start = lines.index('$Elements')
    except ValueError:
        raise ValueError("未在文件中找到 $Elements 段落，请检查文件格式。")

    # 读取单元数 M
    M = int(lines[idx_elements_start + 1])
    # 读取 M 行单元数据
    elem_data_lines = lines[idx_elements_start + 2 : idx_elements_start + 2 + M]

    # 用于存储三角形单元
    triangles = []

    for line in elem_data_lines:
        parts = line.split()
        # parts[0] 是单元编号
        elm_type = int(parts[1])      # 单元类型
        num_tags = int(parts[2])      # 标签数量

        # Gmsh v2 格式中，三角形的 elementType == 2
        if elm_type == 2:
            # 节点编号开始位置 = 3 + num_tags
            node_indices_start = 3 + num_tags
            # 对于三角形，后面有 3 个节点编号
            n1 = int(parts[node_indices_start])   - 1  # 转换为0-based索引
            n2 = int(parts[node_indices_start+1]) - 1
            n3 = int(parts[node_indices_start+2]) - 1
            triangles.append([n1, n2, n3])

    triangles = np.array(triangles, dtype=int)

    return nodes, triangles


def read_elements_from_msh2(file_path:str)->Tuple[List[tuple], List[tuple]]:
    """读取 .msh 文件中的四面体元素

    Args:
        file_path (str): .msh 文件路径

    Returns:
        Tuple[List[tuple], List[tuple]]:
            - nodes (List[tuple]): 每个元组格式为 (node_id, x, y, z)
            - elements (List[tuple]): 每个元组格式为 (element_id, [n1, n2, n3, n4])
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


def generate_edges_and_surfaces(elements:npt.NDArray)->Tuple[list, list]:
    """
    从四面体元素中生成线和面
    :param: elements: [n1, n2, n3, n4], dim: ele_numx4
    :return: edges: [(n1, n2), ...], faces: [(n1, n2, n3), ...]
    """
    edges = set()
    faces = set()

    for n1, n2, n3, n4 in elements:
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


def write_yaml(file_path:str, data:dict):
    """将数据写入yaml文件
    Args:
        file_path (str): yaml文件路径
        data (dict): 要写入的数据
    """
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, encoding='utf-8', allow_unicode=True)
    print(f"YAML data has been written to {file_path}")


def write_msh2(file_path:str, points:npt.NDArray, cells:npt.NDArray):
    """只将节点和四面体元素写入格式版本2的 .msh 文件
    Args:
        file_path (str): _description_
        points (npt.NDArray): [x, y, z], dim: node_num×3
        cells (npt.NDArray): [n1, n2, n3, n4], dim: ele_num×4
    """
    # 创建msh格式的数据
    msh_data = "$NOD\n{num_nodes}\n".format(num_nodes=points.shape[0])

    for i, point in enumerate(points):
        # msh_data += "{index} {x:.5f} {y:.5f} {z:.5f}\n".format(index=i+1, x=point[0], y=point[1], z=point[2])
        msh_data += f'{i+1} {point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n'

    msh_data += "$ENDNOD\n$ELM\n{num_elements}\n".format(num_elements=cells.shape[0])

    for i, cell in enumerate(cells):
        msh_data += f'{i+1} 4 1 1 4 {int(cell[0]+1):d} {int(cell[1]+1):d} {int(cell[2]+1):d} {int(cell[3]+1):d}\n'

    msh_data += "$ENDELM\n"

    # 将数据写入文件
    with open(file_path, 'w') as f:
        f.write(msh_data)

    print(f"Mesh data has been written to {file_path}")


def write_mshv2_tri(filename:str, nodes:npt.NDArray, triangles:npt.NDArray):
    """ 写入版本为2的三角形网格的.msh文件
    
    Args:
        nodes (npt.NDArray): 节点列表，每个元素为 (x, y) 或 (x, y, z)
        triangles (npt.NDArray): 三角形面片列表，每个元素为 (i, j, k), 假定索引从0开始
        filename (str): 输出文件名
    """
    with open(filename, "w") as f:
        # 写入 MeshFormat 部分
        f.write("$MeshFormat\n")
        # 版本号2.2，文件类型0（ASCII），数据大小8
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # 写入 Nodes 部分
        f.write("$Nodes\n")
        f.write("{}\n".format(len(nodes)))
        for i, node in enumerate(nodes, start=1):
            # 如果节点只有两个坐标，则默认 z=0.0
            if node.shape[0] == 2:
                x, y = node
                z = 0.0
            else:
                x, y, z = node
            f.write("{} {} {} {}\n".format(i, x, y, z))
        f.write("$EndNodes\n")
        
        # 写入 Elements 部分
        f.write("$Elements\n")
        f.write("{}\n".format(triangles.shape[0]))
        for i, tri in enumerate(triangles, start=1):
            # Gmsh中，三角形单元的类型为2
            # 此处 tags 数量设为0（可以根据需要增加物理区域等信息）
            # 注意：将0开始的索引转换为1开始
            n1, n2, n3 = tri
            f.write("{} 2 0 {} {} {}\n".format(i, n1+1, n2+1, n3+1))
        f.write("$EndElements\n")
        

def write_msh4(file_path:str, points, cells):
    """将节点、单元信息写入格式4.1版本的 .msh 文件
    """
    num_points = len(points)
    num_cells = len(cells)

    msh_data = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n$Nodes\n"
    msh_data += f"1 {num_points} 1 {num_points}\n"
    msh_data += f"3 1 0 {num_points}\n"

    for i, point in enumerate(points):
        msh_data += f'{i+1}\n'

    for point in points:
        msh_data += f"{point[0]:.5f} {point[1]:.5f} {point[2]:.5f}\n"

    msh_data += f"$EndNodes\n$Elements\n"
    msh_data += f"1 {num_cells} 1 600\n"
    msh_data += f"3 1 4 {num_cells}\n"

    for i, cell in enumerate(cells):
        msh_data += f'{i+1} {int(cell[0]+1):d} {int(cell[1]+1):d} {int(cell[2]+1):d} {int(cell[3]+1):d}\n'

    msh_data += "$EndElements\n"

    # 将数据写入文件
    with open(file_path, 'w') as f:
        f.write(msh_data)

    print(f"Mesh data has been written to {file_path}")


def write_obj(file_path:str, points:npt.NDArray, cells:npt.NDArray):
    """将面网格写入为obj格式; 不能用于保存体积网格
    """
    with open(file_path, 'w') as f:
        for point in points:
            f.write(f'v {point[0]:.10e} {point[1]:.10e} {point[2]:.10e}\n')
        for cell in cells:
            # 从1开始索引
            f.write(f'f {cell[0]+1:d} {cell[1]+1:d} {cell[2]+1:d}\n')

    print(f"Mesh data has been written to {file_path}")

def save_vtu(mesh_file:str, pos:npt.NDArray, write_name:str):
    """Save the node position to a .vtu file

    Args:
        mesh_file (str): The initial mesh file name
        pos (npt.NDArray): The node position
        write_name (str): The write file name
    """
    _, triangles = read_mshv2_triangle(mesh_file)

    cells_write = [("triangle", triangles)]
    mesh = meshio.Mesh(points=pos, cells=cells_write)
    mesh.write(f"{write_name}")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # generate_cube_msh('cube_new.msh', [0.5, 0.5, 0.03], [0.5/11, 0.5/11, 0.03])
    points, cells = read_elements_from_msh2('cube_new.msh')
    points_np = np.array([point[1:] for point in points])
    cells_np = np.array([cell[1] for cell in cells])

    cells_np = np.hstack([np.ones((cells_np.shape[0], 1)) * 4, cells_np-1]).astype(int)

    ugrid = pv.UnstructuredGrid(np.hstack(cells_np), np.array([10] * len(cells_np)), points_np)

    # 可视化四面体网格
    plotter = pv.Plotter()
    plotter.add_mesh(ugrid, show_edges=True, opacity=0.5)
    plotter.show_axes()
    plotter.show()