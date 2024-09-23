"""
将msh文件转换为三角形面片的obj文件，然后使用QEM技术对面片进行简化
msh文件顶点序号从1开始
created at 2024-09-19 by hsy
"""

import numpy as np
from collections import defaultdict
import open3d as o3d


def read_elements_from_msh(file_path:str):
    # 读取 .msh 文件中的四面体元素
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


def extract_surface_triangles(tetrahedrons: list):
    # 从四面体网格中提取几何体表面的三角形
    face_count = defaultdict(int)
    # A tetrahedron has 4 triangular faces, add each face to the dictionary
    for tet in tetrahedrons:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]
        for face in faces:
            face_count[face] += 1

    # Only keep the faces that appear exactly once (these are surface faces)
    surface_faces = [face for face, count in face_count.items() if count == 1]

    return np.array(surface_faces)


def filter_surface_vertices(vertices, surface_triangles):
    # Step 1: Extract the unique vertex indices used in the surface triangles
    unique_vertex_indices = sorted(set(surface_triangles.flatten()))

    # Step 2: Create a mapping from the original vertex indices to the new ones for surface vertices
    vertex_map = {old_idx: new_idx + 1 for new_idx, old_idx in enumerate(unique_vertex_indices)}

    # Step 3: Filter vertices to only include those used in the surface triangles
    surface_vertices = [vertices[i - 1] for i in unique_vertex_indices]  # Convert 1-based index to 0-based

    # Step 4: Update the surface triangles to use the new vertex indices
    new_surface_triangles = np.array([[vertex_map[v] for v in triangle] for triangle in surface_triangles])

    return np.array(surface_vertices), new_surface_triangles


def write_obj_file(vertices, faces, output_path):
    obj_lines = []

    # Add vertices to the OBJ file
    for vertex in vertices:
        obj_lines.append(f"v {vertex[0]} {vertex[1]} {vertex[2]}")

    # Add triangle faces to the OBJ file
    for face in faces:
        obj_lines.append(f"f {face[0]} {face[1]} {face[2]}")

    # Save the OBJ content to a new file
    with open(output_path, 'w') as obj_file:
        obj_file.write("\n".join(obj_lines))


def find_nearst_vertices(node_origin, node_simplify):
    # 网格简化后，找到原始网格中与简化网格中某个顶点最近的顶点
    nearst_node = []
    for node in node_simplify:
        distance = np.linalg.norm(np.array(node_origin) - np.array(node), axis=1)
        index = np.argmin(distance)
        nearst_node.append((index, node_origin[index]))

    return nearst_node


if __name__ == '__main__':
    nodes, elements = read_elements_from_msh('Mesh/liver.msh')
    surf_tri = extract_surface_triangles([el[1] for el in elements])
    surface_nodes, _ = filter_surface_vertices([[node[1], node[2], node[3]] for node in nodes], surf_tri)
    write_obj_file(surface_nodes, surf_tri, 'Mesh/liver-surface.obj')

    # 读取网格
    mesh = o3d.io.read_triangle_mesh("Mesh/liver-surface.obj")  # 替换为你的文件路径
    print("Original mesh has", len(mesh.vertices), "vertices and", len(mesh.triangles), "triangles")

    # 进行网格简化（目标三角面片数设定为 10000，可以根据需要调整）
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=80)

    print("Simplified mesh has", len(simplified_mesh.vertices), "vertices and", len(simplified_mesh.triangles),
          "triangles")

    o3d.io.write_triangle_mesh("Mesh/liver_surface_simplify.obj", simplified_mesh)

    # 可视化简化后的网格
    # o3d.visualization.draw_geometries([simplified_mesh])

    # 找到原始网格的表面节点与简化网格中某个顶点最近的顶点
    nearst_nodes = find_nearst_vertices(surface_nodes, simplified_mesh.vertices)

    # 找到简化的节点在原始网格中的索引
    corres_node = find_nearst_vertices([[node[1], node[2], node[3]] for node in nodes], [node[1] for node in nearst_nodes])

    print(f"Corresponding nodes index: {[node[0] for node in corres_node]}")