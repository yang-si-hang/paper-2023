"""Update the mesh topology after cutting the mesh with a bounded plane (sector).
created by hsy at 2025-01-14
"""

import os, sys
import numpy as np
# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import mesh_obj_tri


def construct_sector_plane(center, v1, v2):
    """Construct a sector plane with center and two vectors.
    """
    normal = np.cross(v1, v2)
    normal /= np.linalg.norm(normal)

    # Plane equation: ax + by + cz + d = 0
    d = -np.dot(normal, center)
    plane_eq = (*normal, d)

    return plane_eq


def signed_distance(plane, points):
    a, b, c, d = plane
    return (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d)


def main():
    node_np, edge_np, ele_np = mesh_obj_tri([0.1, 0.1], 0.01)
    node_np = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))
    # np.savetxt("node_np.csv", node_np, fmt="%.6f", delimiter=",")
    # np.savetxt("edge_np.csv", edge_np, fmt="%d", delimiter=",")
    # np.savetxt("ele_np.csv", ele_np, fmt="%d", delimiter=",")

    sector_center = np.array([0.051, -0.001, 0.])
    r = 0.01
    theta = np.pi / 3
    v1 = np.array([0., r*np.cos(theta/2), r*np.sin(theta/2)])
    v2 = np.array([0., r*np.cos(theta/2), -r*np.sin(theta/2)])

    plane_eq = construct_sector_plane(sector_center, v1, v2)
    plane_normal = np.array(plane_eq[:3])

    distance = signed_distance(plane_eq, node_np)
    # np.savetxt("distance.csv", distance, fmt="%.6f", delimiter=",")

    intersect_edges = []
    for edge in edge_np:
        idx1, idx2 = edge
        p1, p2 = node_np[idx1], node_np[idx2]
        d1, d2 = distance[idx1], distance[idx2]
        if d1 * d2 <= 0:        # 穿过节点也视为切断
            # print(f"Cut edge: {edge} with distance {d1}, {d2}")
            t = abs(d1) / (abs(d1) + abs(d2))
            intersect = p1 + t * (p2 - p1)

            vec_to_center = intersect - sector_center
            if np.linalg.norm(vec_to_center) < 0.01:        # 弧长
                # print("Intersect point is in arc.")
                cross1 = np.cross(v1, vec_to_center)
                cross2 = np.cross(vec_to_center, v2)

                dot1 = np.dot(cross1, plane_normal)
                dot2 = np.dot(cross2, plane_normal)

                if dot1 > 0 and dot2 > 0:
                    print(f"Intersect point is in the sector. {edge}")
                    intersect_edges.append(tuple(edge))

    cutted_faces = []
    for face in ele_np:
        face_edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
        cut_edges = [edge for edge in face_edges if tuple(edge) in intersect_edges or tuple(reversed(edge)) in intersect_edges]

        if len(cut_edges) != 0:
            cutted_faces.append(face)

    print(f"Cutted faces: {cutted_faces}")

if __name__ == "__main__":
    main()
