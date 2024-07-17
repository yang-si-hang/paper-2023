"""
关于坐标变换的一些函数
"""

import numpy as np
import numpy.typing as npt

def pixel_to_camera_coordinates(pixel, K_inv):
    """
    将图像坐标转换为相机坐标系中的射线
    """
    pixel_homogeneous = np.append(pixel, 1.0)
    camera_ray = K_inv @ pixel_homogeneous
    return camera_ray


def line_plane_intersection(line_dir, plane_normal, plane_point):
    """
    计算直线与平面的交点
    """
    d = np.dot(plane_normal, plane_point)
    t = d / np.dot(plane_normal, line_dir)          # 这里不需要对line_dir进行归一化
    intersection_point = t * line_dir
    return intersection_point


def dot_in_soft(dot_pixel, trans_soft, intrinsic:npt.NDArray):
    """
    将标记点的像素坐标转换到软体坐标系
    :return:
    """
    intrisic_inv = np.linalg.inv(intrinsic)
    dot_camera = pixel_to_camera_coordinates(dot_pixel, intrisic_inv)

    plane_normal = trans_soft[:3, 2]            # Z轴方向
    plane_point = trans_soft[:3, 3]             # 变换矩阵的平移部分

    dot_soft = line_plane_intersection(dot_camera, plane_normal, plane_point)       # 在相机坐标系下的三维位置
    dot_soft = np.linalg.inv(trans_soft) @ np.append(dot_soft, 1.)

    return dot_soft[0:2]


def dot_in_pixel(dot_soft, trans_soft, intrinsic):
    """
    将软体坐标系中的点转换到像素坐标
    """
    dot_soft = np.append(dot_soft, 0.)
    dot_camera = trans_soft @ np.append(dot_soft, 1.)
    dot_pixel = intrinsic @ dot_camera[:3]
    return dot_pixel[:2] / dot_pixel[2]


def feature_barycentric_coordinates(p, mesh_nodes):
    """
    Compute the barycentric coordinates of a point p with respect to the triangle p0, p1, p2
    """
    p0, p1, p2 = mesh_nodes
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])


def find_element(tri, dot_pos):
    """
    Find the element which contains the dot
    :param tri:
    :param dot_pos:
    :return: element index
    """
    # 查找包含点的三角形
    simplex = tri.find_simplex(dot_pos)

    if simplex != -1:
        # 返回包含点的三角形的顶点索引
        return tri.simplices[simplex]
    else:
        return None