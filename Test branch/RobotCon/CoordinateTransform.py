"""
关于坐标变换的一些函数
"""

import numpy as np
import numpy.typing as npt
from scipy.spatial import Delaunay


def pixel_to_camera_coordinates(pixel:npt.NDArray, K_inv):
    """
    将图像坐标转换为相机坐标系中的射线
    """
    pixel_homogeneous = np.append(pixel, 1.0)           # pixel shape: (2, )
    camera_ray = K_inv @ pixel_homogeneous
    return camera_ray                                   # camera_ray shape: (3, )


def line_plane_intersection(line_dir, plane_normal, plane_point)->npt.NDArray:
    """
    计算直线与平面的交点
    """
    d = np.dot(plane_normal, plane_point)
    t = d / np.dot(plane_normal, line_dir)          # 这里不需要对line_dir进行归一化
    intersection_point = t * line_dir
    return intersection_point


def dot_in_soft(dots_pixel:npt.NDArray, trans_soft:npt.NDArray, intrinsic:npt.NDArray)->npt.NDArray:
    """
    将标记点的像素坐标转换到软体的二维坐标系
    """
    intrisic_inv = np.linalg.inv(intrinsic)
    plane_normal = trans_soft[:3, 2]                # Z轴方向
    plane_point = trans_soft[:3, 3]                 # 变换矩阵的平移部分

    dots_soft_list = []

    dim = dots_pixel.ndim
    if dim == 2:
        for dot_pixel in dots_pixel:                    # dot_pixel shape: (POINTS_NUM, 2)
            dot_camera = pixel_to_camera_coordinates(dot_pixel, intrisic_inv)
            dot_soft = line_plane_intersection(dot_camera, plane_normal, plane_point)       # 在相机坐标系下的三维位置
            dot_soft = np.linalg.inv(trans_soft) @ np.append(dot_soft, 1.)

            dots_soft_list.append(dot_soft[:2])         # shape: (POINTS_NUM, 2)
        return np.array(dots_soft_list)
    else:
        dot_camera = pixel_to_camera_coordinates(dots_pixel, intrisic_inv)
        dot_soft = line_plane_intersection(dot_camera, plane_normal, plane_point)

        return dot_soft[:2]


def dot_in_pixel(dots_soft, trans_soft, intrinsic):
    """
    将软体坐标系中的点转换到像素坐标
    """
    dim = dots_soft.ndim
    if dim == 2:
        dots_pixel = np.zeros((dots_soft.shape[0], 2))
        for idx, dot_soft in enumerate(dots_soft):
            dot_soft = np.append(dot_soft, 0.)
            dot_camera = trans_soft @ np.append(dot_soft, 1.)
            dot_pixel = intrinsic @ dot_camera[:3]
            dots_pixel[idx, :] = dot_pixel[:2] / dot_pixel[2]
        return dots_pixel
    else:
        dot_soft = np.append(dots_soft, 0.)
        dot_camera = trans_soft @ np.append(dot_soft, 1.)
        dot_pixel = intrinsic @ dot_camera[:3]
        return dot_pixel[:2] / dot_pixel[2]


def feature_barycentric_coordinates(p, mesh_nodes)->npt.NDArray[np.float64]:
    """
    Compute the barycentric coordinates of a point p with respect to the triangle p0, p1, p2
    """
    p0, p1, p2 = mesh_nodes
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0                 # p shape: (2, )
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


def feature_barycentric_coordinates_tet(p: npt.NDArray[float], mesh_nodes: npt.NDArray[float]) -> npt.NDArray[np.float64]:
    """
    Compute the barycentric coordinates of a point p with respect to the tetrahedron p0, p1, p2, p3.

    Parameters:
    p : ndarray of shape (3,)
        The point for which the barycentric coordinates are to be calculated.
    mesh_nodes : ndarray of shape (4, 3)
        The vertices of the tetrahedron.

    Returns:
    ndarray of shape (4,)
        The barycentric coordinates of point p with respect to the tetrahedron.
    """
    p0, p1, p2, p3 = mesh_nodes
    # Vectors relative to p0
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p3 - p0
    vp = p - p0

    # Compute the determinant of the matrix formed by v0, v1, and v2
    d00 = np.dot(v0, np.cross(v1, v2))
    if d00 == 0:
        raise ValueError("The provided points do not form a valid tetrahedron.")

    # Compute the determinants for barycentric coordinates
    d1 = np.dot(vp, np.cross(v1, v2))
    d2 = np.dot(v0, np.cross(vp, v2))
    d3 = np.dot(v0, np.cross(v1, vp))

    # Calculate barycentric coordinates
    u = 1.0 - (d1 + d2 + d3) / d00
    v = d1 / d00
    w = d2 / d00
    t = d3 / d00

    return np.array([u, v, w, t])


def find_element(tri:Delaunay, dot_pos):
    """
    Find the element which contains the dot
    :param tri:
    :param dot_pos:
    :return: element node index
    """
    # 查找包含点的三角形
    simplex = tri.find_simplex(dot_pos)

    if simplex != -1:
        # 返回包含点的三角形的顶点索引
        return tri.simplices[simplex]           # shape: (3, )
    else:
        return None