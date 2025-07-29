""" Math function of numpy
created by hsy on 2025-07-20
"""
import numpy as np
import numpy.typing as npt

def compress_vectors(vectors:npt.NDArray, threshold:float)->npt.NDArray:
    """
    Args:
        vectors (np.ndarray): An N x 3 or 3 array of vectors
        threshold (float): The threshold value for the vector norms.

    Returns:
        np.ndarray: An N x 2 array of vectors after applying the compression.
    """
    # Check if vectors is 1D (single vector) or 2D (multiple vectors)
    if vectors.ndim == 1:
        # Single vector case
        norm = np.linalg.norm(vectors)
        scale = threshold / norm if norm > threshold else 1.0
        return vectors * scale
    else:
        # Multiple vectors case
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        scales = np.where(norms > threshold, threshold / norms, 1.0)
        return vectors * scales
    
def find_triangle(points, nodes, triangles):
    """查找一个或多个点所在的三角形，并计算重心坐标。

    Args:
        points: (N, 2) NumPy 数组，包含 N 个点的坐标。
        nodes: (M, 2) NumPy 数组，包含所有节点坐标。
        triangles: (T, 3) NumPy 数组，包含三角形顶点索引。

    Returns:
        (N,) NumPy 数组，包含每个点所在的三角形索引，如果点不在任何三角形内部，则返回 -1。
        (N, 3) NumPy 数组，包含每个点的重心坐标，如果点不在任何三角形内部，则返回 NaN。
    """
    num_points = points.shape[0]
    num_triangles = triangles.shape[0]
    results = np.full(num_points, -1, dtype=int)
    barycentric_results = np.full((num_points, 3), np.nan)

    for i in range(num_triangles):
        triangle = triangles[i]
        triangle_nodes = nodes[triangle]
        is_inside, barycentric_coords = point_in_triangle(points, triangle_nodes)
        results[is_inside] = i
        barycentric_results[is_inside] = barycentric_coords[is_inside]

    return results, barycentric_results

def point_in_triangle(points:npt.NDArray, triangle_nodes:npt.NDArray):
    """检查一个或多个点是否在一个三角形内部，并计算重心坐标

    Args:
        points (npt.NDArray(N, 2)): 包含 N 个点的坐标
        triangle_nodes (npt.NDArray(3, 2)): 包含三角形顶点的坐标

    Returns:
        (N,) NumPy 数组，包含布尔值，表示每个点是否在三角形内部
        (N, 3) NumPy 数组，包含每个点的重心坐标
    """
    x, y = points[:, 0], points[:, 1]
    x1, y1 = triangle_nodes[0, 0], triangle_nodes[0, 1]
    x2, y2 = triangle_nodes[1, 0], triangle_nodes[1, 1]
    x3, y3 = triangle_nodes[2, 0], triangle_nodes[2, 1]

    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    
    if denominator == 0:
      return np.full(x.shape, False), np.full((x.shape[0],3), np.nan)

    u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
    v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
    w = 1 - u - v

    is_inside = (u >= 0) & (v >= 0) & (w >= 0)
    barycentric_coords = np.stack((u, v, w), axis=-1)

    return is_inside, barycentric_coords