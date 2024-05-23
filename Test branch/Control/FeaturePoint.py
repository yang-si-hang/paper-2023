"""
Construct relationship between feature point and mesh node
"""


import numpy as np


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


p = np.array([0.5, 0.5])
mesh_nodes = np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.]])

print(feature_barycentric_coordinates(p, mesh_nodes))