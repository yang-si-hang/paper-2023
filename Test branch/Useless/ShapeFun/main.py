"""
对目标函数的优化,使得三个点完成某个形状(我忘了),固定的变形雅可比矩阵
"""

import numpy as np
np.set_printoptions(precision=5, suppress=True)
from scipy import optimize
import matplotlib.pyplot as plt


A = np.array([0., 0.])
B = np.array([0., 1.])

P = np.array([1., 0.])


def compute_barycentric_coordinates(nodes, point):
    """
    Compute barycentric coordinates for a point inside a triangular element.

    Parameters:
    nodes (array-like): Coordinates of the triangular element nodes, shape (3, 2).
    point (array-like): Coordinates of the point of interest, shape (2,).

    Returns:
    bary_coords (array-like): Barycentric coordinates of the point, shape (3,).
    """
    # Extract the coordinates of the triangle nodes
    x1, y1 = nodes[0]
    x2, y2 = nodes[1]
    x3, y3 = nodes[2]

    # Compute the area of the triangle
    area = 0.5 * ((y2 - y3) * (x1 - x3) + (y3 - y1) * (x2 - x3))

    # Compute the barycentric coordinates
    bary_coords = np.zeros(3)
    bary_coords[0] = ((y2 - y3) * (point[0] - x3) + (x3 - x2) * (point[1] - y3)) / (2 * area)
    bary_coords[1] = ((y3 - y1) * (point[0] - x3) + (x1 - x3) * (point[1] - y3)) / (2 * area)
    bary_coords[2] = 1.0 - bary_coords[0] - bary_coords[1]

    return bary_coords


# Define the coordinates of the triangle nodes
nodes = np.vstack((A, B, P))

# Define the coordinates of the point of interest
point1 = np.array([1., 0.]) - 0.9*np.array([np.cos(0.), np.sin(0.)])
point2 = np.array([1., 0.]) - 0.9*np.array([np.cos(-np.pi/8), np.sin(-np.pi/8)])
point3 = np.array([1., 0.]) - 0.9*np.array([np.cos(-2*np.pi/8), np.sin(-2*np.pi/8)])
print('Points:', point1, point2, point3)

# Compute the barycentric coordinates
barycentric_coords1 = compute_barycentric_coordinates(nodes, point1)
barycentric_coords2 = compute_barycentric_coordinates(nodes, point2)
barycentric_coords3 = compute_barycentric_coordinates(nodes, point3)

print("Barycentric Coordinates:", barycentric_coords1, barycentric_coords2, barycentric_coords3)
plt.plot(nodes[:, 0], nodes[:, 1], 'o', color='black')
plt.plot(point1[0], point1[1], 'o', color='red')
plt.plot(point2[0], point2[1], 'o', color='red')
plt.plot(point3[0], point3[1], 'o', color='red')


def objective_fun(x):
    P_ = P + x
    nodes_ = np.vstack((A, B, P_))
    point1_ = nodes_.transpose() @ barycentric_coords1
    point2_ = nodes_.transpose() @ barycentric_coords2
    point3_ = nodes_.transpose() @ barycentric_coords3
    direction1 = (point2_ - point1_) / np.linalg.norm(point2_ - point1_)
    direction2 = (point2_ - point3_) / np.linalg.norm(point2_ - point3_)
    f = np.dot(direction1, direction2)

    return f


def inequality_constraint1(x):
    return x[0]


def inequality_constraint2(x):
    return x[0] - x[1]


inequality1 = {'type': 'ineq', 'fun': inequality_constraint1}
inequality2 = {'type': 'ineq', 'fun': inequality_constraint2}


def evaluate_fun(x):
    P_ = P + x
    nodes_ = np.vstack((A, B, P_))
    point1_ = nodes_.transpose() @ barycentric_coords1
    point2_ = nodes_.transpose() @ barycentric_coords2
    point3_ = nodes_.transpose() @ barycentric_coords3
    print(point1_-point1, point2_-point2, point3_-point3)
    direction1 = (point2_ - point1_) / np.linalg.norm(point2_ - point1_)
    direction2 = (point2_ - point3_) / np.linalg.norm(point2_ - point3_)
    f = np.dot(direction1, direction2)
    print('Directions:', direction1, direction2)
    print(f)

    plt.plot(point1_[0], point1_[1], 'o', color='green')
    plt.plot(point2_[0], point2_[1], 'o', color='green')
    plt.plot(point3_[0], point3_[1], 'o', color='green')
    plt.plot(P_[0], P_[1], 'o', color='blue')

    return point1_, point2_, point3_

result = optimize.minimize(objective_fun, np.array([0., 0.]),
                           constraints=[inequality1, inequality2])
print(result)
# evaluate_fun(result.x)
evaluate_fun(np.array([0.1, -0.05]))

plt.show()