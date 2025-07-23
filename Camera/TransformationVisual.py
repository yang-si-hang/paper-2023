"""
visualize the transformation matrix in 3D space
"""


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def arrow3D(ax, start, end, color='r'):
    arrow_length_ratio = 0.3
    arrow_head_width = 0.1

    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], color=color)

    direction = end - start
    length = np.linalg.norm(direction)
    direction = direction / length

    # Arrow head
    arrow_end = start + direction * (length - arrow_length_ratio * length)
    head = np.array([arrow_end,
                     arrow_end + direction * arrow_length_ratio * length * arrow_head_width])

    ax.plot([arrow_end[0], end[0]], [arrow_end[1], end[1]], [arrow_end[2], end[2]], color=color)

    for i in range(3):
        angle = np.pi / 6 * (i + 1)
        rotation = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        rotated_direction = np.dot(rotation, direction) * arrow_length_ratio * length * arrow_head_width
        arrow_side = np.array([end, end - rotated_direction])
        ax.plot(arrow_side[:, 0], arrow_side[:, 1], arrow_side[:, 2], color=color)


def plot_transform(ax, T, name=''):
    origin = T[:3, 3]
    x_axis = T[:3, 0]
    y_axis = T[:3, 1]
    z_axis = T[:3, 2]

    arrow3D(ax, origin, origin + x_axis, color='r')
    arrow3D(ax, origin, origin + y_axis, color='g')
    arrow3D(ax, origin, origin + z_axis, color='b')
    ax.text(*origin, name, size=12, zorder=1)


# 示例齐次变换矩阵
T1 = np.array([[0, 1, 0, 0],
               [1, 0, 0, 0],
               [0, 0, -1, 0],
               [0, 0, 0, 1]])

T2 = np.array( [[ 0.99645022, -0.04152775, -0.07322848, -0.00163096],
                 [ 0.05698976,  0.97298996,  0.22370227, -0.02939376],
                 [ 0.06196072, -0.22708145,  0.97190271,  0.23990188],
                 [ 0.,          0.,          0.,          1.,        ]])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plot_transform(ax, T1, 'T1')
plot_transform(ax, T1@np.linalg.inv(T2), 'T2')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
