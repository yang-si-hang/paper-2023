"""
This file for plot the result (\partial x/ \partial y) of the _DiffPDConstraint2.py
Two figures: one is leftside positional; another is downside positional. Variable
<fix_flag_temp> need to be changed to plot the corresponding figure.
"""

import numpy as np
import matplotlib.pyplot as plt


# Case one ---------------------------------------------------------------------------------------
dx_dy = np.loadtxt('dx_dy.csv', delimiter=',')
node_dis = np.diag(dx_dy)

Nx, Ny = 11, 11

node_dis_x = node_dis[::2]
node_dis_y = node_dis[1::2]

node_dis_x_reshape = node_dis_x.reshape(Nx, Ny)
node_dis_y_reshape = node_dis_y.reshape(Nx, Ny)
node_dis_norm = np.sqrt(node_dis_x_reshape**2 + node_dis_y_reshape**2)

# 2D 热力图
fig, ax = plt.subplots(figsize=(8, 6))
plt.axis('off')
plt.grid(True)

plt.imshow(node_dis_norm, cmap='viridis', interpolation='nearest')
plt.title('Node Deformation Ability', fontdict={'family': 'Times New Roman', 'size': 20})
cbar = plt.colorbar()
# cbar.ax.tick_params(labelsize=15)
for l in cbar.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
    l.set_size(16)

plt.show()


# # Case two -----------------------------------------------------------------------------------------
# dx_dy = np.loadtxt('dx_dy.csv', delimiter=',')
# idx = 42
# idx_x = idx*2
# idx_y = idx*2+1
# x_row_sum = dx_dy[idx_x,:]
# y_row_sum = dx_dy[idx_y,:]
#
# Nx, Ny = 11, 11
#
# x_row_sum_x = x_row_sum[::2]
# x_row_sum_y = x_row_sum[1::2]
#
# y_row_sum_x = y_row_sum[::2]
# y_row_sum_y = y_row_sum[1::2]
#
# x_row_sum_reshape = np.column_stack((x_row_sum_x, y_row_sum_x))
# y_row_sum_reshape = np.column_stack((x_row_sum_y, y_row_sum_y))
#
# d = np.array([1,-1])
# normalized_d = d / np.linalg.norm(d)
#
# weight_x = np.zeros(Nx*Ny)
# weight_y = np.zeros(Nx*Ny)
# weight_sum = np.zeros(Nx*Ny)
# weight_angle = np.zeros(Nx*Ny)
#
# # 计算每一对向量的权重
# for idx, (u, v) in enumerate(zip(x_row_sum_reshape, y_row_sum_reshape)):
#     if np.abs(np.sum(u)) < 1e-6 or np.abs(np.sum(v)) < 1e-6:
#         # weight_x[idx], weight_y[idx] = 0., 0.
#         continue
#     weight_x[idx], weight_y[idx] = np.linalg.solve(np.column_stack((u, v)), normalized_d)
#     weight_sum[idx] = 1. / np.sqrt(weight_x[idx]**2 + weight_y[idx]**2)
#     weight_angle[idx] = np.arctan2(weight_y[idx], weight_x[idx])
#
# weight_sum_x = weight_sum * np.cos(weight_angle)
# weight_sum_y = weight_sum * np.sin(weight_angle)
# weight_sum_reshape = weight_sum.reshape(Nx, Ny)
# weight_sum_x_reshape = weight_sum_x.reshape(Nx, Ny)
# weight_sum_y_reshape = weight_sum_y.reshape(Nx, Ny)
#
# # 2D 热力图
# fig, ax = plt.subplots(figsize=(16, 6))
# plt.axis('off')
# plt.grid(True)
#
# plt.subplot(1, 2, 1)
# plt.imshow(weight_sum_reshape, cmap='viridis', interpolation='nearest')
# plt.title('Node Deformation Action Sum', fontdict={'family': 'Times New Roman', 'size': 20})
# cbar = plt.colorbar()
# # cbar.ax.tick_params(labelsize=15)
# for l in cbar.ax.yaxis.get_ticklabels():
#     l.set_family('Times New Roman')
#     l.set_size(16)
#
# x, y = np.meshgrid(np.arange(0, Nx, 1), np.arange(-Ny+1, 0+1, 1))
# plt.subplot(1, 2, 2)
# delta = 1.2
# plt.xlim(0-delta, Nx-1 + delta)
# plt.ylim(0-delta, Ny-1 + delta)
# plt.xticks(np.arange(-1, 12, 1.))
# plt.yticks(np.arange(-1, 12, 1.))
# # ax = plt.gca()
# # ax.spines['top'].set_visible(False)
# # ax.spines['right'].set_visible(False)
# # ax.spines['left'].set_visible(False)
# # ax.spines['bottom'].set_visible(False)
# plt.quiver(x, -y, weight_sum_x_reshape, weight_sum_y_reshape, weight_sum_reshape,
#              zorder=3)
# plt.grid(True, zorder=2)
#
# plt.show()