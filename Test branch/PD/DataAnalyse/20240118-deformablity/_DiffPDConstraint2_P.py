"""
This file for plot the result (\partial x/ \partial y) of the _DiffPDDeformability.py
Two figures: one is leftside positional; another is downside positional. Variable
<fix_flag_temp> need to be changed to plot the corresponding figure.
"""

from platform import node
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
import os
script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

Nx, Ny = 11, 11

# Leftside positional
dx_dy_1 = np.loadtxt('dx_dy1.csv', delimiter=',')
node_dis_1 = np.diag(dx_dy_1)

# Leftside & downside positional
dx_dy_2 = np.loadtxt('dx_dy2.csv', delimiter=',')
node_dis_2 = np.diag(dx_dy_2)


node_dis_x_1 = node_dis_1[::2]
node_dis_y_1 = node_dis_1[1::2]

node_dis_x_reshape_1 = node_dis_x_1.reshape(Nx, Ny)
node_dis_y_reshape_1 = node_dis_y_1.reshape(Nx, Ny)
node_dis_norm_1 = np.sqrt(node_dis_x_reshape_1**2 + node_dis_y_reshape_1**2)

node_dis_x_2 = node_dis_2[::2]
node_dis_y_2 = node_dis_2[1::2]

node_dis_x_reshape_2 = node_dis_x_2.reshape(Nx, Ny)
node_dis_y_reshape_2 = node_dis_y_2.reshape(Nx, Ny)
node_dis_norm_2 = np.sqrt(node_dis_x_reshape_2**2 + node_dis_y_reshape_2**2)


# 2D 热力图
fig, axs = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
fig.patch.set_alpha(0.)

im1 = axs[0].imshow(node_dis_norm_1, cmap='jet', interpolation='nearest')
axs[0].set_title('Node Deformation Ability', fontsize=18)
axs[0].axis('off')
axs[0].grid(False)

min_val, max_val = node_dis_norm_1.min(), node_dis_norm_1.max()
mid_val = (min_val + max_val) / 2
cbar1 = fig.colorbar(im1, ax=axs[0])
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)
cbar1.set_ticks([min_val, mid_val, max_val])
cbar1.set_ticklabels(['Low', 'Medium', 'High'])


im2 = axs[1].imshow(node_dis_norm_2, cmap='jet', interpolation='nearest')
axs[1].set_title('Node Deformation Ability', fontsize=18)
axs[1].axis('off')
axs[1].grid(True)

min_val, max_val = node_dis_norm_2.min(), node_dis_norm_2.max()
mid_val = (min_val + max_val) / 2
cbar2 = fig.colorbar(im2, ax=axs[1])
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)
cbar2.set_ticks([min_val, mid_val, max_val])
cbar2.set_ticklabels(['Low', 'Medium', 'High'])

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