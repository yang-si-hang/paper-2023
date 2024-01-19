"""
This file for plot the result (\partial x/ \partial y) of the _DiffPDDeformability.py
Two figures: one is leftside positional; another is downside positional. Variable
<fix_flag_temp> need to be changed to plot the corresponding figure.
"""

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