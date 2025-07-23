import numpy as np
import matplotlib.pyplot as plt
from matplotlib import axes, gridspec, rcParams
import os
script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

n = 1
loss = np.loadtxt(f'loss{n}.csv', delimiter=',')
grasp_pos = np.loadtxt(f'grasp_pos{n}.csv', delimiter=',')
marker_pos = np.loadtxt(f'marker_pos{n}.csv', delimiter=',')

fig = plt.figure(figsize=(7, 7), dpi=100)
fig.patch.set_alpha(0.)

gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])  # 设置高度比例为1:1:1
axs = [None] * 3
axs[0] = plt.subplot(gs[0])
axs[1] = plt.subplot(gs[1])
axs[2] = plt.subplot(gs[2])

# Plot 1: Loss
axs[0].plot(loss, color='black', linewidth=2)
axs[0].set_title('Loss Over Time', fontsize=20)
axs[0].set_xlabel('Time Step', fontsize=16)
axs[0].set_ylabel('Loss', fontsize=16)
axs[0].tick_params(axis='both', which='major', labelsize=12)  # 设置刻度标签大小
axs[0].grid(True)

# Plot 2: Grasp Position
axs[1].plot(grasp_pos[:, 0], -grasp_pos[:, 1], color='black', linewidth=2)
axs[1].set_title('Grasp Position Trajectory', fontsize=20)
axs[1].set_xlabel('X Position', fontsize=16)
axs[1].set_ylabel('Y Position', fontsize=16)
axs[1].tick_params(axis='both', which='major', labelsize=12)
axs[1].axis('equal')  # Setting equal aspect ratio
axs[1].grid(True)

# Plot 3: Marker Position
axs[2].plot(marker_pos[:, 0], -marker_pos[:, 1], color='black', linewidth=2)
axs[2].plot(0.09, 0.02, marker='o', markersize=10, color="red")
axs[2].plot(0.09+0.002, 0.02, marker='o', markersize=10, color="blue")
axs[2].set_title('Marker Position Trajectory', fontsize=20)
axs[2].set_xlabel('X Position', fontsize=16)
axs[2].set_ylabel('Y Position', fontsize=16)
axs[2].tick_params(axis='both', which='major', labelsize=12)
axs[2].axis('equal')  # Setting equal aspect ratio
axs[2].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()