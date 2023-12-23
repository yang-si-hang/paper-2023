"""
Idx 1 represent that grasp bottom node
Idx 2 represent that grasp top node
Idx 3 & 4 for test trajectory correction
"""

import numpy as np
import matplotlib.pyplot as plt


loss = np.loadtxt('loss1.csv', delimiter=',')
grasp_pos = np.loadtxt('grasp_pos1.csv', delimiter=',')
marker_pos = np.loadtxt('marker_pos1.csv', delimiter=',')

fig, axs = plt.subplots(3, 1, figsize=(10, 12), dpi=100)
plt.rcParams['font.family'] = 'Times New Roman'  # 设置刻度标签的字体

# Plot 1: Loss
axs[0].plot(loss, color='black', linewidth=2)
axs[0].set_title('Loss Over Time', fontsize=20, fontname='Times New Roman')
axs[0].set_xlabel('Time Step', fontsize=16, fontname='Times New Roman')
axs[0].set_ylabel('Loss', fontsize=16, fontname='Times New Roman')
axs[0].tick_params(axis='both', which='major', labelsize=12)  # 设置刻度标签大小
axs[0].grid(True)

# Plot 2: Grasp Position
axs[1].plot(grasp_pos[:, 0], -grasp_pos[:, 1], color='black', linewidth=2)
axs[1].set_title('Grasp Position Trajectory', fontsize=20, fontname='Times New Roman')
axs[1].set_xlabel('X Position', fontsize=16, fontname='Times New Roman')
axs[1].set_ylabel('Y Position', fontsize=16, fontname='Times New Roman')
axs[1].tick_params(axis='both', which='major', labelsize=12)  # 设置刻度标签大小
axs[1].axis('equal')  # Setting equal aspect ratio
axs[1].grid(True)

# Plot 3: Marker Position
axs[2].plot(marker_pos[:, 0], -marker_pos[:, 1], color='black', linewidth=2)
axs[2].plot(0.09, 0.02, marker='o', markersize=10, color="red")
axs[2].plot(0.09+0.002, 0.02, marker='o', markersize=10, color="blue")
axs[2].set_title('Marker Position Trajectory', fontsize=20, fontname='Times New Roman')
axs[2].set_xlabel('X Position', fontsize=16, fontname='Times New Roman')
axs[2].set_ylabel('Y Position', fontsize=16, fontname='Times New Roman')
axs[2].tick_params(axis='both', which='major', labelsize=12)  # 设置刻度标签大小
axs[2].axis('equal')  # Setting equal aspect ratio
axs[2].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()