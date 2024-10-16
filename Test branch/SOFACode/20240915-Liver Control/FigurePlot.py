"""
用来绘制3D轨迹图
created at 2024-09-16 by hsy
"""


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.gridspec import GridSpec
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

folder_path = os.path.dirname(os.path.abspath(__file__))

POINTS_NUM:int = 1
num:int = 2

loss_np = np.loadtxt(f'{folder_path}/loss_list{num}.csv', delimiter=',')
dot_sofa_np = np.loadtxt(f'{folder_path}/dots_sofa_list{num}.csv', delimiter=',')
rob_mov_np = np.loadtxt(f'{folder_path}/rob_movement_list{num}.csv', delimiter=',')
delta_pos = np.loadtxt(f'{folder_path}/delta_pos_list{num}.csv', delimiter=',')
delta_pos_model = np.loadtxt(f'{folder_path}/delta_pos_model_list{num}.csv', delimiter=',')
sofa_contact_pos = np.loadtxt(f'{folder_path}/sofa_contact_pos_list{num}.csv', delimiter=',')

inital_pos = dot_sofa_np[0,:]
desired_pos = inital_pos + np.array([0.02, 0.01, 0.01])

t = np.linspace(0, 1, len(sofa_contact_pos[:,0]))
norm = plt.Normalize(t.min(), t.max())
colors = cm.viridis(norm(t))

dot_pos_xyz = np.stack([dot_sofa_np[:,0], dot_sofa_np[:,1], dot_sofa_np[:,2]], axis=1).reshape(-1, 1, 3)
segments = np.concatenate([dot_pos_xyz[:-1], dot_pos_xyz[1:]], axis=1)
lc = Line3DCollection(segments, color=colors, linewidth=1.5, zorder=2)

fig_marker = plt.figure(figsize=(4, 3.5), dpi=150)
gs = GridSpec(1, 1, figure=fig_marker)

ax_marker = fig_marker.add_subplot(gs[0, 0], projection='3d')
ax_marker.add_collection(lc, autolim=False)
# ax_marker.plot(dot_sofa_np[:,0], dot_sofa_np[:,1], dot_sofa_np[:,2], label='3D Trajectory')

ax_marker.set_xlim([np.min(dot_sofa_np[:,0]), np.max(dot_sofa_np[:,0])])
ax_marker.set_ylim([np.min(dot_sofa_np[:,1]), np.max(dot_sofa_np[:,1])])
ax_marker.set_zlim([np.min(dot_sofa_np[:,2]), np.max(dot_sofa_np[:,2])])
ax_marker.scatter(inital_pos[0], inital_pos[1], inital_pos[2], marker='^', s=35, c='b', label='Initial Position', zorder=10)
ax_marker.scatter(desired_pos[0], desired_pos[1], desired_pos[2], marker='*', s=35, c='red', label='Desired Position')
# ax_marker.set_title('Marker Point Trajectory', fontsize=16, fontname='Times New Roman')

# ax_marker.set_xlabel('X', fontsize=16, fontname='Times New Roman')
# ax_marker.set_ylabel('Y', fontsize=16, fontname='Times New Roman')
# ax_marker.set_zlabel('Z', fontsize=16, fontname='Times New Roman')
ax_marker.xaxis.set_major_locator(MultipleLocator(0.005))
ax_marker.yaxis.set_major_locator(MultipleLocator(0.005))
ax_marker.zaxis.set_major_locator(MultipleLocator(0.005))
ax_marker.set_aspect('equal', 'datalim')
ax_marker.tick_params(axis='both', which='major', labelsize=12)
ax_marker.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, 0.9), ncol=1)

fig_marker.tight_layout()
fig_marker.savefig(f'{folder_path}/3D_Marker_Trajectory.png', transparent=True, dpi=300, bbox_inches='tight')

indices = np.arange(len(sofa_contact_pos[:,0]))

rob_pose_xyz = np.stack([sofa_contact_pos[:,0][indices], sofa_contact_pos[:,1][indices], sofa_contact_pos[:,2][indices]], axis=1).reshape(-1, 1, 3)
segments = np.concatenate([rob_pose_xyz[:-1], rob_pose_xyz[1:]], axis=1)
lc = Line3DCollection(segments, colors=colors, linewidth=2)

fig_rob = plt.figure(figsize=(4, 3.5), dpi=150)
gs = GridSpec(1, 1, figure=fig_rob)
ax_rob = fig_rob.add_subplot(gs[0, 0], projection='3d')
ax_rob.add_collection(lc, autolim=True)

ax_rob.set_xlim([np.min(sofa_contact_pos[:,0]), np.max(sofa_contact_pos[:,0])])
ax_rob.set_ylim([np.min(sofa_contact_pos[:,1]), np.max(sofa_contact_pos[:,1])])
ax_rob.set_zlim([np.min(sofa_contact_pos[:,2]), np.max(sofa_contact_pos[:,2])])

ax_rob.set_xlabel('X', fontsize=16, fontname='Times New Roman')
ax_rob.set_ylabel('Y', fontsize=16, fontname='Times New Roman')
ax_rob.set_zlabel('Z', fontsize=16, fontname='Times New Roman')

# ax_rob.set_title('Robot Tool Trajectory', fontsize=16, fontname='Times New Roman')
ax_rob.set_aspect('equal', 'datalim')
ax_rob.tick_params(axis='both', which='major', labelsize=12)
fig_rob.tight_layout()
fig_rob.savefig(f'{folder_path}/3D_Robot_Trajectory.png')

plt.show()
# plt.savefig(f'{folder_path}/3D_Trajectory{num}.png')
# fig.savefig(f'{folder_path}/trajectory.png', transparent=True, dpi=300)