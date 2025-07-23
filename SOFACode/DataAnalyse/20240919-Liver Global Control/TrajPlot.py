"""

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

# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

folder_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(folder_path)
print("当前工作目录:", os.getcwd())

def plot_traj(contact_traj, plot_step_num):
    fig = plt.figure(figsize=(7, 6), dpi=100)
    gs = GridSpec(1, 1, figure=fig)
    # plot_step_num = 100

    # 选择点的索引
    indices = np.arange(len(contact_traj[:plot_step_num,0]))

    # 计算颜色渐变
    t = np.linspace(0, 1, len(contact_traj[:plot_step_num,0]))  # 颜色映射的变量，假设是时间
    norm = plt.Normalize(t.min(), t.max())
    colors = cm.viridis(norm(t))  # 使用 viridis 颜色映射
    colors = plt.cm.plasma(norm(t))

    # 创建 3D 坐标点
    rob_pose_xyz = np.stack([contact_traj[:plot_step_num,0][indices], contact_traj[:plot_step_num,1][indices], contact_traj[:plot_step_num,2][indices]], axis=1).reshape(-1, 1, 3)

    # 创建线段（每一段是连续的两个点之间的线段）
    segments = np.concatenate([rob_pose_xyz[:-1], rob_pose_xyz[1:]], axis=1)

    # 使用 Line3DCollection 绘制具有渐变色的轨迹线
    lc = Line3DCollection(segments, colors=colors, linewidth=2)

    ax_traj = fig.add_subplot(gs[0, 0], projection='3d')
    ax_traj.add_collection(lc, autolim=True)

    ax_traj.scatter(contact_traj[0,0], contact_traj[0,1], contact_traj[0,2], c='blue', s=25, label='Initial Position')
    ax_traj.scatter(contact_traj[0,0]+0.025, contact_traj[0,1], contact_traj[0,2], c='red', marker='*', s=25, label='Desired Position')

    ax_traj.set_xlim([np.min(contact_traj[:plot_step_num,0]), np.max(contact_traj[:plot_step_num,0])])
    ax_traj.set_ylim([np.min(contact_traj[:plot_step_num,1]), np.max(contact_traj[:plot_step_num,1])])
    ax_traj.set_zlim([np.min(contact_traj[:plot_step_num,2]), np.max(contact_traj[:plot_step_num,2])])

    ax_traj.set_xlabel('X', fontsize=14, fontname='Times New Roman')
    ax_traj.set_ylabel('Y', fontsize=14, fontname='Times New Roman')
    ax_traj.set_zlabel('Z', fontsize=14, fontname='Times New Roman')

    ax_traj.set_title('Robot Tool Trajectory', fontsize=16, fontname='Times New Roman')
    ax_traj.set_aspect('equal', 'datalim')
    ax_traj.legend(fontsize=12, loc='upper right', bbox_to_anchor=(0.95, 0.8))


num:int = 3

loss_np = np.loadtxt(f'loss_list{num}.csv', delimiter=',')
dot_sofa_np = np.loadtxt(f'dots_sofa_list{num}.csv', delimiter=',')
rob_mov_np = np.loadtxt(f'rob_movement_list{num}.csv', delimiter=',')
sofa_contact_pos = np.loadtxt(f'sofa_contact_pos_list{num}.csv', delimiter=',')

plot_traj(sofa_contact_pos, 100)
# plt.show()

plt.savefig(f'{folder_path}/3D_Trajectory{num}.png', dpi=100, transparent=True)