"""
This file for plot the result (\partial x/ \partial y) of the _DiffPDConstraint2.py
Two figures: one is leftside positional; another is downside positional. Variable
<fix_flag_temp> need to be changed to plot the corresponding figure.
"""

import numpy as np
import matplotlib.pyplot as plt


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