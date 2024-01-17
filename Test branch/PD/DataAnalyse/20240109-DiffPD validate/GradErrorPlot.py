"""
This file is used to plot the error of the gradient of the soft object deformation by the DiffPD
method which compared to the finit difference method.
"""


from cProfile import label
from tkinter import font
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties


# 设置全局字体为 Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = 'Times New Roman'

dx_dy_grasp = np.loadtxt('dx_dy_grasp.csv', delimiter=',')
grad_finite_grasp = np.loadtxt('grad_finite.csv', delimiter=',')          # the ground truth

error = np.abs(grad_finite_grasp - dx_dy_grasp)

Nx, Ny = 11, 11

error_x = error[:,0]
error_y = error[:,1]

error_x_rel = error_x / np.abs(grad_finite_grasp[:,0])
error_y_rel = error_y / np.abs(grad_finite_grasp[:,1])

error_x_reshape = error_x.reshape(Nx, Ny)
error_y_reshape = error_y.reshape(Nx, Ny)

error_x_rel_reshape = error_x_rel.reshape(Nx, Ny)
error_y_rel_reshape = error_y_rel.reshape(Nx, Ny)


# Plot the error of the gradient in x and y direction
fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
fig.patch.set_alpha(0.)
im1 = ax[0].imshow(error_x_reshape, cmap='jet')
ax[0].set_title('Error of the Gradient in X Direction', fontsize=18)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].grid(False)

cbar1 = fig.colorbar(im1, ax=ax[0])
cbar1.formatter.set_powerlimits((0,0))
cbar1.ax.tick_params(labelsize=15)           # Tickers' font size
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)            # Offset's font size
# cbar.update_ticks()


im2 = ax[1].imshow(error_y_reshape, cmap='jet')
ax[1].set_title('Error of the Gradient in Y Direction', fontsize=18)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].grid(False)

cbar2 = fig.colorbar(im2, ax=ax[1])
cbar2.formatter.set_powerlimits((0,0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

# plt.show()


# Plot the relative error of the gradient in x and y direction
fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
fig.patch.set_alpha(0.)
im1 = ax[0].imshow(error_x_rel_reshape, cmap='jet')
ax[0].set_title('Relative Error of the Gradient in X Direction', fontsize=18)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].grid(False)

cbar1 = fig.colorbar(im1, ax=ax[0])
cbar1.formatter.set_powerlimits((0,0))
cbar1.ax.tick_params(labelsize=15)
cbar1.ax.yaxis.get_offset_text().set_fontsize(15)


im2 = ax[1].imshow(error_y_rel_reshape, cmap='jet')
ax[1].set_title('Relative Error of the Gradient in Y Direction', fontsize=18)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].grid(False)

cbar2 = fig.colorbar(im2, ax=ax[1])
cbar2.formatter.set_powerlimits((0,0))
cbar2.ax.tick_params(labelsize=15)
cbar2.ax.yaxis.get_offset_text().set_fontsize(15)

plt.show()