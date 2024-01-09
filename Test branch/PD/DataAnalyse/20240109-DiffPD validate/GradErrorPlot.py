"""
This file is used to plot the error of the gradient of the soft object deformation by the DiffPD
method.
"""


import numpy as np
import matplotlib.pyplot as plt


dx_dy_grasp = np.loadtxt('dx_dy_grasp.csv', delimiter=',')
grad_finite_grasp = np.loadtxt('grad_finite_grasp.csv', delimiter=',')          # the ground truth

error = np.abs(grad_finite_grasp - dx_dy_grasp)

Nx, Ny = 11, 11

error_x = error[::2]
error_y = error[1::2]

error_x_reshape = error_x.reshape(Nx, Ny)
error_y_reshape = error_y.reshape(Nx, Ny)

# 2D 热力图
fig, ax = plt.subplots(figsize=(8, 6))
plt.axis('off')
plt.grid(True)

plt.subplot(1, 2, 1)
plt.imshow(error_x_reshape, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(error_y_reshape, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar()

plt.show()