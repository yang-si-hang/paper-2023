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
grad_finite_grasp_x = grad_finite_grasp[::2]
grad_finite_grasp_y = grad_finite_grasp[1::2]

grad_finite_grasp_x_reshape = grad_finite_grasp_x.reshape(Nx, Ny)
grad_finite_grasp_y_reshape = grad_finite_grasp_y.reshape(Nx, Ny)

fig, ax = plt.subplots(figsize=(16, 6))
plt.axis('off')
plt.grid(True)

plt.subplot(1, 2, 1)
plt.imshow(np.abs(grad_finite_grasp_x_reshape), cmap='Greys', interpolation='nearest')
cbar = plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.abs(grad_finite_grasp_y_reshape), cmap='Greys', interpolation='nearest')
cbar = plt.colorbar()

plt.show()