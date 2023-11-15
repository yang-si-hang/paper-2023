import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# # Case one ---------------------------------------------------------------------------------------
# # 创建一个 10x10 的 numpy 数组
# raw_data = np.loadtxt('partial_displacement.csv')
# partial_x = raw_data[::2]  # 从索引0开始，每两个取一个
# partial_y = raw_data[1::2] # 从索引1开始，每两个取一个
# data_x = partial_x.reshape((10, 10))
# data_y = partial_y.reshape((10, 10))
#
# """
# # 3D 柱状图
# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
#
# xpos, ypos = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")
#
# xpos = xpos.flatten()
# ypos = ypos.flatten()
# zpos = np.zeros_like(xpos)
#
# dx = dy = 0.75
# dz = data.flatten()
#
# # 根据Z轴的高度为柱子分配颜色
# norm = plt.Normalize(dz.min(), dz.max())
# colors = plt.cm.viridis(norm(dz))
#
# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors)
#
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_zlabel('Z Axis')
# ax.set_title('3D Bar Chart')
# ax.view_init(90, -90)  # 90度竖直方向，-90度水平方向（从正x轴开始旋转）
#
# # 添加颜色条
# mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
# mappable.set_array(dz)
# cbar = plt.colorbar(mappable, ax=ax)
# cbar.set_label('Z Height')
# """
#
# data = data_x
#
# # 2D 热力图
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 使用 imshow 显示热力图
# cax = ax.imshow(data, cmap='viridis', interpolation='nearest')
#
# # 添加颜色条
# cbar = fig.colorbar(cax, ax=ax)
# cbar.set_label('Z Value')
#
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_title('2D Heatmap Representing Z Values')
#
# plt.show()


# Case two -----------------------------------------------------------------------------------------
# This part can plot the dx/dy matrix with 2-dim heat map.
# np.set_printoptions(precision=7, suppress=True, linewidth=200)
# dx_dy = np.loadtxt('dx_dy.csv', delimiter=',')
# print(dx_dy)
# print('row sum = ', np.sum(dx_dy, axis=1))
# print('row abs sum = ', np.sum(np.abs(dx_dy), axis=1))
#
# data = dx_dy
# # 2D 热力图
# fig, ax = plt.subplots(figsize=(8, 6))
#
# # 使用 imshow 显示热力图
# cax = ax.imshow(data, cmap='viridis', interpolation='nearest')
#
# # 添加颜色条
# cbar = fig.colorbar(cax, ax=ax)
# cbar.set_label('Z Value')
#
# ax.set_xlabel('X Axis')
# ax.set_ylabel('Y Axis')
# ax.set_title('2D Heatmap Representing Z Values')
#
# plt.show()


# Case three ---------------------------------------------------------------------------------------
# This part analyse the manipulation of each node
np.set_printoptions(precision=7, suppress=True, linewidth=200)
dx_dy = np.loadtxt('dx_dy.csv', delimiter=',')
row_sum = np.sum(dx_dy, axis=0)
print('row sum = ', row_sum)
row_asc = np.argsort(row_sum)
row_desc = row_asc[::-1]
print('descending row index = ', row_desc)

data = row_sum
arr_x = data[::2]
arr_y = data[1::2]

# 重塑为 10x10 的二维数组
Nx, Ny = 11, 11
arr_x_reshaped = arr_x.reshape(Nx,Ny)
arr_y_reshaped = arr_y.reshape(Nx, Ny)

# 相加两个矩阵
arr_sum = arr_x_reshaped + arr_y_reshaped

# 应用 3x3 的均值滤波
filtered_arr = gaussian_filter(arr_sum, sigma=0.5, mode='nearest')

# 绘制热图
plt.figure(figsize=(12, 12))

# 绘制偶数索引元素的热图
plt.subplot(2, 2, 1)
plt.imshow(arr_x_reshaped, cmap='plasma', interpolation='nearest')
plt.title('Heatmap of X Dierction')
plt.colorbar()

# 绘制奇数索引元素的热图
plt.subplot(2, 2, 2)
plt.imshow(arr_y_reshaped, cmap='plasma', interpolation='nearest')
plt.title('Heatmap of Y Direction')
plt.colorbar()

# 使用默认颜色映射绘制相加后的矩阵的热图
plt.subplot(2, 2, 3)
plt.imshow(arr_sum, cmap='plasma', interpolation='nearest')
plt.title('Heatmap of Sum of X and Y Direction')
plt.colorbar()

# 绘制滤波后的矩阵的热图
plt.subplot(2, 2, 4)
plt.imshow(filtered_arr, cmap='plasma', interpolation='nearest')
plt.title('Heatmap of 3x3 Mean Filtered Matrix')
plt.colorbar()

plt.show()
