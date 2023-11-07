import numpy as np
import matplotlib.pyplot as plt

# 创建一个 10x10 的 numpy 数组
raw_data = np.loadtxt('partial_displacement.csv')
partial_x = raw_data[::2]  # 从索引0开始，每两个取一个
partial_y = raw_data[1::2] # 从索引1开始，每两个取一个
data_x = partial_x.reshape((10, 10))
data_y = partial_y.reshape((10, 10))

""" 
# 3D 柱状图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

xpos, ypos = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing="ij")

xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)

dx = dy = 0.75
dz = data.flatten()

# 根据Z轴的高度为柱子分配颜色
norm = plt.Normalize(dz.min(), dz.max())
colors = plt.cm.viridis(norm(dz))

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True, color=colors)

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_zlabel('Z Axis')
ax.set_title('3D Bar Chart')
ax.view_init(90, -90)  # 90度竖直方向，-90度水平方向（从正x轴开始旋转）

# 添加颜色条
mappable = plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.viridis)
mappable.set_array(dz)
cbar = plt.colorbar(mappable, ax=ax)
cbar.set_label('Z Height')
"""

data = data_x

# 2D 热力图
fig, ax = plt.subplots(figsize=(8, 6))

# 使用 imshow 显示热力图
cax = ax.imshow(data, cmap='viridis', interpolation='nearest')

# 添加颜色条
cbar = fig.colorbar(cax, ax=ax)
cbar.set_label('Z Value')

ax.set_xlabel('X Axis')
ax.set_ylabel('Y Axis')
ax.set_title('2D Heatmap Representing Z Values')

plt.show()
