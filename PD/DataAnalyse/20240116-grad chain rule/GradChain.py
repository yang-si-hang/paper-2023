

import numpy as np
import matplotlib.pyplot as plt

xx, yy = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
A = np.array([[1,1],[1,1]])

# r = np.zeros((100,100))
f = np.zeros((100,100))
for i in range(100):
    for j in range(100):
        v = np.array([xx[i,j], yy[i,j]])
        r_tmp = A @ v
        # r[i,j] = r_tmp
        f[i, j] = np.linalg.norm(r_tmp)**2

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制表面
surf = ax.plot_surface(xx, yy, f, cmap='viridis')

# 添加颜色条
fig.colorbar(surf)

# 显示图形
plt.show()