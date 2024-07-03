import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

s1 = np.linspace(0.1, 2, 100)
s2 = np.linspace(0.1, 2, 100)
x, y = np.meshgrid(s1, s2)

ss1 = np.zeros((s1.shape[0], s2.shape[0]))
ss2 = np.zeros((s1.shape[0], s2.shape[0]))

for i, j in np.ndindex(s1.shape[0], s2.shape[0]):
    D, max_itr, tol = np.array([10., 10.]), 80, 1.e-6
    s1_tmp, s2_tmp = s1[i], s2[j]
    for itr in range(max_itr):
        aa, bb = D[0] + s1_tmp, D[1] + s2_tmp
        C = aa * bb - 1
        partial_C = np.array([bb, aa])

        D_temp = (partial_C.dot(D) - C) / np.linalg.norm(partial_C) ** 2 * partial_C
        D_error = np.linalg.norm(D - D_temp)
        D = D_temp
        if D_error < tol:
            break
    ss1[i, j], ss2[i, j] = D[0] + s1_tmp, D[1] + s2_tmp

fig = plt.figure(figsize=(6,6), dpi=100)
gs = GridSpec(2, 1, figure=fig)

ax_ss1 = fig.add_subplot(gs[0], projection='3d')
ax_ss1.plot_surface(x, y, ss1, cmap='viridis')

ax_ss2 = fig.add_subplot(gs[1], projection='3d')
ax_ss2.plot_surface(x, y, ss2, cmap='viridis')

axs = [ax_ss1, ax_ss2]
for ax in axs:
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.tight_layout()
plt.show()