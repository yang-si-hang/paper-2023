"""
Validation of the gradient of the curvature of three feature pos in 2D
Gradient is wrong, need check!
"""

import numpy as np
# import taichi as ti
# ti.init(arch=ti.cpu)
# import taichi.math as tm


"""
feature_pos = ti.Vector.field(2, dtype=ti.f64, shape=3)
dL = ti.Vector.field(2, dtype=ti.f64, shape=3)
U = ti.Matrix([[0., -1., 0.],
               [1., 0., 0.],
               [0., 0., 1.]])
feature_pos_np = np.array([[1, 0], [np.sqrt(2.), np.sqrt(2, )], [0, 1]])
feature_pos.from_numpy(feature_pos_np)


@ti.kernel
def kappa_grad():
    feature_1, feature_2, feature_3 = feature_pos[0], feature_pos[1], feature_pos[2]

    # g are intermediate variables
    g1_tmp = feature_1 - feature_2
    g2_tmp = feature_2 - feature_3

    kappa_1 = 1. / g1_tmp.norm()
    kappa_2 = 1. / g2_tmp.norm()
    kappa_3 = tm.cross(g1_tmp, g2_tmp).norm()

    kappa_sum = kappa_1 * kappa_2 * kappa_3
    desired_curvature = 0.
    current_curvature = kappa_1 * kappa_2 * kappa_3
    L = (current_curvature - desired_curvature) ** 2

    dL_1 = (kappa_sum*kappa_1**2*g1_tmp.transpose() + kappa_1*kappa_2*g2_tmp.transpose()@U.transpose())
    dL_2 = ((kappa_sum*kappa_2**2*g2_tmp.transpose() + kappa_1*kappa_2*g1_tmp.transpose()@U) -
            (kappa_sum*kappa_1**2*g1_tmp.transpose() + kappa_1*kappa_2*g2_tmp.transpose()@U.transpose()))
    dL_3 = -(kappa_sum*kappa_2**2*g2_tmp.transpose() + kappa_1*kappa_2*g1_tmp.transpose()@U)

    dL[0] = dL_1
    dL[1] = dL_2
    dL[2] = dL_3

    print('dL = ', dL.to_numpy())


if __name__ == '__main__':
    kappa_grad()
    # print('dL = ', dL.to_numpy())
"""


U = np.array([[0., -1.],
              [-1, 0.]])
feature_pos_np = np.array([[1, 0], [np.sqrt(2.), np.sqrt(2.)], [0, 1]])
feature_1, feature_2, feature_3 = feature_pos_np

# g are intermediate variables
g1_tmp = feature_1 - feature_2
g2_tmp = feature_2 - feature_3

kappa_1 = 1. / np.linalg.norm(g1_tmp)
kappa_2 = 1. / np.linalg.norm(g2_tmp)
kappa_3 = np.linalg.norm(np.cross(g1_tmp, g2_tmp))

kappa_sum = kappa_1 * kappa_2 * kappa_3
desired_curvature = 0.
current_curvature = kappa_1 * kappa_2 * kappa_3
L = (current_curvature - desired_curvature) ** 2

dL_1 = (kappa_sum*kappa_1**2*g1_tmp.transpose() + kappa_1*kappa_2*g2_tmp.transpose()@U.transpose())
dL_2 = ((kappa_sum*kappa_2**2*g2_tmp.transpose() + kappa_1*kappa_2*g1_tmp.transpose()@U) -
        (kappa_sum*kappa_1**2*g1_tmp.transpose() + kappa_1*kappa_2*g2_tmp.transpose()@U.transpose()))
dL_3 = -(kappa_sum*kappa_2**2*g2_tmp.transpose() + kappa_1*kappa_2*g1_tmp.transpose()@U)

print('dL_1 = ', dL_1)
print('dL_2 = ', dL_2)
print('dL_3 = ', dL_3)

delta_x = 1.e-8
for i, j in np.ndindex((3, 2)):
    feature_pos_np_tmp = feature_pos_np.copy()
    feature_pos_np_tmp[i, j] += delta_x
    h1_tmp = feature_pos_np_tmp[0] - feature_pos_np_tmp[1]
    h2_tmp = feature_pos_np_tmp[1] - feature_pos_np_tmp[2]
    kappa_1_tmp = 1. / np.linalg.norm(h1_tmp)
    kappa_2_tmp = 1. / np.linalg.norm(h2_tmp)
    kappa_3_tmp = np.linalg.norm(np.cross(h1_tmp, h2_tmp))
    kappa_sum_tmp = kappa_1_tmp * kappa_2_tmp * kappa_3_tmp
    current_curvature_tmp = kappa_1_tmp * kappa_2_tmp * kappa_3_tmp
    L_tmp = (current_curvature_tmp - desired_curvature) ** 2
    dL_tmp = (L_tmp - L) / delta_x
    print(f'dL_tmp[{i},{j}] = ', dL_tmp)