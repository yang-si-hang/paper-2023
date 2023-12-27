"""
Validation of the gradient of the curvature of three feature pos in 2D
Gradient is wrong, need check!
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.cpu, debug=True)
import taichi.math as tm



feature_pos = ti.Vector.field(2, dtype=ti.f64, shape=3)
dL = ti.Vector.field(2, dtype=ti.f64, shape=3)
dkappa1 = ti.Vector.field(2, dtype=ti.f64, shape=3)
dkappa2 = ti.Vector.field(2, dtype=ti.f64, shape=3)
dkappa3 = ti.Vector.field(2, dtype=ti.f64, shape=3)
U = ti.Matrix([[0., 1.],
               [-1., 0.]])
feature_pos_np = np.array([[1, 0], [np.sqrt(2.), np.sqrt(2, )], [0, 1]])
feature_pos.from_numpy(feature_pos_np)


@ti.kernel
def kappa_grad():
    feature_1, feature_2, feature_3 = feature_pos[0], feature_pos[1], feature_pos[2]

    # f are intermediate variables
    f1 = feature_1 - feature_2
    f2 = feature_2 - feature_3

    kappa_1 = 1. / f1.norm()
    kappa_2 = 1. / f2.norm()
    kappa_3 = tm.cross(f1, f2)

    kappa_sum = kappa_1 * kappa_2 * kappa_3
    desired_curvature = 0.
    current_curvature = kappa_1 * kappa_2 * kappa_3
    L = (current_curvature - desired_curvature) ** 2
    print('Loss:', L)

    dL_0 = 2*(current_curvature - desired_curvature)

    dkappa1[0] = -kappa_1**3*f1
    dkappa1[1] = kappa_1**3*f1
    dkappa2[1] = -kappa_2**3*f2
    dkappa2[2] = kappa_2**3*f2
    dkappa3[0] = f2@U.transpose()
    dkappa3[1] = f1@U - f2@U.transpose()
    dkappa3[2] = -f1@U

    dL_1 = dL_0 * (dkappa1[0] * kappa_2 * kappa_3 + kappa_1 * dkappa2[0] * kappa_3 +
                   kappa_1 * kappa_2 * dkappa3[0])
    dL_2 = dL_0 * (dkappa1[1] * kappa_2 * kappa_3 + kappa_1 * dkappa2[1] * kappa_3 +
                   kappa_1 * kappa_2 * dkappa3[1])
    dL_3 = dL_0 * (dkappa1[2] * kappa_2 * kappa_3 + kappa_1 * dkappa2[2] * kappa_3 +
                   kappa_1 * kappa_2 * dkappa3[2])

    print('dL_1 = ', dL_1)
    print('dL_2 = ', dL_2)
    print('dL_3 = ', dL_3)


if __name__ == '__main__':
    kappa_grad()
    print('dL = ', dL.to_numpy())


"""
# Loss gradient check with numpy form
U = np.array([[0., 1.],
              [-1, 0.]])
feature_pos_np = np.array([[1, 0], [np.sqrt(2.), np.sqrt(2.)], [0, 1]])
feature_1, feature_2, feature_3 = feature_pos_np

# g are intermediate variables
f1 = feature_1 - feature_2
f2 = feature_2 - feature_3


kappa_1 = 1. / np.linalg.norm(f1)
kappa_2 = 1. / np.linalg.norm(f2)
kappa_3 = np.cross(f1, f2)

kappa_sum = kappa_1 * kappa_2 * kappa_3
desired_curvature = 0.
current_curvature = kappa_1 * kappa_2 * kappa_3
L = (current_curvature - desired_curvature) ** 2

dL_0 = 2*(current_curvature - desired_curvature)
dkappas = [np.zeros((3, 2)) for _ in range(3)]
dkappa_1, dkappa_2, dkappa_3 = dkappas

dkappa_1[0, :] = -kappa_1**3*f1.transpose()
dkappa_1[1, :] = kappa_1**3*f1.transpose()
dkappa_2[1, :] = -kappa_2**3*f2.transpose()
dkappa_2[2, :] = kappa_2**3*f2.transpose()
dkappa_3[0, :] = f2.transpose()@U.transpose()
dkappa_3[1, :] = f1.transpose()@U - f2.transpose()@U.transpose()
dkappa_3[2, :] = -f1.transpose()@U

dL_1 = dL_0 * (dkappa_1[0,:]*kappa_2*kappa_3 + kappa_1*dkappa_2[0,:]*kappa_3 +
               kappa_1*kappa_2*dkappa_3[0,:])
dL_2 = dL_0 * (dkappa_1[1,:]*kappa_2*kappa_3 + kappa_1*dkappa_2[1,:]*kappa_3 +
               kappa_1*kappa_2*dkappa_3[1,:])
dL_3 = dL_0 * (dkappa_1[2,:]*kappa_2*kappa_3 + kappa_1*dkappa_2[2,:]*kappa_3 +
               kappa_1*kappa_2*dkappa_3[2,:])

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
    dkappa_sum = (kappa_sum_tmp - kappa_sum) / delta_x
    print(f'dL_tmp[{i},{j}] = ', dL_tmp)
    # print(f'dkappa[{i},{j}] = ', dkappa_sum)
"""


"""
# Subcase test for check the gradient of kappa
f1 = feature_1 - feature_2
f2 = feature_2 - feature_3

kappa_1 = 1. / np.linalg.norm(f1)
kappa_2 = 1. / np.linalg.norm(f2)
kappa_3 = np.cross(f1, f2)          # not need "norm" due to 2D vector

dkappas = [np.zeros((3, 2)) for _ in range(3)]
dkappa_1, dkappa_2, dkappa_3 = dkappas

dkappa_1[0, :] = -kappa_1**3*f1.transpose()
dkappa_1[1, :] = kappa_1**3*f1.transpose()
dkappa_2[1, :] = -kappa_2**3*f2.transpose()
dkappa_2[2, :] = kappa_2**3*f2.transpose()
dkappa_3[0, :] = f2.transpose()@U.transpose()
dkappa_3[1, :] = f1.transpose()@U - f2.transpose()@U.transpose()
dkappa_3[2, :] = -f1.transpose()@U
print('dkappa_1:', dkappa_1)
print('dkappa_2:', dkappa_2)
print('dkappa_3:', dkappa_3)

delta_x = 1.e-8
for i ,j in np.ndindex((3, 2)):
    feature_pos_tmp_np = feature_pos_np.copy()
    feature_pos_tmp_np[i, j] += delta_x

    f1_tmp = feature_pos_tmp_np[0] - feature_pos_tmp_np[1]
    f2_tmp = feature_pos_tmp_np[1] - feature_pos_tmp_np[2]
    kappa1_tmp = 1. / np.linalg.norm(f1_tmp)
    kappa2_tmp = 1. / np.linalg.norm(f2_tmp)
    kappa3_tmp = np.cross(f1_tmp, f2_tmp)

    print(f'The gradient of {i}, {j}')
    print('dkappa_1_tmp:', (kappa1_tmp-kappa_1)/delta_x)
    print('dkappa_2_tmp:', (kappa2_tmp-kappa_2)/delta_x)
    print('dkappa_3_tmp:', (kappa3_tmp-kappa_3)/delta_x)
"""