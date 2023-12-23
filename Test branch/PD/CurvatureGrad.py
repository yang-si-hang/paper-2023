"""
Validation of the gradient of the curvature of three feature pos in 2D
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.cpu)
import taichi.math as tm


feature_pos = ti.Vector.field(2, dtype=ti.f64, shape=3)
feature_pos_np = np.array([[1, 0], [np.sqrt(2.), np.sqrt(2,)], [0, 1]])
feature_pos.from_numpy(feature_pos_np)
feature_1, feature_2, feature_3 = feature_pos

# g are intermediate variables
g1_tmp = feature_1 - feature_2
g2_tmp = feature_2 - feature_3
U = ti.Matrix([[0., -1., 0.],
               [1., 0., 0.],
               [0., 0., 1.]])
kappa_1 = 1. / ti.norm(g1_tmp)
kappa_2 = 1. / ti.norm(g2_tmp)
kappa_3 = ti.norm(tm.cross(g1_tmp, g2_tmp))
kappa_sum = kappa_1 * kappa_2 * kappa_3
desired_curvature = 0.
current_curvature = kappa_1 * kappa_2 * kappa_3
L = (current_curvature - desired_curvature) ** 2

dL_1 = (kappa_sum*kappa_1**2*g1_tmp.transpose() + kappa_1*kappa_2*g2_tmp.transpose()@U.transpose())
dL_2 = ((kappa_sum*kappa_2**2*g2_tmp.transpose() + kappa_1*kappa_2*g1_tmp.transpose()@U) -
        (kappa_sum*kappa_1**2*g1_tmp.transpose() + kappa_1*kappa_2*g2_tmp.transpose()@U.transpose()))
dL_3 = -(kappa_sum*kappa_2**2*g2_tmp.transpose() + kappa_1*kappa_2*g1_tmp.transpose()@U)