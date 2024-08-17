"""使用numpy实现的一些数学函数
created at 2024-08-17 by hsy
"""


import numpy as np


def quatfromtwovectors_np(a, b):
    # a -> b的旋转四元数
    v1 = a / np.linalg.norm(a)
    v2 = b / np.linalg.norm(b)
    cos_theta = v1.dot(v2)

    quat = np.zeros(4)
    if cos_theta < -1 + 1e-6:
        cos_theta = max(cos_theta, -1)
        m = np.vstack((v1, v2))
        u, s, vh = np.linalg.svd(m, np.float64)             # 奇异值分解得到垂直的特征向量v3
        axis_tmp = vh[2, :]
        w2 = (1 + cos_theta) * 0.5              # w2=cos^2(theta/2)
        w = np.sqrt(w2)
        vec = axis_tmp * np.sqrt(1 - w2)
        quat[0] = w
        quat[1:] = vec
    else:
        axis_tmp = np.cross(v1, v2)             # 旋转轴*sin(theta)
        s = np.sqrt((1 + cos_theta) * 2)        # s=2*cos(theta/2)
        invs = 1 / s
        vec = axis_tmp * invs
        w = s * 0.5
        quat[0] = w
        quat[1:] = vec
    
    return quat


def quatconj_np(u):
    return np.array([u[0], -u[1], -u[2], -u[3]])


def quatmul_np(u1, u2):
    tmp1 = u1[0] * u2[0] - u1[1] * u2[1] - u1[2] * u2[2] - u1[3] * u2[3]
    tmp2 = u1[0] * u2[1] + u1[1] * u2[0] + u1[2] * u2[3] - u1[3] * u2[2]
    tmp3 = u1[0] * u2[2] - u1[1] * u2[3] + u1[2] * u2[0] + u1[3] * u2[1]
    tmp4 = u1[0] * u2[3] + u1[1] * u2[2] - u1[2] * u2[1] + u1[3] * u2[0]
    return np.array([tmp1, tmp2, tmp3, tmp4])


def quatrotvec_np(u, v):
    # 四元数u对向量v进行旋转
    q = np.array([0., v[0], v[1], v[2]])
    q_conj = quatconj_np(u)
    q_rot = quatmul_np(quatmul_np(u, q), q_conj)
    return np.array([q_rot[1], q_rot[2], q_rot[3]])


