import taichi as ti
import taichi.math as tm


@ti.func
def quatconj(u):
    # 实部在前,虚部在后
    return ti.Vector([u[0], -u[1], -u[2], -u[3]])


@ti.func
def quatconj2d(u):
    # 实部在前,虚部在后
    return ti.Vector([u[0], -u[1]])


@ti.func
def quatmul(u1, u2):
    tmp1 = u1[0] * u2[0] - u1[1] * u2[1] - u1[2] * u2[2] - u1[3] * u2[3]
    tmp2 = u1[0] * u2[1] + u1[1] * u2[0] + u1[2] * u2[3] - u1[3] * u2[2]
    tmp3 = u1[0] * u2[2] - u1[1] * u2[3] + u1[2] * u2[0] + u1[3] * u2[1]
    tmp4 = u1[0] * u2[3] + u1[1] * u2[2] - u1[2] * u2[1] + u1[3] * u2[0]
    return ti.Vector([tmp1, tmp2, tmp3, tmp4])


@ti.func
def quatmul2d(u1, u2):
    return ti.Vector([u1[0]*u2[0]-u1[1]*u2[1], u1[0]*u2[1]+u1[1]*u2[0]])


@ti.func
def quatnormalize(u):
    return u.normalized()


@ti.func
def quatfromtwovectors(a, b):
    # a -> b的旋转四元数
    v1 = a.normalized()
    v2 = b.normalized()
    cos_theta = v1.dot(v2)

    quat = ti.Vector.zero(ti.f64, 4)
    if cos_theta < -1 + 1e-6:
        pass
        # cos_theta = max(cos_theta, -1)
        # m = ti.Matrix.rows([v1, v2])
        # u, s, v = ti.svd(m, ti.f64)             # 奇异值分解得到垂直的特征向量v3
        # axis = v[:, 2]
        # w2 = (1 + cos_theta) * 0.5              # w2=cos^2(theta/2)
        # w = np.sqrt(w2)
        # vec = axis * np.sqrt(1 - w2)
        # quat[0] = w
        # quat[1:] = vec
    else:
        axis = v1.cross(v2)                     # 旋转轴*sin(theta)
        s = ti.sqrt((1 + cos_theta) * 2)        # s=2*cos(theta/2)
        invs = 1 / s
        vec = axis * invs
        w = s * 0.5
        quat[0] = w
        quat[1:] = vec
    
    return quat


@ti.func
def quatrotvec(u, v):
    # 四元数u对向量v进行旋转
    q = ti.Vector([0., v[0], v[1], v[2]])
    q_conj = quatconj(u)
    q_rot = quatmul(quatmul(u, q), q_conj)
    return ti.Vector([q_rot[1], q_rot[2], q_rot[3]])