"""
一些基于taichi的数学工具函数
created at 2024-12-12 by hsy
"""

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.lang import impl, ops


@ti.func
def dsytrd3(A, Q, dt):
    Q[0, 0] = 1.0
    Q[1, 1] = 1.0
    Q[2, 2] = 1.0
    e = ti.Vector([0.0, 0.0, 0.0], dt=dt)
    u = ti.Vector([0.0, 0.0, 0.0], dt=dt)
    q = ti.Vector([0.0, 0.0, 0.0], dt=dt)
    d = ti.Vector([0.0, 0.0, 0.0], dt=dt)
    h = A[0, 1] ** 2 + A[0, 2] ** 2
    g = 0.0
    if A[0, 1] > 0:
        g = -ops.sqrt(h)
    else:
        g = ops.sqrt(h)
    e[0] = g
    f = g * A[0, 1]
    u[1] = A[0, 1] - g
    u[2] = A[0, 2]
    omega = h - f
    if omega > 0.0:
        omega = 1.0 / omega
        K = 0.0
        f = A[1, 1] * u[1] + A[1, 2] * u[2]
        q[1] = omega * f  # p
        K += u[1] * f  # u* A u

        f = A[1, 2] * u[1] + A[2, 2] * u[2]
        q[2] = omega * f  # p
        K += u[2] * f  # u* A u

        K *= 0.5 * omega * omega

        q[1] = q[1] - K * u[1]
        q[2] = q[2] - K * u[2]

        d[0] = A[0, 0]
        d[1] = A[1, 1] - 2.0 * q[1] * u[1]
        d[2] = A[2, 2] - 2.0 * q[2] * u[2]

        for j in range(1, 3):
            f = omega * u[j]
            for i in range(1, 3):
                Q[i, j] = Q[i, j] - f * u[i]

        # Calculate updated A[1, 2] and store it in e[1]
        e[1] = A[1, 2] - q[1] * u[2] - u[1] * q[2]
    else:
        d[0] = A[0, 0]
        d[1] = A[1, 1]
        d[2] = A[2, 2]
        e[1] = A[1, 2]
    return d, e, Q


@ti.func
def dsyevq3(A, Q, w, dt):
    w, e, Q = dsytrd3(A, Q, dt)
    for l in range(0, 2):
        nIter = 0
        while True:
            # Check for convergence and exit iteration loop if off-diagonal
            # element e(l) is zero
            m = 0
            for i in range(l, 2):
                m = i
                g = ops.abs(w[m]) + ops.abs(w[m + 1])
                if ops.abs(e[m]) + g == g:
                    break
            if m == l:
                break

            nIter += 1
            assert nIter <= 30, "Timeout"

            # Calculate g = d_m - k
            g = (w[l + 1] - w[l]) / (e[l] + e[l])
            r = ops.sqrt(g * g + 1.0)
            if g > 0:
                g = w[m] - w[l] + e[l] / (g + r)
            else:
                g = w[m] - w[l] + e[l] / (g - r)

            s = c = 1.0
            p = 0.0
            i = m - 1
            while i >= l:
                f = s * e[i]
                b = c * e[i]
                if ops.abs(f) > ops.abs(g):
                    c = g / f
                    r = ops.sqrt(c * c + 1.0)
                    e[i + 1] = f * r
                    s = 1.0 / r
                    c *= s
                else:
                    s = f / g
                    r = ops.sqrt(s * s + 1.0)
                    e[i + 1] = g * r
                    c = 1.0 / r
                    s *= c

                g = w[i + 1] - p
                r = (w[i] - g) * s + 2.0 * c * b
                p = s * r
                w[i + 1] = g + p
                g = c * r - b

                for k in range(0, 3):
                    t = Q[k, i + 1]
                    Q[k, i + 1] = s * Q[k, i] + c * t
                    Q[k, i] = c * Q[k, i] - s * t

                i -= 1
            w[l] -= p
            e[l] = g
            e[m] = 0.0
    return Q, w


@ti.func
def sym_eig3x3(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 3x3 real symmetric matrix using Cardano's method.

    Mathematical concept refers to https://www.mpi-hd.mpg.de/personalhomes/globes/3x3/.

    Args:
        A (ti.Matrix(3, 3)): input 3x3 symmetric matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        eigenvalues (ti.Vector(3)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (ti.Matrix(3, 3)): The eigenvectors. Each column stores one eigenvector.
    """
    assert all(A == A.transpose()), "A needs to be symmetric"
    M_SQRT3 = 1.73205080756887729352744634151
    DBL_EPSILON = 2.2204460492503131e-16
    m = A.trace()
    dd = A[0, 1] * A[0, 1]
    ee = A[1, 2] * A[1, 2]
    ff = A[0, 2] * A[0, 2]
    c1 = A[0, 0] * A[1, 1] + A[0, 0] * A[2, 2] + A[1, 1] * A[2, 2] - (dd + ee + ff)
    c0 = A[2, 2] * dd + A[0, 0] * ee + A[1, 1] * ff - A[0, 0] * A[1, 1] * A[2, 2] - 2.0 * A[0, 2] * A[0, 1] * A[1, 2]

    p = m * m - 3.0 * c1
    q = m * (p - 1.5 * c1) - 13.5 * c0
    sqrt_p = ops.sqrt(ops.abs(p))
    phi = 27.0 * (0.25 * c1 * c1 * (p - c1) + c0 * (q + 6.75 * c0))
    phi = (1.0 / 3.0) * ops.atan2(ops.sqrt(ops.abs(phi)), q)

    c = sqrt_p * ops.cos(phi)
    s = (1.0 / M_SQRT3) * sqrt_p * ops.sin(phi)
    eigenvalues = ti.Vector([0.0, 0.0, 0.0], dt=dt)
    eigenvalues_final = ti.Vector([0.0, 0.0, 0.0], dt=dt)
    eigenvalues[1] = (1.0 / 3.0) * (m - c)
    eigenvalues[2] = eigenvalues[1] + s
    eigenvalues[0] = eigenvalues[1] + c
    eigenvalues[1] = eigenvalues[1] - s

    t = ops.abs(eigenvalues[0])
    u = ops.abs(eigenvalues[1])
    if u > t:
        t = u
    u = ops.abs(eigenvalues[2])
    if u > t:
        t = u
    if t < 1.0:
        u = t
    else:
        u = t * t
    error = 256.0 * DBL_EPSILON * u * u
    Q = ti.Matrix.zero(dt, 3, 3)
    Q_final = ti.Matrix.zero(dt, 3, 3)
    Q[0, 1] = A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]
    Q[1, 1] = A[0, 2] * A[0, 1] - A[1, 2] * A[0, 0]
    Q[2, 1] = A[0, 1] * A[0, 1]

    Q[0, 0] = Q[0, 1] + A[0, 2] * eigenvalues[0]
    Q[1, 0] = Q[1, 1] + A[1, 2] * eigenvalues[0]
    Q[2, 0] = (A[0, 0] - eigenvalues[0]) * (A[1, 1] - eigenvalues[0]) - Q[2, 1]
    norm = Q[0, 0] * Q[0, 0] + Q[1, 0] * Q[1, 0] + Q[2, 0] * Q[2, 0]
    early_ret = 0
    if norm <= error:
        Q_final, eigenvalues_final = dsyevq3(A, Q, eigenvalues, dt)
        early_ret = 1
    else:
        norm = ops.sqrt(1.0 / norm)
        Q[0, 0] *= norm
        Q[1, 0] *= norm
        Q[2, 0] *= norm

    if not early_ret:
        Q[0, 1] = Q[0, 1] + A[0, 2] * eigenvalues[1]
        Q[1, 1] = Q[1, 1] + A[1, 2] * eigenvalues[1]
        Q[2, 1] = (A[0, 0] - eigenvalues[1]) * (A[1, 1] - eigenvalues[1]) - Q[2, 1]
        norm = Q[0, 1] * Q[0, 1] + Q[1, 1] * Q[1, 1] + Q[2, 1] * Q[2, 1]
        if norm <= error:
            Q_final, eigenvalues_final = dsyevq3(A, Q, eigenvalues, dt)
            early_ret = 1
        else:
            norm = ops.sqrt(1.0 / norm)
            Q[0, 1] *= norm
            Q[1, 1] *= norm
            Q[2, 1] *= norm

        Q[0, 2] = Q[1, 0] * Q[2, 1] - Q[2, 0] * Q[1, 1]
        Q[1, 2] = Q[2, 0] * Q[0, 1] - Q[0, 0] * Q[2, 1]
        Q[2, 2] = Q[0, 0] * Q[1, 1] - Q[1, 0] * Q[0, 1]

    if early_ret:
        Q = Q_final
        eigenvalues = eigenvalues_final

    if eigenvalues[1] > eigenvalues[0]:
        tmp = eigenvalues[0]
        eigenvalues[0] = eigenvalues[1]
        eigenvalues[1] = tmp
        tmp2 = Q[:, 0]
        Q[:, 0] = Q[:, 1]
        Q[:, 1] = tmp2

    if eigenvalues[2] > eigenvalues[0]:
        tmp = eigenvalues[0]
        eigenvalues[0] = eigenvalues[2]
        eigenvalues[2] = tmp
        tmp2 = Q[:, 0]
        Q[:, 0] = Q[:, 2]
        Q[:, 2] = tmp2

    if eigenvalues[2] > eigenvalues[1]:
        tmp = eigenvalues[1]
        eigenvalues[1] = eigenvalues[2]
        eigenvalues[2] = tmp
        tmp2 = Q[:, 1]
        Q[:, 1] = Q[:, 2]
        Q[:, 2] = tmp2

    # Verify eigendecomposition
    Lambda = ti.Matrix.zero(dt, 3, 3)
    Lambda[0,0] = eigenvalues[0]
    Lambda[1,1] = eigenvalues[1] 
    Lambda[2,2] = eigenvalues[2]
    recon_A = Q @ Lambda @ Q.transpose()
    # assert ((A - recon_A).norm() < 1e-8), "3x3 Eigendecomposition failed"

    return eigenvalues, Q


@ti.func
def sym_eig2x2(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real symmetric matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (ti.Matrix(2, 2)): input 2x2 symmetric matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        eigenvalues (ti.Vector(2)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (ti.Matrix(2, 2)): The eigenvectors. Each column stores one eigenvector.
    """
    assert all(A == A.transpose()), "A needs to be symmetric"
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    lambda1 = (tr + ops.sqrt(gap)) * 0.5
    lambda2 = (tr - ops.sqrt(gap)) * 0.5
    eigenvalues = ti.Vector([lambda1, lambda2], dt=dt)

    A1 = A - lambda1 * ti.Matrix.identity(dt, 2)
    A2 = A - lambda2 * ti.Matrix.identity(dt, 2)
    v1 = ti.Vector.zero(dt, 2)
    v2 = ti.Vector.zero(dt, 2)
    if all(A1 == ti.Matrix.zero(dt, 2, 2)) and all(A1 == ti.Matrix.zero(dt, 2, 2)):
        v1 = ti.Vector([0.0, 1.0]).cast(dt)
        v2 = ti.Vector([1.0, 0.0]).cast(dt)
    else:
        v1 = ti.Vector([A2[0, 0], A2[1, 0]], dt=dt).normalized()
        v2 = ti.Vector([A1[0, 0], A1[1, 0]], dt=dt).normalized()
    eigenvectors = ti.Matrix.cols([v1, v2])

    # Verify eigendecomposition
    Lambda = ti.Matrix.zero(dt, 2, 2)
    Lambda[0,0] = eigenvalues[0]
    Lambda[1,1] = eigenvalues[1] 
    recon_A = eigenvectors @ Lambda @ eigenvectors.transpose()
    assert ((A - recon_A).norm() < 1e-8), "2x2 Eigendecomposition failed"

    return eigenvalues, eigenvectors


@ti.func
def svd_3x2(A:ti.types.matrix(3, 2, ti.f64)):
    """3x2矩阵的奇异值分解, A = U * Σ * V^T, A为奇异矩阵会失败(未修复)

    Args:
        A (ti.types.matrix(3, 2, ti.f64)): 待分解的3x2矩阵

    Returns:
        U (ti.types.matrix(3, 3, ti.f64)): A的左奇异向量
        sigma (ti.types.vector(2, ti.f64)): A的奇异值
        V (ti.types.matrix(2, 2, ti.f64)): A的右奇异向量
    """
    dt = ti.f64
    # 计算 A^T*A 和 A*A^T
    ATA = A.transpose() @ A
    AAT = A @ A.transpose()
    
    # 特征值分解
    eigenvals_V, V = sym_eig2x2(ATA, dt)        # sim: 2*2
    eigenvals_U, U = sym_eig3x3(AAT, dt)        # sim: 3*3
    print("eigenvals_U:", eigenvals_U)
    print("U:", U)
    
    # 计算奇异值并排序
    # sigma = ti.sqrt(eigenvals_V)
    sigma = ti.Vector([ti.sqrt(eigenvals_V[0]), ti.sqrt(eigenvals_V[1])], dt=dt)
    # print("Inital sovle:\n", U, sigma, V)

    tmp = 0.0
    tmp_col = ti.Vector([0.0, 0.0], dt=dt)
    if sigma[1] > sigma[0]:
        tmp = sigma[0]
        sigma[0] = sigma[1]
        sigma[1] = tmp
        # 同时交换V的列
        tmp_col = V[:, 0]
        V[:, 0] = V[:, 1]
        V[:, 1] = tmp_col

    alignment = ti.Vector([0.0, 0.0], dt=dt)
    tmp_col = ti.Vector([0.0, 0.0], dt=dt)
    if ti.abs(sigma[0] - sigma[1]) < 1.e-5:
        for i in range(2):
            Av = A @ V[:, i]
            alignment[i] = ti.abs(Av.dot(U[:, 0]))
        if alignment[1] > alignment[0]:
            tmp_col = V[:, 0]
            V[:, 0] = V[:, 1]
            V[:, 1] = tmp_col

    # 确保奇异向量方向一致性
    for i in range(2):
        v = V[:, i]
        u = U[:, i]
        # 验证 A*v = σ*u
        Av = A @ v
        sigma_u = sigma[i] * u
        # 如果方向不一致，调整u的方向
        if (Av - sigma_u).norm() > (Av + sigma_u).norm():
            U[:, i] = -U[:, i]
    
    # print("Final solve:\n", U, sigma, V)
    # print("A_reconstructed:", U @ ti.Matrix([[sigma[0], 0], [0, sigma[1]], [0, 0]]) @ V.transpose())
    return U, sigma, V


@ti.func
def sym_eig2x2_new(A, dt):
    """Compute the eigenvalues and right eigenvectors (Av=lambda v) of a 2x2 real symmetric matrix.

    Mathematical concept refers to https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix.

    Args:
        A (ti.Matrix(2, 2)): input 2x2 symmetric matrix `A`.
        dt (DataType): date type of elements in matrix `A`, typically accepts ti.f32 or ti.f64.

    Returns:
        eigenvalues (ti.Vector(2)): The eigenvalues. Each entry store one eigen value.
        eigenvectors (ti.Matrix(2, 2)): The eigenvectors. Each column stores one eigenvector.
    """
    EPS = 1e-12
    assert all(A == A.transpose()), "A needs to be symmetric"
    a = ti.cast(A[0, 0], dt)
    b = ti.cast((A[0, 1] + A[1, 0])/2, dt)
    c = ti.cast(A[1, 1], dt)
    tr = ti.cast(a + c, dt)
    gap = ti.cast((a - c)**2 + 4 * b**2, dt)
    assert gap >= 0, "Gap is negative"
    lambda1 = ti.cast((tr + ops.sqrt(gap)) * 0.5, dt)
    lambda2 = ti.cast((tr - ops.sqrt(gap)) * 0.5, dt)
    eigenvalues = ti.Vector([lambda1, lambda2], dt=dt)

    A1 = A - lambda1 * ti.Matrix.identity(dt, 2)
    A2 = A - lambda2 * ti.Matrix.identity(dt, 2)
    v1 = ti.Vector.zero(dt, 2)
    v2 = ti.Vector.zero(dt, 2)
    if all(A1 == ti.Matrix.zero(dt, 2, 2)):
        v1 = ti.Vector([1.0, 0.0]).cast(dt)
        v2 = ti.Vector([0.0, 1.0]).cast(dt)
    else:
        if ti.abs((A[0, 1] + A[1, 0])/2) < EPS:
            v1 = ti.Vector([1.0, 0.0], dt=dt)
            v2 = ti.Vector([0.0, 1.0], dt=dt)
        else:
            v1 = ti.Vector([A1[0, 1], -A1[0, 0]], dt=dt).normalized()
            v2 = ti.Vector([A2[1, 1], -A2[1, 0]], dt=dt).normalized()
    assert v1.dot(v2) < EPS, "v1 and v2 are not orthogonal"
    eigenvectors = ti.Matrix.cols([v1, v2])

    # Verify eigendecomposition
    Lambda = ti.Matrix.zero(dt, 2, 2)
    Lambda[0, 0], Lambda[1, 1] = eigenvalues[0], eigenvalues[1]
    recon_A = eigenvectors @ Lambda @ eigenvectors.transpose()
    assert ((A - recon_A).norm() < 1e-8), "2x2 Eigendecomposition failed"

    return eigenvalues, eigenvectors
     

@ti.func
def svd_3x2_new(A):
    """SVD decomposition of 3*2 matrix 
    """
    dt = ti.f64
    ATA = A.transpose() @ A

    # 特征值分解
    eigenvals_V, V = sym_eig2x2_new(ATA, dt)        # sim: 2*2

    sigma = ti.Vector([ti.sqrt(eigenvals_V[0]), ti.sqrt(eigenvals_V[1])], dt=dt)

    tmp = 0.0
    tmp_col = ti.Vector([0.0, 0.0], dt=dt)
    if sigma[1] > sigma[0]:
        tmp = sigma[0]
        sigma[0] = sigma[1]
        sigma[1] = tmp
        # 同时交换V的列
        tmp_col = V[:, 0]
        V[:, 0] = V[:, 1]
        V[:, 1] = tmp_col

    U = ti.Matrix.zero(dt, 3, 3)
    for i in range(2):
        U[:, i] = tm.normalize(A @ V[:, i] / sigma[i])

    u3 = U[:, 0].cross(U[:, 1])
    U[:, 2] = u3.normalized()        # 归一化

    # Verify SVD decomposition
    Sigma = ti.Matrix.zero(dt, 3, 2)
    Sigma[0,0] = sigma[0]
    Sigma[1,1] = sigma[1]
    recon_A = U @ Sigma @ V.transpose()
    # assert ((A - recon_A).norm() < 1e-8), "3x2 SVD decomposition failed"

    return U, sigma, V



@ti.kernel
def test():
    A = ti.Matrix([[1.000000260773e+00, -5.993538998439e-11], [9.386224575358e-09, 9.999994824084e-01], [0.000000000000e+00, 0.000000000000e+00]], ti.f64)
    U, sigma, V = svd_3x2_new(A)
    print(f"A:{A:e}")
    print(f"U:{U:e}, \nSigma:{sigma:e}, \nV:{V:e}")
    print(f"reconstructed:{U @ ti.Matrix([[sigma[0], 0], [0, sigma[1]], [0, 0]]) @ V.transpose():e}")


@ti.kernel
def eigen2x2_test():
    F = ti.Matrix([[1.000000000000e+00, -3.330669073875e-16], [-2.370834950873e-17, 1.000000000000e+00], [0.000000000000e+00, 0.000000000000e+00]], ti.f64)
    A = F.transpose() @ F
    print(f"ATA: {A:e}")

    dt = ti.f64
    EPS = 1e-8
    assert all(A == A.transpose()), "A needs to be symmetric"
    tr = A.trace()
    det = A.determinant()
    gap = tr**2 - 4 * det
    print(f"Gap: {gap:e}")
    lambda1 = (tr + ops.sqrt(gap)) * 0.5
    lambda2 = (tr - ops.sqrt(gap)) * 0.5
    eigenvalues = ti.Vector([lambda1, lambda2], dt=dt)

    A1 = A - lambda1 * ti.Matrix.identity(dt, 2)
    A2 = A - lambda2 * ti.Matrix.identity(dt, 2)
    v1 = ti.Vector.zero(dt, 2)
    v2 = ti.Vector.zero(dt, 2)
    if all(A1 == ti.Matrix.zero(dt, 2, 2)):
        v1 = ti.Vector([1.0, 0.0]).cast(dt)
        v2 = ti.Vector([0.0, 1.0]).cast(dt)
    else:
        v1 = ti.Vector([A1[0, 1], -A1[0, 0]], dt=dt).normalized() if ti.abs(A1[0, 1]) > EPS else ti.Vector([1.0, 0.0], dt=dt)
        v2 = ti.Vector([A2[1, 1], -A2[1, 0]], dt=dt).normalized() if ti.abs(A2[1, 0]) > EPS else ti.Vector([0.0, 1.0], dt=dt)
    assert v1.dot(v2) < EPS, "v1 and v2 are not orthogonal"
    eigenvectors = ti.Matrix.cols([v1, v2])

    print(f"eigenvals_V: {eigenvalues:e}")
    print(f"V: {eigenvectors:e}")


    # Verify eigendecomposition
    Lambda = ti.Matrix.zero(dt, 2, 2)
    Lambda[0,0] = eigenvalues[0]
    Lambda[1,1] = eigenvalues[1] 
    recon_A = eigenvectors @ Lambda @ eigenvectors.transpose()
    print(f"recon_A: {recon_A:e}")


if __name__ == '__main__':
    ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)
    test()

    A = np.array([[1.000000260773e+00, -5.993538998439e-11], [9.386224575358e-09, 9.999994824084e-01], [0.000000000000e+00, 0.000000000000e+00]], dtype=np.float64)
    u, s, vh  = np.linalg.svd(A)
    print(f"u: {u}")
    print(f"s: {s}")
    print(f"v: {vh.T}")
    # eigen2x2_test()


