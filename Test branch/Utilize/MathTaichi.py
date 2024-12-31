"""
一些基于taichi的数学工具函数
created at 2024-12-12 by hsy
"""

import numpy as np
import taichi as ti
import taichi.math as tm
from taichi.lang import impl, ops


@ti.func
def cotangent_ti(u, v):
    """计算两个三维向量的cot值
    """
    assert u.get_shape() == v.get_shape(), "The shape of two vectors should be the same"
    cross = u.cross(v).norm()
    return u.dot(v) / u.cross(v).norm() if cross > 1.e-8 else 0.0


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
    ORTHOGONAL_EPS = 1e-4 if dt == ti.f32 else 1e-11
    DECOMP_EPS = 1e-4 if dt == ti.f32 else 1e-11

    # print(f"A: {A:e}")
    # print(A==A.transpose())
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
        # if ti.abs((A[0, 1] + A[1, 0]) / 2) < EPS:
        #     v1 = ti.Vector([1.0, 0.0], dt=dt)
        #     v2 = ti.Vector([0.0, 1.0], dt=dt)
        # else:
        if ti.abs(A1[0, 0]) > ti.abs(A1[1, 1]):
            v1 = ti.Vector([A1[0, 1], -A1[0, 0]], dt=dt)
        else:
            v1 = ti.Vector([A1[1, 1], -A1[1, 0]], dt=dt)
            # if ti.abs(A2[0, 0]) > ti.abs(A2[1, 1]):
            #     v2 = ti.Vector([A2[0, 1], -A2[0, 0]], dt=dt)
            # else:
            #     v2 = ti.Vector([A2[1, 1], -A2[1, 0]], dt=dt)

        v2 = ti.Vector([-v1[1], v1[0]], dt=dt)
        # print(f"A1: {A1:e}; A2: {A2:e}")
        # print(f"v1: {v1:e}; v2: {v2:e}")
        # print(f"v1 norm: {v1.norm():e}; v2 norm: {v2.norm():e}")
        # print(f"v1 dot v2 (no norm): {v1 @ v2:e}")
        v1 = v1.normalized()
        v2 = v2.normalized()
        # print(f"v1 dot v2: {ti.abs(v1.dot(v2)):e}")
    assert ti.abs(v1.dot(v2)) < ORTHOGONAL_EPS, "v1 and v2 are not orthogonal"
    eigenvectors = ti.Matrix.cols([v1, v2])

    # Verify eigendecomposition
    Lambda = ti.Matrix.zero(dt, 2, 2)
    Lambda[0, 0], Lambda[1, 1] = eigenvalues[0], eigenvalues[1]
    recon_A = eigenvectors @ Lambda @ eigenvectors.transpose()
    # print(f"recon error: {(A - recon_A).norm():e}")
    assert ((A - recon_A).norm() < DECOMP_EPS), "2x2 Eigendecomposition failed"

    return eigenvalues, eigenvectors
     

@ti.func
def svd_3x2_new(A):
    """SVD decomposition of 3*2 matrix A.
    
    Args:
        A (ti.types.matrix(3, 2)): 

    Returns:
        U (ti.types.matrix(3, 3)):
        sigma (ti.types.vector(2)):
        V (ti.types.matrix(2, 2)):
    """
    dt = ti.f64
    for m, n in ti.ndrange(3, 2):
        assert ti.math.isnan(A[m, n]) == 0, "A contains NaN"
        assert ti.math.isinf(A[m, n]) == 0, "A contains infinite value"
    ATA = ti.cast(A.transpose() @ A, dt)

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
    assert ((A - recon_A).norm() < 1e-8), "3x2 SVD decomposition failed"

    if U.determinant() < 0:
        U[:, 2] = -U[:, 2]
        # sigma[0] = -sigma[0]
    if V.determinant() < 0:
        V[:, 1] = -V[:, 1]
        sigma[1] = -sigma[1]

    return U, sigma, V


@ti.kernel
def test():
    a = ti.Matrix([[1.0, 2.0], [2.0, 1.0]])
    b = ti.Matrix.identity(ti.f32, 2)

    c = kronecker_product(a, b)
    print(c)


if __name__ == '__main__':
    ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

    test()