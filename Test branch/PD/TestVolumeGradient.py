"""
测试Volume Constraint中的gradient计算是否正确
created at 2024-7-7 by hsy
"""

from typing import Union
import taichi as ti
import numpy as np
import copy
np.set_printoptions(suppress=True, linewidth=200, precision=8)
ti.init(arch=ti.gpu, debug=True, default_fp=ti.f64)


@ti.kernel
def solve_test():
    A = ti.Matrix([[1, 1], [1, 1]])
    B = ti.Vector([2, 2])
    sol = ti.solve(A, B, ti.f64)
    nan_flag = ti.math.isnan(sol)
    print(nan_flag.sum())
    if nan_flag.sum() > 0:
        print('No solution')
    else:
        print(sol)

@ti.func
def get_D_3d(sigma_1:float, sigma_2:float, sigma_3:float) -> ti.types.vector(3, ti.f64):
    D, max_itr, tol = ti.Vector([10., 10., 10.], ti.f64), ti.i16(50), 1.e-9
    sol = ti.Vector([0., 0., 0.], ti.f64)
    ti.loop_config(serialize=True)
    for itr in range(max_itr):
        aa, bb, cc = D[0] + sigma_1, D[1] + sigma_2, D[2] + sigma_3
        C = aa * bb * cc - 1
        partial_C = ti.Vector([bb * cc, aa * cc, aa * bb])

        D_temp = (partial_C.dot(D) - C) / partial_C.dot(partial_C) * partial_C
        D_error = (D - D_temp).norm()
        D = D_temp
        if D_error < tol:
            sol = ti.Vector([D[0] + sigma_1, D[1] + sigma_2, D[2] + sigma_3], ti.f64)
    return sol


@ti.func
def qr_decomposition(A):
    n = A.get_shape()
    Q = ti.Matrix.zero(ti.f64, 9, 9)
    R = ti.Matrix.zero(ti.f64, 9, 9)

    ti.loop_config(serialize=True)
    for col_idx in range(n):
        Q[:, col_idx] = A[:, col_idx]
        for i in range(col_idx):
            R[i, col_idx] = Q[:, i].dot(A[:, col_idx])
            Q[:, col_idx] -= R[i, col_idx] * Q[:, i]
        R[col_idx, col_idx] = Q[:, col_idx].norm()
        Q[:, col_idx] /= R[col_idx, col_idx]

    return Q, R

@ti.func
def solve_qr(Q, R, b):
    y = Q.transpose() @ b
    n = b.get_shape()[0]
    x = ti.Vector.zero(ti.f64, 9)
    Rx = ti.Vector.zero(ti.f64, 9)

    ti.loop_config(serialize=True)
    for i in range(n):
        j = n - 1 - i
        for k in range(j, n):
            Rx[j] += R[j, k] * x[k]
        x[j] = (y[j] - Rx[j]) / R[j, j]

    return x

@ti.func
def my_solve(A, b):
    Q, R = qr_decomposition(A)
    x = solve_qr(Q, R, b)
    return x


@ti.func
def qr_solve9(A, b):
    # 用QR分解求解9*9的线性方程组
    Q = ti.Matrix.zero(ti.f64, 9, 9)        # 正交矩阵
    R = ti.Matrix.zero(ti.f64, 9, 9)        # 上三角矩阵

    # QR分解
    ti.loop_config(serialize=True)
    for col_idx in ti.static(range(9)):
        Q[:, col_idx] = A[:, col_idx]
        for i in range(col_idx):
            R[i, col_idx] = Q[:, i].dot(A[:, col_idx])
            Q[:, col_idx] -= R[i, col_idx] * Q[:, i]
        R[col_idx, col_idx] = Q[:, col_idx].norm()
        Q[:, col_idx] /= R[col_idx, col_idx]

    x = ti.Vector.zero(ti.f64, 9)
    Rx = ti.Vector.zero(ti.f64, 9)
    y = Q.transpose() @ b

    # 从最后一行往前的高斯消元
    ti.loop_config(serialize=True)
    for i in range(9):
        j = 9 - 1 - i
        for k in range(j, 9):
            Rx[j] += R[j, k] * x[k]
        x[j] = (y[j] - Rx[j]) / R[j, j]

    return x


def get_D_3d_np(sigma_1, sigma_2, sigma_3):
    D, max_itr, tol = np.array([10., 10., 10.]), 80, 1.e-9
    for itr in range(max_itr):
        aa, bb, cc = D[0] + sigma_1, D[1] + sigma_2, D[2] + sigma_3
        C = aa * bb * cc - 1
        partial_C = np.array([bb*cc, aa*cc, aa*bb])
    
        D_temp = (partial_C.dot(D)-C) / partial_C.dot(partial_C) * partial_C
        D_error = np.linalg.norm(D-D_temp)
        D = D_temp
        if D_error < tol:
            break
    return np.array([D[0]+sigma_1, D[1]+sigma_2, D[2]+sigma_3])


def finite_difference_s(F_np):
    U_init, s_init, Vh_init = np.linalg.svd(F_np)
    J = np.zeros((9, 3))
    finite_delta = 1.e-6
    for m, n in np.ndindex(F_np.shape):
        F_np_copy = copy.deepcopy(F_np)
        F_np_copy[m, n] += finite_delta
        U, s, Vh = np.linalg.svd(F_np_copy)
        J[m*3+n, :] = (s - s_init) / finite_delta
        print(f'{[m ,n]}:, {J[m*3+n, :]}')
    # print('Finite difference dsig_df:\n', J)


def finite_difference_S(s, ss):
    s1, s2, s3 = s
    ss1_init, ss2_init, ss3_init = ss
    finite_delta = 1.e-6
    J = np.zeros((3, 3))            # 一阶矩阵
    ss1_tmp, ss2_tmp, ss3_tmp = get_D_3d_np(s1+finite_delta, s2, s3)
    J[:, 0] = (np.array([ss1_tmp, ss2_tmp, ss3_tmp]) - np.array([ss1_init, ss2_init, ss3_init])) / finite_delta

    ss1_tmp, ss2_tmp, ss3_tmp = get_D_3d_np(s1, s2+finite_delta, s3)
    J[:, 1] = (np.array([ss1_tmp, ss2_tmp, ss3_tmp]) - np.array([ss1_init, ss2_init, ss3_init])) / finite_delta

    ss1_tmp, ss2_tmp, ss3_tmp = get_D_3d_np(s1, s2, s3+finite_delta)
    J[:, 2] = (np.array([ss1_tmp, ss2_tmp, ss3_tmp]) - np.array([ss1_init, ss2_init, ss3_init])) / finite_delta

    return J.flatten()


def finite_difference(F_np):
    U, s, Vh = np.linalg.svd(F_np)
    ss = get_D_3d_np(s[0], s[1], s[2])
    ss_diag = np.diag(ss)
    T_init = U @ ss_diag @ Vh
    dT_dF = np.zeros((9, 9))
    # J_flatten = finite_difference_S(s, ss)
    # print('J flattened', J_flatten)

    finite_delta = 1.e-8
    for m, n in np.ndindex(F_np.shape):
        F_np_copy = copy.deepcopy(F_np)
        F_np_copy[m, n] += finite_delta
        U, s, Vh = np.linalg.svd(F_np_copy)
        ss = get_D_3d_np(s[0], s[1], s[2])
        ss_diag = np.diag(ss)
        T = U @ ss_diag @ Vh
        dT_dF[m*3+n, :] = (T - T_init).flatten() / finite_delta
    
    print('Finite difference method:\n', dT_dF)
    return dT_dF


@ti.kernel
def partial():
    dim = 3
    F_i = F[0]
    U, sig, V = ti.svd(F_i, ti.f64)
    s1, s2, s3 = sig[0, 0], sig[1, 1], sig[2, 2]
    ss1, ss2, ss3 = get_D_3d(s1, s2, s3)
    print('Sigma:', s1, s2, s3)
    print('Sigma*:', ss1, ss2, ss3)

    PP_coef_A = ti.Matrix.zero(ti.f64, 9, 9)

    for d_idx in range(dim):
        PP_coef_A[d_idx, d_idx], PP_coef_A[d_idx, d_idx + dim], PP_coef_A[d_idx, d_idx + 2 * dim] = (
            ss2 * ss3, ss1 * ss3, ss1 * ss2)

    for d_idx in range(dim):
        PP_coef_A[d_idx + dim, d_idx], PP_coef_A[d_idx + dim, d_idx + 2 * dim] = (
            sig[0, 0] - 2 * ss1, -(sig[2, 2] - 2 * ss3))

    for d_idx in range(dim):
        PP_coef_A[d_idx + 2 * dim, d_idx + dim], PP_coef_A[d_idx + 2 * dim, d_idx + 2 * dim] = (
            sig[1, 1] - 2 * ss2), -(sig[2, 2] - 2 * ss3)

    PP_coef_B = ti.Vector([0, 0, 0, -ss1, 0, ss3, 0, -ss2, ss3])
    # dPP_dsig_vec = coef_A.inverse() @ coef_B
    dPP_dsig_vec = qr_solve9(PP_coef_A, PP_coef_B)
    # print('dPP_dsig_vec:', dPP_dsig_vec)

    dPP_dsig = ti.Matrix([[dPP_dsig_vec[0], dPP_dsig_vec[1], dPP_dsig_vec[2]],
                          [dPP_dsig_vec[3], dPP_dsig_vec[4], dPP_dsig_vec[5]],
                          [dPP_dsig_vec[6], dPP_dsig_vec[7], dPP_dsig_vec[8]]])
    # print('dPP_dsig:', dPP_dsig)

    # ti.loop_config(serialize=True)
    for row_idx, col_idx in ti.ndrange(3, 3):
        Omega_UVS = ti.Matrix.zero(ti.f64, 3, 3)
        # 用系数求和的方法
        A = ((U[row_idx, 0] * V[col_idx, 1] - U[row_idx, 1] * V[col_idx, 0]) / 2 / 
             (s1 + s2) * (ss1 + ss2))
        B = ((U[row_idx, 0] * V[col_idx, 1] + U[row_idx, 1] * V[col_idx, 0]) / 2 )
        if ti.abs(s1 - s3) < 1.e-6:
            B *= (dPP_dsig[1, 1] - dPP_dsig[0, 1])      # 不同的s*对同一个s的梯度
        else:
            B *= (ss2 - ss1) / (s2 - s1)
        C = ti.cast(-B, ti.f64)
        Omega_UVS[0, 1] = A + B
        Omega_UVS[1, 0] = -(A + C)

        # 用解方程的方法
        # Omega_coef_A = ti.Matrix([[s2, s1], [s1, s2]])
        # Omega_coef_B = ti.Vector([U[row_idx, 0] * V[col_idx, 1], -U[row_idx, 1] * V[col_idx, 0]])

        # sol = ti.solve(Omega_coef_A, Omega_coef_B, ti.f64)
        # print('sol', [row_idx, col_idx], ':', sol)

        # Omega_UVS[0, 1] = ss2*sol[0] + ss1*sol[1]
        # Omega_UVS[1, 0] = -ss1*sol[0] - ss2*sol[1]
        # print('Omega_UVS', [row_idx, col_idx], ':', Omega_UVS)

        A = ((U[row_idx, 0] * V[col_idx, 2] - U[row_idx, 2] * V[col_idx, 0]) / 2 / 
            (s1 + s3) * (ss1 + ss3))
        B = (U[row_idx, 0] * V[col_idx, 2] + U[row_idx, 2] * V[col_idx, 0]) / 2 
        if ti.abs(s1 - s3) < 1.e-6:
            B *= (dPP_dsig[2, 2] - dPP_dsig[0, 2])
        else:
            B *= (ss3 - ss1) / (s3 - s1)
        C = ti.cast(-B, ti.f64)
        Omega_UVS[0, 2] = (A + B)
        Omega_UVS[2, 0] = -(A + C)

        A = ((U[row_idx, 1] * V[col_idx, 2] - U[row_idx, 2] * V[col_idx, 1]) / 2 / 
            (s2 + s3) * (ss2 + ss3))
        B = (U[row_idx, 1] * V[col_idx, 2] + U[row_idx, 2] * V[col_idx, 1]) / 2 
        if ti.abs(s2 - s3) < 1.e-6:
            B *= (dPP_dsig[2, 2] - dPP_dsig[1, 2])
        else:
            B *= (ss3 - ss2) / (s3 - s2)
        C = ti.cast(-B, ti.f64)
        Omega_UVS[1, 2] = (A + B)
        Omega_UVS[2, 1] = -(A + C)
            # Omega_coef_A = ti.Matrix([[s3, s2], [s2, s3]])
            # Omega_coef_B = ti.Vector([U[row_idx, 1]*V[col_idx, 2], -U[row_idx, 2]*V[col_idx, 1]])
            # sol = ti.solve(Omega_coef_A, Omega_coef_B, ti.f64)
            # # print('sol', sol)
            # Omega_UVS[1, 2] = ss3*sol[0] + ss2*sol[1]
            # Omega_UVS[2, 1] = -ss2*sol[0] - ss3*sol[1]

        # print('Omega_UVS:', Omega_UVS)
        dsig_df = ti.Vector([U[row_idx, 0] * V[col_idx, 0], U[row_idx, 1] * V[col_idx, 1],
                            U[row_idx, 2] * V[col_idx, 2]])
        dPP_df_vec = dPP_dsig @ dsig_df
        dPP_df = ti.Matrix.zero(ti.f64, 3, 3)
        dPP_df[0, 0], dPP_df[1, 1], dPP_df[2, 2] = dPP_df_vec[0], dPP_df_vec[1], dPP_df_vec[2]
        dBp_df = U @ (Omega_UVS + dPP_df) @ V.transpose()
        dBp_df_vec = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[0, 2],
                                dBp_df[1, 0], dBp_df[1, 1], dBp_df[1, 2],
                                dBp_df[2, 0], dBp_df[2, 1], dBp_df[2, 2]])
        # print('dBp_df:', dBp_df)
        # dBp_dF[row_idx][:, col_idx] = dBp_df_vec
        dBp_dF[0][:, dim*row_idx + col_idx] = dBp_df_vec

F = ti.Matrix.field(3, 3, dtype=ti.f64, shape=1)
sig_np = np.diag([1.200, 1.2, 1.2])
dBp_dF = ti.Matrix.field(9, 9, dtype=ti.f64, shape=1)

# np.random.seed(42)
random_matrix = np.random.randn(3, 3)
U_construct, _ = np.linalg.qr(random_matrix)
print('U_construct:\n', U_construct)

random_matrix = np.random.randn(3, 3)
V_construct, _ = np.linalg.qr(random_matrix)
print('V_construct:\n', V_construct)

F_np = U_construct @ sig_np @V_construct.transpose()
print('F_np:\n', F_np)
F.from_numpy(np.expand_dims(F_np, axis=0))
partial()
print('Analytic method:\n', dBp_dF[0].to_numpy())

# finite_difference_s(F_np)

finite_gradient = finite_difference(F[0].to_numpy())

print(f'Error:\n {dBp_dF[0].to_numpy()-finite_gradient}')