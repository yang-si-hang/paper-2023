from typing import Union
import taichi as ti
import numpy as np

ti.init(arch=ti.gpu, debug=True)

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

# solve_test()


@ti.func
def mp_inverse(X):
    u, s, v = ti.svd(X)
    m, n = s.get_shape()
    s_inv = ti.Matrix.zero(ti.f64, m, n)
    for i in range(min(m, n)):
        if s[i, i] > 1e-6:
            s_inv[i, i] = 1/s[i, i]
    mp_inv = v @ s_inv @ u.transpose()
    return mp_inv


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


def finite_difference(F_np):
    u, s, vh = np.linalg.svd(F_np)



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
    print(PP_coef_A.get_shape())

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

    dPP_dsig = ti.Matrix([[dPP_dsig_vec[0], dPP_dsig_vec[1], dPP_dsig_vec[2]],
                          [dPP_dsig_vec[3], dPP_dsig_vec[4], dPP_dsig_vec[5]],
                          [dPP_dsig_vec[6], dPP_dsig_vec[7], dPP_dsig_vec[8]]])

    for row_idx, col_idx in ti.ndrange(3, 3):
        Omega_UVS = ti.Matrix.zero(ti.f64, 3, 3)
        # 用MP逆求解
        if ti.abs(s1-s2) < 1.e-6:
            Omega_UVS[0, 1] = ((U[row_idx, 0]*V[col_idx, 1] - U[row_idx, 1]*V[col_idx, 0]) /
                               (s1+s2) * (ss1+ss2)/2)
            Omega_UVS[1, 0] = -Omega_UVS[0, 1]
        else:
            Omega_coef_A = ti.Matrix([[s2, s1], [s1, s2]])
            Omega_coef_B = ti.Vector([U[row_idx, 0] * V[col_idx, 1], -U[row_idx, 1] * V[col_idx, 0]])
            print(Omega_coef_A.get_shape(), Omega_coef_B.get_shape())
       
            sol = ti.solve(Omega_coef_A, Omega_coef_B, ti.f64)
            # 或者用系数求和
            # A = ((U[row_idx, 0] * V[col_idx, 1] - U[row_idx, 1] * V[col_idx, 0]) /
            #      (s1 + s2) * (ss1 + ss2) / 2)
            # B = ((U[row_idx, 0] * V[col_idx, 1] + U[row_idx, 1] * V[col_idx, 0]) /
            #      (s1 - s2) * (ss1 - ss2) / 2)
            # C = ((U[row_idx, 0] * V[col_idx, 1] + U[row_idx, 1] * V[col_idx, 0]) /
            #      (s1 - s2) * (ss2 - ss1) / 2)
            # Omega_UVS[0, 1] = A + B
            # Omega_UVS[1, 0] = A + C
            Omega_UVS[0, 1] = ss2*sol[0] + ss1*sol[1]
            Omega_UVS[1, 0] = -ss1*sol[0] - ss2*sol[1]

        if ti.abs(s1 -s3) < 1.e-6:
            Omega_UVS[0, 2] = ((U[row_idx, 0] * V[col_idx, 2] - U[row_idx, 2] * V[col_idx, 0]) /
                               (s1 + s3) * (ss1 + ss3) / 2)
            Omega_UVS[2, 0] = -Omega_UVS[0, 2]
        else:
            Omega_coef_A = ti.Matrix([[s3, s1], [s1, s3]])
            Omega_coef_B = ti.Vector([U[row_idx, 0]*V[col_idx, 2], -U[row_idx, 2]*V[col_idx, 0]])
            sol = ti.solve(Omega_coef_A, Omega_coef_B, ti.f64)
            Omega_UVS[0, 2] = ss3*sol[0] + ss1*sol[1]
            Omega_UVS[2, 0] = -ss1*sol[0] - ss3*sol[1]

        if ti.abs(s2 - s3) < 1.e-6:
            Omega_UVS[1, 2] = ((U[row_idx, 1] * V[col_idx, 2] - U[row_idx, 2] * V[col_idx, 1]) /
                               (s2 + s3) * (ss2 + ss3) / 2)
            Omega_UVS[2, 1] = -Omega_UVS[1, 2]
        else:
            Omega_coef_A = ti.Matrix([[s3, s2], [s2, s3]])
            Omega_coef_B = ti.Vector([U[row_idx, 1]*V[col_idx, 2], -U[row_idx, 2]*V[col_idx, 1]])
            sol = ti.solve(Omega_coef_A, Omega_coef_B, ti.f64)
            # print('sol', sol)
            Omega_UVS[1, 2] = ss3*sol[0] + ss2*sol[1]
            Omega_UVS[2, 1] = -ss2*sol[0] - ss3*sol[1]

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
        dBp_dF[0][:, dim*row_idx+col_idx] = dBp_df_vec


F = ti.Matrix.field(3, 3, dtype=ti.f64, shape=1)
sig_np = np.diag([1., 1., 0.9])
dBp_dF = ti.Matrix.field(9, 9, dtype=ti.f64, shape=1)

random_matrix = np.random.randn(3, 3)
U_construct, _ = np.linalg.qr(random_matrix)

random_matrix = np.random.randn(3, 3)
V_construct, _ = np.linalg.qr(random_matrix)

F_np = U_construct @ sig_np @V_construct.transpose()
print('F_np:', F_np)
F.from_numpy(np.expand_dims(F_np, axis=0))
partial()
print(dBp_dF.to_numpy())

finite_difference(F[0].to_numpy())