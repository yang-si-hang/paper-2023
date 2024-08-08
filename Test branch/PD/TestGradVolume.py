"""
检验3D版本中Volume Constraint的DiffPD求解的梯度是否正确
created at 2024-08-08 by hsy
"""

import taichi as ti
import taichi.math as tm
import numpy as np
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)
np.set_printoptions(linewidth=200)


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


@ti.func
def get_D_3d(sigma_1:float, sigma_2:float, sigma_3:float):
    D, max_itr, tol = ti.Vector([10., 10., 10.]), 80, 1.e-8
    ti.loop_config(serialize=True)
    for itr in range(max_itr):
        aa, bb, cc = D[0] + sigma_1, D[1] + sigma_2, D[2] + sigma_3
        C = aa * bb * cc - 1
        partial_C = ti.Vector([bb*cc, aa*cc, aa*bb])
    
        D_temp = (partial_C.dot(D)-C) / partial_C.dot(partial_C) * partial_C
        D_error = (D-D_temp).norm()
        D = D_temp
        if D_error < tol:
            break
    return ti.Vector([D[0]+sigma_1, D[1]+sigma_2, D[2]+sigma_3])


dim = 3
dBp_volume_dF_ti = ti.Matrix.field(dim**2, dim**2, dtype=ti.f64, shape=())
Omega_UVS_ti = ti.Matrix.field(dim, dim, dtype=ti.f64, shape=(dim , dim))


@ti.kernel
def main():
    dBp_volume_dF = ti.Matrix.zero(ti.f64, dim**2, dim**2)
    F = ti.Matrix([[1.2, 0.2, 0.1],
                   [0.1, 1.1, 0.2],
                   [0.2, 0.1, 1.3]], ti.f64)
    U_i, sig, V_i = ti.svd(F, ti.f64)
    s1, s2, s3 = sig[0, 0], sig[1, 1], sig[2, 2]

    # Volume Constraint的DiffPD求解
    # PP_i = self.PP[ele_idx]
    ss1, ss2, ss3 = get_D_3d(s1, s2, s3)
    PP_coef_A = ti.Matrix.zero(ti.f64, dim**2, dim**2)

    for d_idx in range(dim):
        PP_coef_A[d_idx, d_idx], PP_coef_A[d_idx, d_idx+dim], PP_coef_A[d_idx, d_idx+2*dim] = ss2*ss3, ss1*ss3, ss1*ss2

    for d_idx in range(dim):
        PP_coef_A[d_idx+dim, d_idx], PP_coef_A[d_idx+dim, d_idx+2*dim] = (s1-2*ss1), -(s3-2*ss3)

    for d_idx in range(dim):
        PP_coef_A[d_idx+2*dim, d_idx+dim], PP_coef_A[d_idx+2*dim, d_idx+2*dim] = (s2-2*ss2), -(s3-2*ss3)

    PP_coef_B = ti.Vector([0, 0, 0, -ss1, 0, ss3, 0, -ss2, ss3])
    dPP_dsig_vec = qr_solve9(PP_coef_A, PP_coef_B)
    dPP_dsig = ti.Matrix([[dPP_dsig_vec[0], dPP_dsig_vec[1], dPP_dsig_vec[2]],
                          [dPP_dsig_vec[3], dPP_dsig_vec[4], dPP_dsig_vec[5]],
                          [dPP_dsig_vec[6], dPP_dsig_vec[7], dPP_dsig_vec[8]]])

    for row_idx, col_idx in ti.ndrange(dim, dim):
        Omega_UVS = ti.Matrix.zero(ti.f64, 3, 3)
        # 用系数求和的方法
        A = ((U_i[row_idx, 0] * V_i[col_idx, 1] - U_i[row_idx, 1] * V_i[col_idx, 0]) / 2 /
            (s1 + s2) * (ss1 + ss2))
        B = ((U_i[row_idx, 0] * V_i[col_idx, 1] + U_i[row_idx, 1] * V_i[col_idx, 0]) / 2 )
        if ti.abs(s1 - s3) < 1.e-6:
            B *= (dPP_dsig[1, 1] - dPP_dsig[0, 1])      # 不同的s*对同一个s的梯度
        else:
            B *= (ss2 - ss1) / (s2 - s1)
        C = ti.cast(-B, ti.f64)
        Omega_UVS[0, 1] = A + B
        Omega_UVS[1, 0] = -(A + C)

        A = ((U_i[row_idx, 0] * V_i[col_idx, 2] - U_i[row_idx, 2] * V_i[col_idx, 0]) / 2 /
                (s1 + s3) * (ss1 + ss3))
        B = (U_i[row_idx, 0] * V_i[col_idx, 2] + U_i[row_idx, 2] * V_i[col_idx, 0]) / 2
        if ti.abs(s1 - s3) < 1.e-6:
            B *= (dPP_dsig[2, 2] - dPP_dsig[0, 2])
        else:
            B *= (ss3 - ss1) / (s3 - s1)
        C = ti.cast(-B, ti.f64)
        Omega_UVS[0, 2] = (A + B)
        Omega_UVS[2, 0] = -(A + C)

        A = ((U_i[row_idx, 1] * V_i[col_idx, 2] - U_i[row_idx, 2] * V_i[col_idx, 1]) / 2 /
            (s2 + s3) * (ss2 + ss3))
        B = (U_i[row_idx, 1] * V_i[col_idx, 2] + U_i[row_idx, 2] * V_i[col_idx, 1]) / 2
        if ti.abs(s2 - s3) < 1.e-6:
            B *= (dPP_dsig[2, 2] - dPP_dsig[1, 2])
        else:
            B *= (ss3 - ss2) / (s3 - s2)
        C = ti.cast(-B, ti.f64)
        Omega_UVS[1, 2] = (A + B)
        Omega_UVS[2, 1] = -(A + C)

        Omega_UVS_ti[row_idx, col_idx] = Omega_UVS

        dsig_df = ti.Vector([U_i[row_idx, 0] * V_i[col_idx, 0], U_i[row_idx, 1] * V_i[col_idx, 1],
                                U_i[row_idx, 2] * V_i[col_idx, 2]])
        dPP_df_vec = dPP_dsig @ dsig_df
        dPP_df = ti.Matrix.zero(ti.f64, 3, 3)
        dPP_df[0, 0], dPP_df[1, 1], dPP_df[2, 2] = dPP_df_vec[0], dPP_df_vec[1], dPP_df_vec[2]
        dBp_volume_df = U_i @ (Omega_UVS + dPP_df) @ V_i.transpose()
        dBp_volume_df_vec = ti.Vector([dBp_volume_df[0, 0], dBp_volume_df[0, 1], dBp_volume_df[0, 2],
                                        dBp_volume_df[1, 0], dBp_volume_df[1, 1], dBp_volume_df[1, 2],
                                        dBp_volume_df[2, 0], dBp_volume_df[2, 1], dBp_volume_df[2, 2]])
        dBp_volume_dF[:, dim*row_idx + col_idx] = dBp_volume_df_vec

    dBp_volume_dF_ti[None] = dBp_volume_dF



def get_D_3d_np(sigma_1, sigma_2, sigma_3):
    D, max_itr, tol = np.array([10., 10., 10.]), 80, 1.e-6
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


def finite_difference():
    eps = 1.e-8
    F_init = np.array([[1.2, 0.2, 0.1],
                       [0.1, 1.1, 0.2],
                       [0.2, 0.1, 1.3]])
    u_init, sig_init, vh_init = np.linalg.svd(F_init)
    s1_init, s2_init, s3_init = sig_init[0], sig_init[1], sig_init[2]
    ss1_init, ss2_init, ss3_init = get_D_3d_np(s1_init, s2_init, s3_init)
    T_init = u_init @ np.diag([ss1_init, ss2_init, ss3_init]) @ vh_init
    print('T_init:\n', T_init)

    J = np.zeros((dim**2, dim**2))
    J_u = np.zeros((dim**2, dim**2))
    J_v = np.zeros((dim**2, dim**2))
    J_uv = np.zeros((dim, dim, dim, dim))

    for row_idx, col_idx in np.ndindex(dim, dim):
        F = F_init.copy()
        F[row_idx, col_idx] += eps
        u, sig, vh = np.linalg.svd(F)
        s1, s2, s3 = sig[0], sig[1], sig[2]
        ss1, ss2, ss3 = get_D_3d_np(s1, s2, s3)
        T = u @ np.diag([ss1, ss2, ss3]) @ vh
        J[:, row_idx*dim+col_idx] = (T - T_init).flatten() / eps
        J_u_tmp = (u_init.transpose() @ (u - u_init) @ np.diag([ss1_init, ss2_init, ss3_init])) / eps
        J_v_tmp = (np.diag([ss1_init, ss2_init, ss3_init]) @ (vh - vh_init) @ vh_init.transpose()) / eps
        # print('J_u_tmp:\n', J_u_tmp)
        # print('J_v_tmp:\n', J_v_tmp)
        J_uv[row_idx, col_idx, :, :] = J_u_tmp + J_v_tmp
        print('J_u_tmp + J_v_tmp:\n', J_u_tmp + J_v_tmp)

    return J, J_uv


if __name__ == '__main__':
    main()
    grad = dBp_volume_dF_ti.to_numpy()
    Omega_UVS_np = Omega_UVS_ti.to_numpy()

    J, J_uv = finite_difference()

    error = (grad - J)
    # 整体的梯度误差
    print('Grad:\n', grad)
    print('J:\n', J)
    print('Error:\n', error)
    print('Error max:', np.max(np.abs(error)))

    # Omega_UVS的误差
    print('Omega_UVS_np:\n', Omega_UVS_np)
    print('J_uv:\n', J_uv)
    print('Error max:', np.max(np.abs(Omega_UVS_np - J_uv)))
