"""
所有的矩阵每一个都应该检查维度，是否存在转置。以及每一个矩阵运算是否正确。
"""

import numpy as np


def main():
    a0, b0, c0 = np.array([0.,0.]), np.array([0.,1.]), np.array([1.,0.])
    a, b, c = np.array([0.,0.]), np.array([-1.,1.]), np.array([1.,0.])
    q = np.hstack((a, b, c))
    B = np.vstack((b0 - a0, c0 - a0)).T
    D = np.vstack((b - a, c - a)).T
    F0 = D @ np.linalg.inv(B)

    # Cal gradient F w.r.t. a,b,c
    B_inv = np.linalg.inv(B)
    B11, B12, B21, B22 = B_inv[0,0], B_inv[0,1], B_inv[1,0], B_inv[1,1]
    A = np.zeros([4,6])
    A[0, 0] = -B11-B21
    A[0, 2] = B11
    A[0, 4] = B21
    A[1, 0] = -B12-B22
    A[1, 2] = B12
    A[1, 4] = B22
    A[2, 1] = -B11-B21
    A[2, 3] = B11
    A[2, 5] = B21
    A[3, 1] = -B12-B22
    A[3, 3] = B12
    A[3, 5] = B22
    print(A)
    print('------------------')

    # Check gradient F
    eps = 1.e-5
    A_diff = np.zeros([4,6])         # Matrix A using finite difference
    for i in range(6):
        q_tmp = q.copy()
        q_tmp[i] += eps
        D_tmp = np.array([[q_tmp[2] - q_tmp[0], q_tmp[4] - q_tmp[0]],
                          [q_tmp[3] - q_tmp[1], q_tmp[5] - q_tmp[1]]])
        # D_tmp = D_tmp.T
        F_tmp = D_tmp @ np.linalg.inv(B)
        A_diff[:, i] = (F_tmp - F0).reshape(-1, order='C')/eps
        # print((F_tmp - F0)/eps)
    print(A_diff)

    print('------------------')
    print('A error: ', np.linalg.norm(A_diff - A))
    print('------------------')

    # Check deformation gradient
    # F0 = np.array([[0., -1.], [1., 0.]])
    # F = u@diag(s)@v
    u0, s0, v0 = np.linalg.svd(F0)
    T0 = u0 @ v0

    # Get gradient of T w.r.t. a,b,c using finite difference
    eps = 1.e-5
    dT_diff = np.zeros((4,6))
    for i in range(6):
        q_tmp = q.copy()
        q_tmp[i] += eps
        D_tmp = np.array([[q_tmp[2] - q_tmp[0], q_tmp[4] - q_tmp[0]],
                          [q_tmp[3] - q_tmp[1], q_tmp[5] - q_tmp[1]]])
        # D_tmp = D_tmp.T
        F_tmp = D_tmp @ np.linalg.inv(B)
        u_tmp, s_tmp, v_tmp = np.linalg.svd(F_tmp)
        T = u_tmp @ v_tmp
        dT_diff[:, i] = ((T - T0)/eps).reshape(-1, order='C')
        # print(((T - T0)/eps).reshape(-1, order='C'))
    print(dT_diff)

    print('------------------')

    # Get gradient of T w.r.t. a,b,c using analytical method
    dT_dF = np.zeros((4, 4))
    for i,j in np.ndindex(2, 2):
        Omega_uv = np.zeros([2,2])
        Omega_uv[0,1] = (u0[i,0]*v0[1,j] - u0[i,1]*v0[0,j])/(s0[0]+s0[1])
        Omega_uv[1,0] = -Omega_uv[0,1]
        dT_df = u0 @ Omega_uv @ v0
        dT_df_vec = dT_df.reshape(-1, order='C')
        # print(dT_df_vec@A)
        dT_dF[:, 2*i+j] = dT_df_vec
        # print(u0@Omega_uv@v0)

    print(dT_dF@A)

    print('------------------')
    print('dT error: ', np.linalg.norm(dT_diff - dT_dF@A))




    exit(0)
    eps = 1e-4
    EPS = np.zeros((2, 2))
    i, j = 0, 0
    EPS[i, j] += eps
    F1 = np.array([[0., -1.], [1., 0.]]) + EPS
    np.linalg.svd(F1)
    u1, s1, v1 = np.linalg.svd(F1)

    # gradient_u = (u1 - u0)/eps
    # gradient_s = (np.diag(s1) - np.diag(s0))/eps
    # gradient_v_T = (v1.T - v0.T)/eps
    #
    # gradient_F = gradient_u @ np.diag(s0) @ v0.T + u0 @ gradient_s @ v0.T + \
    #              u0 @ np.diag(s0) @ gradient_v_T
    # print(gradient_F)

    gradient_s = np.diag([u0[i,0]*v0[j,0],u0[i,1]*v0[j,1]])
    print(gradient_s)


if __name__ == '__main__':
    main()