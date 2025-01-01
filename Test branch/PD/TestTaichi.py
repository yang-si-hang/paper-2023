import os, sys
from collections import defaultdict
import numpy as np
np.set_printoptions(linewidth=200)
from numba import njit, prange
import taichi as ti
import taichi.math as tm
ti.init(arch=ti.cpu, debug=True)

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.MathTaichi import svd_3x2_new


@ti.kernel
def test():
    a = ti.Vector([1, 2, 3])
    b = ti.Vector([4, 5])
    c = ti.Matrix([[1, 2], [3, 4], [5, 6]])
    print(a.outer_product(b))
    print(b.outer_product(a))
    print(c.norm())


test()
exit()


@ti.kernel
def compute_F_grad(q:ti.types.matrix(3, 2, ti.f64), q_new:ti.types.matrix(3, 3, ti.f64)):
    q0, q1, q2 = q[0,:], q[1,:], q[2,:]
    B_inv = ti.Matrix.cols([q1-q0, q2-q0])
    B = B_inv.inverse()
    a, b, c, d = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    F_A = ti.Matrix([[-a-c, a, c], [-b-d, b, d]])

    X_f = ti.Matrix.cols([q_new[1,:]-q_new[0,:], q_new[2,:]-q_new[0,:]])
    F = ti.cast(X_f @ B, ti.f64)

    f_Ai = F_A
    print(f_Ai)
    U, sig, V = svd_3x2_new(F)  # 替换为直接读取的方式,不要重复计算

    dBp_dq = ti.Matrix.zero(ti.f64, 6, 9)
    dBp_dF = ti.Matrix.zero(ti.f64, 6, 6)
    for m in range(3):
        dBp_df_n = ti.Matrix.zero(ti.f64, 6, 2)     # 按维度拼接
        for n in range(2):
            Omega_uv = ti.Matrix.zero(ti.f64, 3, 2)
            Omega_uv[0, 1] = (U[m,0]*V[n,1] - U[m,1]*V[n,0]) / (sig[0] + sig[1])
            Omega_uv[1, 0] = -Omega_uv[0, 1]
            Omega_uv[2, 0] = U[m,2]*V[n,0] / sig[0]
            Omega_uv[2, 1] = U[m,2]*V[n,1] / sig[1]
            dBp_df = U @ Omega_uv @ V.transpose()
            # 行优先，按维度拼接
            dBp_df_n[:, n] = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[1, 0], dBp_df[1, 1], dBp_df[2, 0], dBp_df[2, 1]])
            dBp_dF[2*m+n, :] = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[1, 0], dBp_df[1, 1], dBp_df[2, 0], dBp_df[2, 1]])
            print(f"dBp_df_n: {dBp_df_n[:, n]}")

        dBp_df_m = dBp_df_n @ f_Ai
        dBp_dq[:, 3*m+0] = dBp_df_m[:, 0]
        dBp_dq[:, 3*m+1] = dBp_df_m[:, 1]
        dBp_dq[:, 3*m+2] = dBp_df_m[:, 2]

    print("dBp_dq:-----------------------------------------")
    for i in range(6):
        print(dBp_dq[i, :])

    f_Ai_kron = ti.Matrix.zero(ti.f64, 9, 6)
    I3 = ti.Matrix.identity(ti.f64, 3)
    f_AiT = f_Ai.transpose()
    for i, j in ti.ndrange(3, 3):
        for k, l in ti.ndrange(3, 2):
            f_Ai_kron[i*3 + k, j*2 + l] = I3[i, j] * f_AiT[k, l]
            
    AT_dBp_dq = f_Ai_kron @ dBp_dq
    print(f"AT_dBp_dq:-----------------------------------------")
    for i in range(9):
        print(AT_dBp_dq[i, :])

    for m, n in ti.ndrange(3, 3):
        dBp_dF_i = ti.Matrix.zero(ti.f64, 2, 2)
        for i, j in ti.ndrange(2, 2):
            dBp_dF_i[i, j] = dBp_dF[2*m+i, 2*n+j]
        AT_dBp_dq_i = f_Ai.transpose() @ dBp_dF_i @ f_Ai
        print(f"AT_dBp_dq_i: {AT_dBp_dq_i}")


def compute_F_difference(q0, q_deform):
    B_inv = np.array([q0[1]-q0[0], q0[2]-q0[0]]).T
    B = np.linalg.inv(B_inv)
    # a, b, c, d = B[0, 0], B[0, 1], B[1, 0], B[1, 1]
    # F_A = np.array([[-a-c, a, c], [-b-d, b, d]])

    X_f = np.array([q_deform[1]-q_deform[0], q_deform[2]-q_deform[0]]).T
    F = X_f @ B
    u, s, vh = np.linalg.svd(F)
    Bp = u @ np.array([[1., 0.], [0., 1.], [0., 0.]]) @ vh

    dBp = np.zeros((6, 9))
    for i in range(3):
        for j in range(3):
            q_deform_add = q_deform.copy()
            q_deform_add[i, j] += 1e-6
            X_f = np.array([q_deform_add[1]-q_deform_add[0], q_deform_add[2]-q_deform_add[0]]).T
            F = X_f @ B
            u, s, vh = np.linalg.svd(F)
            Bp_add = u @ np.array([[1., 0.], [0., 1.], [0., 0.]]) @ vh

            # 按维度拼接
            dBp[:, 3*j+i] = (Bp_add - Bp).reshape(-1, order="C") / 1.e-6

    print(f"dBp:\n{dBp}")


q0 = ti.Matrix([[0.0, 0.0], [1.1, 0.15], [-0.1, 0.95]], dt=ti.f64)
q_deform = ti.Matrix([[0.0, 0.0, 0.0], [1.1, 0.0, 0.1], [0.0, 1.2, -0.1]], dt=ti.f64)
compute_F_grad(q0, q_deform)

compute_F_difference(q0.to_numpy(), q_deform.to_numpy())