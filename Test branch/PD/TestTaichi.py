
import taichi as ti
ti.init(arch=ti.gpu, debug=True, advanced_optimization=False)

@ti.func
def qr_decomposition(A):
    # n = A.get_shape()[0]
    Q = ti.Matrix.zero(ti.f64, 9, 9)
    R = ti.Matrix.zero(ti.f64, 9, 9)

    ti.loop_config(serialize=True)
    for col_idx in ti.static(range(9)):
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


A = ti.Matrix([[1.25, 0., 0., 1., 0., 0., 0.8, 0., 0.],
                [0., 1.25, 0., 0., 1., 0., 0., 0.8, 0.],
                [0., 0., 1.25, 0., 0., 1., 0., 0., 0.8],
                [-0.8, 0., 0., 0., 0., 0., 1.25, 0., 0.],
                [0., -0.8, 0., 0., 0., 0., 0., 1.25, 0.],
                [0., 0., -0.8, 0., 0., 0., 0., 0., 1.25],
                [0., 0., 0., -1., 0., 0., 1.25, 0., 0.],
                [0., 0., 0., 0., -1., 0., 0., 1.25, 0.],
                [0., 0., 0., 0., 0., -1., 0., 0., 1.25]])
B = ti.Vector([ 0., 0., 0., -0.8, 0., 1.25, 0., -1., 1.25])


@ti.kernel
def main():
    pass
    Q, R = qr_decomposition(A)
    # print(Q, R)

if __name__ == '__main__':
    main()
