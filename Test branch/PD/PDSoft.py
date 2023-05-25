
# Acknowledgement: ti example fem99.py
# Demo 3

import taichi as ti
import numpy as np
import time
import csv
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import factorized

ti.init(arch=ti.gpu, default_fp=ti.f64, debug=False)

dim = 2
N = 20  # internal of one edge
W = 20
dt = 1.0/480
dx = 1 / N  # 0.05
rho = 1.e3           # density
NF = 2 * N * W # 2 * N ** 2   # number of faces
NV = (N+1)*(W+1) # (N + 1) ** 2 # number of vertices
E, nu = 5.e4, 0.25  # Young's modulus and Poisson's ratio
mu, lam = E / (2*(1+nu)), E * nu / ((1+nu)*(1-2*nu))  # Lame parameters
ball_pos, ball_radius = ti.Vector([0.5, 0.0]), 0.32
# gravity = ti.Vector([0, -9.8])
gravity = ti.Vector([0.0, 0.0])
GRASP_VEL = ti.Vector([0.005, 0.005])
# Area: 0.000061 0.02*0.02*sin90*0.5
volume = 0.0000125
m_weight_strain = mu * 2 * volume
m_weight_volume = lam * dim * volume
m_weight_positional = 1.e8
print("m_weight_strain/volume", m_weight_strain/volume, "  m_weight_volume/volume", m_weight_volume/volume)

mass = ti.field(ti.f64, NV)

pos = ti.Vector.field(2, ti.f64, NV)
pos_new = ti.Vector.field(2, ti.f64, NV)
pos_init = ti.Vector.field(2, ti.f64, NV)
pos_latest = ti.Vector.field(2, ti.f64, NV)
last_pos_new = ti.Vector.field(2, ti.f64, NV)
particle_show = ti.Vector.field(3, ti.f32, NV)
boundary_labels = ti.field(int, NV)         # 固定约束定义的边

vel = ti.Vector.field(2, ti.f64, NV)
f2v = ti.Vector.field(3, ti.i32, NF)  # ids of three vertices of each face
B = ti.Matrix.field(2, 2, ti.f64, NF)  # The inverse of the init elements -- Dm
F = ti.Matrix.field(2, 2, ti.f64, NF)
A = ti.Matrix.field(4, 6, ti.f64, NF * 2)
Bp = ti.Matrix.field(2, 2, ti.f64, NF * 2)
rhs_np = np.zeros(NV * 2, dtype=np.float64)                 # global update 中的右侧矩阵

Sn = ti.field(ti.f64, NV * 2)
lhs_matrix = ti.field(ti.f64, shape=(NV * 2, NV * 2))       # global update 中的左侧矩阵（Left hand matrix）
phi = ti.field(ti.f64, NF)  # potential energy of each element(face) for linear coratated elasticity material.

tri_per_color = ti.Vector.field(3, ti.f32, shape=NF)
tri_idx = ti.Vector.field(3, ti.i32, shape=NF)
# line_idx = ti.Vector.field(2, ti.i32, shape=)

resolutionX = 512
# pixels = ti.var(ti.f32, shape=(resolutionX, resolutionX))

# drag = 0.2
drag = 0.0

solver_max_iteration = 10
solver_stop_residual = 0.0001


@ti.kernel
def init_pos():
    # 初始化节点位置
    for i, j in ti.ndrange(N + 1, W + 1):
        k = i*(W+1)+j
        pos[k] = ti.Vector([i/N*0.1, j/W*0.1]) + ti.Vector([0.0, -0.05]) # 0.2, 0.4 - 0.6,
        # 0.6  0.02*0.02
        pos_init[k] = pos[k]
        vel[k] = ti.Vector([0, 0])
        if i == 0:
            boundary_labels[k] = 1
        else:
            boundary_labels[k] = 0
    for i in range(NF): # NF number of face
        ia, ib, ic = f2v[i]
        a, b, c = pos[ia], pos[ib], pos[ic]
        B_i_inv = ti.Matrix.cols([b - a, c - a])  # rest B
        B[i] = B_i_inv.inverse()  # rest of B inverse


@ti.kernel
def init_mesh():  # generate two triangles
    # 生成三角形网格的节点序号
    for i, j in ti.ndrange(N, W):
        k = (i * W + j) * 2  # tirangle index   w 2 n 3
        a = i * (W + 1) + j  # 0 0 = 0
        b = a + 1  # 1
        c = a + W + 2  # 12
        d = a + W + 1  # 11
        f2v[k + 0] = [a, d, c]
        f2v[k + 1] = [a, c, b]


def init_edge():
    edge_set = set()
    for ele_idx in range(NF):
        ele_temp = f2v[ele_idx].to_numpy()
        for i in range(3):
            edge_temp = tuple(sorted(ele_temp[[i, (i+1)%3]]))
            edge_set.add(edge_temp)

    edge = np.array(list(edge_set))

    return edge


def fix_particle_No(L: float, W: float, seed_size: float):
    """
    Find the particle No. of fix constraint and grasping constraint
    """
    fix_flag = ti.field(dtype=ti.i32, shape=NV)
    grasp_flag = ti.field(dtype=ti.i32, shape=NV)

    @ti.kernel
    def cal_fix_constraint(L: float, W: float, seed_size: float):
        EPS = seed_size / 3
        # flag = np.array(PARTICLE_NUM, dtype=int)
        for idx in range(NV):
            x_temp = pos_init[idx].x
            z_temp = pos_init[idx].y        # 2D dimension
            # flag_temp = (x_temp > L - EPS or x_temp < 0. + EPS) and (z_temp > W/2 - EPS or z_temp < -W/2 + EPS)
            fix_flag_temp = (x_temp < 0. + EPS)
            grasp_flag_temp = (x_temp > L - EPS) and (z_temp > W/2 -EPS)
            fix_flag[idx] = fix_flag_temp
            grasp_flag[idx] = grasp_flag_temp

    cal_fix_constraint(L, W, seed_size)
    fix_particle_set = set()
    grasp_particle_set = set()
    for i in range(NV):
        if fix_flag[i]:
            fix_particle_set.add(i)
        if grasp_flag[i]:
            grasp_particle_set.add(i)
    fix_particle_list = list(fix_particle_set)
    grasp_particle_list = list(grasp_particle_set)

    grasp_idx = grasp_particle_list[0]
    grasp_ele_list = []
    for i in range(NF):
        ele_temp = f2v[i].to_numpy()
        if grasp_idx in ele_temp:
            grasp_ele_list.append(i)

    return fix_particle_list, grasp_particle_list, grasp_ele_list


@ti.kernel
def precomputation():
    dimp = dim+1
    for e_it in range(NF):
        ia, ib, ic = f2v[e_it]
        mass[ia] += volume/dimp * rho
        mass[ib] += volume/dimp * rho
        mass[ic] += volume/dimp * rho

    # Construct A_i matrix for every element / Build A for all the constraints:
    # Strain constraints and area constraints
    for t in ti.static(range(2)):
        for i in range(NF):
            # Get (Dm)^-1 for this element:
            Dm_inv_i = B[i]
            a = Dm_inv_i[0, 0]
            b = Dm_inv_i[0, 1]
            c = Dm_inv_i[1, 0]
            d = Dm_inv_i[1, 1]
            # Construct A_i:
            # - Why the dimension of A_i is 4*6?
            # - Because the dimension of X_f*X_g^{-1}=Aq is 2*2, and flatten to 4*1
            A[t*NF+i][0, 0] = -a-c
            A[t*NF+i][0, 2] = a
            A[t*NF+i][0, 4] = c
            A[t*NF+i][1, 0] = -b-d
            A[t*NF+i][1, 2] = b
            A[t*NF+i][1, 4] = d
            A[t*NF+i][2, 1] = -a-c
            A[t*NF+i][2, 3] = a
            A[t*NF+i][2, 5] = c
            A[t*NF+i][3, 1] = -b-d
            A[t*NF+i][3, 3] = b
            A[t*NF+i][3, 5] = d

    # Construct lhs matrix without constraints
    for i in range(NV):
        for d in ti.static(range(2)):
            lhs_matrix[i * dim + d, i * dim + d] += (drag / dt) + mass[i] / (dt * dt)

    # Add strain and area/volume constraints to the lhs matrix
    for t in ti.static(range(2)):
        for ele_idx in range(NF):
            A_i = A[t*NF+ele_idx]
            ia, ib, ic = f2v[ele_idx]
            ia_x_idx, ia_y_idx = ia*2, ia*2+1
            ib_x_idx, ib_y_idx = ib*2, ib*2+1
            ic_x_idx, ic_y_idx = ic*2, ic*2+1
            q_idx_vec = ti.Vector([ia_x_idx, ia_y_idx, ib_x_idx, ib_y_idx, ic_x_idx, ic_y_idx])
            # AT_A = A_i.transpose() @ A_i
            for A_row_idx in ti.static(range(6)):
                for A_col_idx in ti.static(range(6)):
                    lhs_row_idx = q_idx_vec[A_row_idx]
                    lhs_col_idx = q_idx_vec[A_col_idx]
                    for idx in ti.static(range(4)):
                        weight = 0.0
                        if t == 0:
                            weight = m_weight_strain
                        else:
                            weight = m_weight_volume
                        lhs_matrix[lhs_row_idx, lhs_col_idx] += (A_i[idx, A_row_idx] * A_i[idx, A_col_idx] * weight)

    # Add positional constraints to the lhs matrix
    # 位置约束将质量设置为无穷大（一个很大的数）
    for i in range(NV):
        if boundary_labels[i] == 1:
            q_i_x_idx = i * 2
            q_i_y_idx = i * 2 + 1
            lhs_matrix[q_i_x_idx, q_i_x_idx] += m_weight_positional  # This is the weight of positional constraints
            lhs_matrix[q_i_y_idx, q_i_y_idx] += m_weight_positional

    for i in ti.static(grasp_particle_list):
        q_i_x_idx = i * 2
        q_i_y_idx = i * 2 + 1
        lhs_matrix[q_i_x_idx, q_i_x_idx] += m_weight_positional
        lhs_matrix[q_i_y_idx, q_i_y_idx] += m_weight_positional


# NOTE: This function doesn't build all constraints
# It just builds strain constraints and area/volume constraints
@ti.kernel
def local_solve_build_bp_for_all_constraints():
    for i in range(NF):
        # Construct strain constraints:
        # Construct Current F_i:
        ia, ib, ic = f2v[i]
        a, b, c = pos_new[ia], pos_new[ib], pos_new[ic]
        D_i = ti.Matrix.cols([b - a, c - a])
        F_i = ti.cast(D_i @ B[i], ti.f64)
        F[i] = F_i

        if i == grasp_ele_list[0]:
            print('Deformation gradient F', F[i])

        if i == 300:
            print('Deformation gradient F--300', F[i])

        # Use current F_i construct current 'B * p' or Ri
        U, sigma, V = ti.svd(F_i, ti.f64)
        # 只有旋转量
        Bp[i] = U @ V.transpose()

        # Construct volume preservation constraints:
        x, y, max_it, tol = 10.0, 10.0, 80, 1e-6
        for t in range(max_it):
            aa, bb = x + sigma[0, 0], y + sigma[1, 1]
            f = aa * bb - 1
            g1, g2 = bb, aa
            bot = g1 * g1 + g2 * g2
            if abs(bot) < tol:
                break
            top = x * g1 + y * g2 - f
            div = top / bot
            x0, y0 = x, y
            x = div * g1
            y = div * g2
            _dx, _dy = x - x0, y - y0
            if _dx * _dx + _dy * _dy < tol * tol:
                break
        PP = ti.Matrix.rows([[x + sigma[0, 0], 0.0], [0.0, sigma[1, 1] + y]])
        Bp[NF + i] = U @ PP @ V.transpose()

    # Calculate Phi for all the elements:
    for i in range(NF):
        Bp_i_strain = Bp[i]
        Bp_i_volume = Bp[NF + i]
        F_i = F[i]
        energy1 = mu * volume * ((F_i - Bp_i_strain).norm() ** 2)
        energy2 = 0.5 * lam * volume * ((F_i - Bp_i_volume).trace() ** 2)
        phi[i] = energy1 + energy2


@ti.kernel
def build_sn():
    # 通过此刻的速度和加速度构建s_n
    for vert_idx in range(NV):  # number of vertices
        Sn_idx1 = vert_idx*2  # m_sn
        Sn_idx2 = vert_idx*2+1
        pos_i = pos[vert_idx]  # pos = m_x
        vel_i = vel[vert_idx]
        Sn[Sn_idx1] = pos_i[0] + dt * vel_i[0]  # x-direction;
        Sn[Sn_idx2] = pos_i[1] + dt * vel_i[1] + dt * dt * gravity[1]  # y-direction;

    for i in ti.static(grasp_particle_list):
        pos_i = pos[i]  # pos = m_x
        vel_i = vel[i]
        idx0 = i*2
        idx1 = i*2 + 1
        Sn[idx0] = pos_i[0] + dt * vel_i[0]  # x-direction;
        Sn[idx1] = pos_i[1] + dt * vel_i[1] + dt * dt * gravity[1]  # y-direction;
        print('Pos i', pos_i[0], pos_i[1])
        print('Vel i', vel_i[0], vel_i[1])
        print('Sn', Sn[idx0], Sn[idx1])


@ti.kernel
def build_rhs(rhs: ti.types.ndarray()):
    one_over_dt2 = 1.0 / (dt ** 2)
    # Construct the first part of the rhs
    for i in range(NV * 2):
        pos_i = pos[int(i/2)]
        p0 = pos_i[0]
        p1 = pos_i[1]
        if i % 2 == 0:
            rhs[i] = one_over_dt2 * mass[int(i/2)] * Sn[i] + (drag/dt*p0)  # 0.000061
        else:
            rhs[i] = one_over_dt2 * mass[int(i/2)] * Sn[i] + (drag/dt*p1)  # 0.000061
    # Add strain and volume/area constraints to the rhs
    for t in ti.static(range(2)):
        for ele_idx in range(NF):
            ia, ib, ic = f2v[ele_idx]
            Bp_i = Bp[t*NF+ele_idx]  # It is a 2x2 matrix now. We want it be a 4x1 vector.
            # 注意这里的Bp_i是一个2x2的矩阵，转换成4x1的向量时的顺序，于对A向量化的顺序是一致的
            Bp_i_vec = ti.Vector([Bp_i[0, 0], Bp_i[0, 1], Bp_i[1, 0], Bp_i[1, 1]])
            A_i = A[ele_idx]
            AT_Bp = A_i.transpose() @ Bp_i_vec  # AT_Bp is a 6x1 vector now.
            weight = 0.0
            if t == 0:
                weight = m_weight_strain
            else:
                weight = m_weight_volume
            AT_Bp *= weight  # m_weight_strain

            # Add AT_Bp back to rhs
            q_ia_x_idx = ia*2
            q_ia_y_idx = q_ia_x_idx+1
            rhs[q_ia_x_idx] += AT_Bp[0]
            rhs[q_ia_y_idx] += AT_Bp[1]

            q_ib_x_idx = ib*2
            q_ib_y_idx = q_ib_x_idx+1
            rhs[q_ib_x_idx] += AT_Bp[2]
            rhs[q_ib_y_idx] += AT_Bp[3]

            q_ic_x_idx = ic*2
            q_ic_y_idx = q_ic_x_idx+1
            rhs[q_ic_x_idx] += AT_Bp[4]
            rhs[q_ic_y_idx] += AT_Bp[5]

    # Add positional constraints Bp to the rhs
    # 位置约束将质量设置为无穷大（一个很大的数）
    for i in range(NV):
        if boundary_labels[i] == 1:
            pos_init_i = pos_init[i]
            q_i_x_idx = i * 2
            q_i_y_idx = i * 2 + 1
            rhs[q_i_x_idx] += (pos_init_i[0] * m_weight_positional)
            rhs[q_i_y_idx] += (pos_init_i[1] * m_weight_positional)

    for i in ti.static(grasp_particle_list):
        pos_new_i = pos_new[i]
        q_i_x_idx = i * 2
        q_i_y_idx = i * 2 + 1
        rhs[q_i_x_idx] += (pos_new_i[0] * m_weight_positional)
        rhs[q_i_y_idx] += (pos_new_i[1] * m_weight_positional)


@ti.kernel
def update_velocity_pos():
    for i in ti.static(grasp_particle_list):
        pos_new[i] =

    for i in range(NV):
        pos_latest[i] = pos[i]
        vel[i] = (pos_new[i] - pos[i]) / dt
        pos[i] = pos_new[i]    # time.sleep(20)

    # for i in ti.static(grasp_particle_list):
    #     print('Vel of grasp particle', vel[i])
    #     print('Pos new', pos_new[i])
    #     pos[i] = ti.Vector[0.102, 0.052]
    #     vel[i] = GRASP_VEL
    #     print('Pos', pos[i])


@ti.kernel
def warm_up():
    for pos_idx in range(NV):
        sn_idx1, sn_idx2 = pos_idx * 2, pos_idx * 2 + 1
        pos_new[pos_idx][0] = Sn[sn_idx1]
        pos_new[pos_idx][1] = Sn[sn_idx2]

    for i in ti.static(grasp_particle_list):
        print('Pos New', pos_new[i])


@ti.kernel
def initinfo():
    EPS = 0.1/20/2
    for i in range(NV):
        if (pos[i][0] > 0.1-EPS):
            vel[i][0] = 0
        elif (pos[i][0] < 0.1-EPS):
            vel[i][0] = 0
        else:
            vel[i][0] = 0

    # for i in ti.static(grasp_particle_list):
    #     pos[i] = ti.Vector([0.102, 0.052])


@ti.kernel
def update_pos_new_from_numpy(sol: ti.types.ndarray()):
    for pos_idx in range(NV):
        sol_idx1, sol_idx2 = pos_idx*2, pos_idx*2+1
        pos_new[pos_idx][0] = sol[sol_idx1]
        pos_new[pos_idx][1] = sol[sol_idx2]


@ti.kernel
def check_residual() -> ti.f32:
    residual = 0.0
    for i in range(NV):
        residual += (last_pos_new[i] - pos_new[i]).norm()
        last_pos_new[i] = pos_new[i]
    # print("residual:", residual)
    return residual


@ti.kernel
def compute_T1_energy() -> ti.f64:
    T1 = 0.0
    for i in range(NV):
        sn_idx1, sn_idx2 = i * 2, i * 2 + 1
        sn_i = ti.Vector([Sn[sn_idx1], Sn[sn_idx2]])
        temp_diff = (pos_new[i] - sn_i) * ti.sqrt(mass[i])
        T1 += (temp_diff[0]**2 + temp_diff[1]**2)
    return T1 / (2.0 * dt**2)


@ti.kernel
def global_compute_T2_energy() -> ti.f64:
    T2_global_energy = ti.cast(0.0, ti.f64)
    # Calculate the energy contributed by strain and volume/area constraints
    for i in range(NF):
        # Construct Current F_i
        ia, ib, ic = f2v[i]
        a, b, c = pos_new[ia], pos_new[ib], pos_new[ic]
        D_i = ti.Matrix.cols([b - a, c - a])
        F_i = ti.cast(D_i @ B[i], ti.f64)
        # Get current Bp
        Bp_i_strain = Bp[i]
        Bp_i_volume = Bp[NF + i]
        energy1 = m_weight_strain * ((F_i - Bp_i_strain).norm() ** 2) / ti.cast(2.0, ti.f64)
        energy2 = m_weight_volume * ((F_i - Bp_i_volume).norm() ** 2) / ti.cast(2.0, ti.f64)
        T2_global_energy += (energy1 + energy2)
    # Calculate the energy contributed by positional constraints
    # total_energy3 = 0.0
    for i in range(NV):
        if boundary_labels[i] == 1 or i == grasp_particle_list[0]:
            pos_init_i = pos_init[i]
            pos_curr_i = pos_new[i]
            energy3 = m_weight_positional * ((pos_curr_i - pos_init_i).norm() ** 2) / ti.cast(2.0, ti.f64)
            # total_energy3 += energy3
            T2_global_energy += energy3
    # print("global energy3:", total_energy3)
    return T2_global_energy


@ti.kernel
def local_compute_T2_energy() -> ti.f64:
    # Calculate T2 energy
    local_T2_energy = ti.cast(0.0, ti.f64)
    # Calculate the energy contributed by strain and volume/area constraints
    for e_it in range(NF):
        Bp_i_strain = Bp[e_it]
        Bp_i_volume = Bp[e_it + NF]
        F_i = F[e_it]
        energy1 = m_weight_strain * ((F_i - Bp_i_strain).norm() ** 2) / ti.cast(2.0, ti.f64)
        energy2 = m_weight_volume * ((F_i - Bp_i_volume).norm() ** 2) / ti.cast(2.0, ti.f64)
        local_T2_energy += (energy1 + energy2)
    # Calculate the energy contributed by positional constraints
    # total_energy3 = 0.0
    for i in range(NV):
        if boundary_labels[i] == 1:
            pos_init_i = pos_init[i]
            pos_curr_i = pos_new[i]
            energy3 = m_weight_positional * ((pos_curr_i - pos_init_i).norm() ** 2) / ti.cast(2.0, ti.f64)
            # total_energy3 += energy3
            local_T2_energy += energy3
    # print("local energy3:", total_energy3)
    return local_T2_energy


def compute_global_step_energy():
    # Calculate global T2 energy
    global_T2_energy = global_compute_T2_energy()
    # Calculate global T1 energy
    global_T1_energy = compute_T1_energy()
    return (global_T1_energy + global_T2_energy)


def compute_local_step_energy():
    local_T2_energy = local_compute_T2_energy()
    # Calculate T1 energy
    local_T1_energy = compute_T1_energy()
    return (local_T1_energy + local_T2_energy)


def paint_phi(canvas):
    pos_np = pos.to_numpy()
    phi_np = phi.to_numpy()
    f2v_np = f2v.to_numpy()
    a, b, c = pos_np[f2v_np[:, 0]], pos_np[f2v_np[:, 1]], pos_np[f2v_np[:, 2]]
    k = phi_np * (8000 / E)
    gb = (1 - k) * 0.7
    tri_idx.from_numpy(f2v_np)
    tri_per_color.from_numpy(np.stack([k+gb, gb, gb], axis=1))
    # print("gb:", gb[0])
    # print("phi_np", phi_np[0])
    # print("k", k[0])
    # canvas.triangles(pos, indices=tri_idx, per_vertex_color=tri_per_color)
    # canvas.lines(pos, width=0.001, indices=line_idx, color=(0.1, 0.1, 0.1))
    # canvas.lines(a, b, color=(0.1, 0.1, 0.1), radius=0.002)
    # canvas.lines(b, c, color=0xffffff, radius=0.002)
    # canvas.lines(c, a, color=0xffffff, radius=0.002)


frame_counter = 0
init_mesh()
init_pos()
edge_np = init_edge()
NE = edge_np.shape[0]
edge = ti.Vector.field(2, dtype=ti.i32, shape=NE)
edge.from_numpy(edge_np)
# print(edge.to_numpy())
fix_particle_list, grasp_particle_list, grasp_ele_list = fix_particle_No(0.1, 0.1, 0.1/20)
print('Particle number', NV)
print('Grasp particle index', grasp_particle_list)
print('Grasp element index', grasp_ele_list)

precomputation()
lhs_matrix_np = lhs_matrix.to_numpy()
s_lhs_matrix_np = sparse.csr_matrix(lhs_matrix_np)
pre_fact_lhs_solve = factorized(s_lhs_matrix_np)

print("sparse lhs matrix:\n", s_lhs_matrix_np)

initinfo()

"""-------------GGUI setting---------------"""
particle_test = ti.Vector.field(3, dtype=ti.f32, shape=2)
particle_test[0] = ti.Vector([0.2, 0., 0.1])
particle_test[1] = ti.Vector([0.1, 0., 0.1])
def gui_set(pos, target, FOV=60):
    # init the window, canvas, scene and camerea
    window = ti.ui.Window("Projective Dynamics", (1080, 720), vsync=True)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initialize camera position
    camera.position(pos[0], pos[1], pos[2])
    camera.lookat(target[0], target[1], target[2])
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    # 设置相机的向上轴的方向，在相机模型中是-Y轴
    camera.up(0., 0., -1.)
    camera.z_near(0.01)
    camera.fov(FOV)

    # set the camera, you can move around by pressing 'wasdeq'
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # set the light
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    # scene.point_light(pos=(0.01, 0, 3), color=(1., 1., 1.))
    scene.ambient_light((1., 1., 1.))
    return window, camera, scene


def gui_show(window, canvas, scene, SHOW_FLAG=True):
    """
    Show the GUI
    """
    if SHOW_FLAG is False:
        return
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    scene.ambient_light((0.8, 0.8, 0.8))
    # the conversion of object particles, etc. the ggui of the taichi only support float32
    particle_show.from_numpy(np.insert(pos.to_numpy(dtype=np.float32), 1, np.zeros(NV), axis=1))

    # particle_test = ti.Vector.field(3, dtype=ti.f32, shape=1)
    # particle_test[0] = ti.Vector([0.0, 0., -0.0])

    # scene.mesh(particle_show, indices=surf_show, color=(1, 1, 0))
    scene.particles(particle_show, radius=0.001, color=(0., 0., 0.))
    scene.lines(particle_show, width=0.9, indices=edge, color=(0. ,0. ,0.))
    # scene.particles(particle_test, radius=0.005, color=(0., 1., 0.))
    canvas.scene(scene)
    canvas.set_background_color((1.0, 1.0, 1.0))
    # if particle_pos[399].x > 0.14:
    #     window.save_image(f'Figure/{global_E}.png')
    #     exit(0)
    window.show()

# frame_counter = 0
sim_t = 0.0
plot_array = []

window, camera, scene = gui_set(pos=[0.1, 0.3, 0.], target=[0.1, 0., 0.])
canvas = window.get_canvas()

while window.running:
    gui_show(window, canvas, scene, SHOW_FLAG=True)

    # for i in ti.static(grasp_particle_list):
    #     pos[i] = ti.Vector([0.102, 0.052])

    build_sn()
    # Warm up:
    warm_up()
    # print("Frame ", frame_counter)
    last_record_energy = 1000000.0
    for itr in range(solver_max_iteration):

        # start_solve_constraints_time = time.perf_counter_ns()
        local_solve_build_bp_for_all_constraints()
        # end_solve_constraints_time = time.perf_counter_ns()
        # print("solve constraints time elapsed:", end_solve_constraints_time - start_solve_constraints_time)

        # start_build_rhs_time = time.perf_counter_ns()
        build_rhs(rhs_np)
        # end_build_rhs_time = time.perf_counter_ns()
        # print("build rhs time elapsed:", end_build_rhs_time - start_build_rhs_time)

        local_step_energy = compute_local_step_energy()
        print("energy after local step:", local_step_energy)
        if local_step_energy > last_record_energy:
            print("Energy Error: LOCAL; Error Amount:", (local_step_energy - last_record_energy) / local_step_energy)
            if (local_step_energy - last_record_energy) / local_step_energy > 0.01:
                print("Large Error: LOCAL")
        last_record_energy = local_step_energy

        # start_linear_solve_time = time.perf_counter_ns()
        pos_new_np = pre_fact_lhs_solve(rhs_np)
        # end_linear_solve_time = time.perf_counter_ns()
        # print("linear solve time elapsed:", end_linear_solve_time - start_linear_solve_time)

        # start_update_pos_time = time.perf_counter_ns()
        update_pos_new_from_numpy(pos_new_np)
        # end_update_pos_time = time.perf_counter_ns()
        # print("update pos new elapsed:", end_update_pos_time - start_update_pos_time)

        global_step_energy = compute_global_step_energy()
        print("energy after global step:", global_step_energy)
        plot_array.append([itr, global_step_energy])
        if global_step_energy > last_record_energy:
            print("Energy Error: GLOBAL; Error Amount:", (global_step_energy - last_record_energy) / global_step_energy)
            if (global_step_energy - last_record_energy) / global_step_energy > 0.01:
                print("Large Error: GLOBAL")
        last_record_energy = global_step_energy

        # start_check_residual_time = time.perf_counter_ns()
        # residual = check_residual()
        # end_check_residual_time = time.perf_counter_ns()
        # print("check residual elapsed:", end_check_residual_time - start_check_residual_time)

        # check_boundary_points()
        # if residual < solver_stop_residual:
        #   break

    # Update velocity and positions
    update_velocity_pos()
    gui_show(window, canvas, scene, SHOW_FLAG=True)
    # paint_phi(canvas)
    # canvas.circles(pos, radius=0.001, color=(0.5, 0.5, 0.5))
    # frame_counter += 1
    # filename = f'FigureDemo/frame_{frame_counter:05d}.png'
    # window.show()
    # print("\n")