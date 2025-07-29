""" 实现2d曲面的diffpd, 用于Sofa仿真环境
created by hsy on 2025-07-22
"""
from pathlib import Path
import time
from typing import List, Dict, DefaultDict, Tuple
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
from scipy.sparse import linalg as spla
import taichi as ti
import meshtaichi_patcher as Patcher
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

from Utilize.GenMsh import mesh_obj_tri, write_obj, read_mshv2_triangle, write_mshv2_tri
from Utilize.GuiTaichi import gui_set
from Utilize.MathTaichi import svd_3x2_new, cotangent_ti
from Utilize.MathNp import compress_vectors
dir_path = Path(__file__).parent    # 设置工作目录为当前脚本所在目录

def read_msh(file_path):
    nodes, faces = read_mshv2_triangle(file_path)
    edges = np.vstack((faces[:, [0, 1]],
                       faces[:, [1, 2]],
                       faces[:, [2, 0]]))

    # 对每条边的顶点索引进行排序，以确保唯一性
    edges.sort(axis=1)
    unique_edges = np.unique(edges, axis=0) # 使用 np.unique 找到唯一的边

    return nodes, unique_edges, faces

@ti.data_oriented
class SoftBend2D:
    def __init__(self, shape, fix, contact, E:float, nu:float, dt:float, density:float, g=-9.8):
        self.shape = shape
        # 是否传入的是 .msh 文件(已划分网格)
        if isinstance(self.shape, str):
            node_np, edge_np, ele_np = read_msh(self.shape)
        else:
            node_np, edge_np, ele_np = mesh_obj_tri(self.shape, 0.02)
            node_np = np.hstack((node_np, np.zeros((node_np.shape[0], 1))))         # di: N*3
            np.savetxt(f"{dir_path}/Data/node_np.csv", node_np, fmt='%f', delimiter=",")
            np.savetxt(f"{dir_path}/Data/edge_np.csv", edge_np, fmt='%d', delimiter=",")
            np.savetxt(f"{dir_path}/Data/ele_np.csv", ele_np, fmt='%d', delimiter=",")

        obj_file:str = f"{dir_path}/Mesh/plane.obj"
        write_obj(obj_file, node_np, ele_np)
        write_mshv2_tri(f"{dir_path}/Mesh/plane.msh", node_np, ele_np)

        self.damp_alpha2 = 3.e-3  # Laplacian Damping for Projective Dynamics
        self.bend_weight = 2.e1
        self.positional_weight = 0.         # define later

        self.solve_itr:int = 10
        self.strain_lim_rate:float = 0.1            # Strain limit rate
        self.E, self.nu, self.dt, self.density, self.g = E, nu, dt, density, g
        self.dim = 3
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        self.PARTICLE_N = node_np.shape[0]
        self.EDGE_N = edge_np.shape[0]
        self.ELEMENT_N = ele_np.shape[0]

        self.mesh = Patcher.load_mesh(obj_file, relations=["VV", "VE", "VF", "EV", "EF", "FV", "FE"])

        self.mesh.verts.place({
            "pos": ti.types.vector(3, ti.f64),
            "pos_init": ti.types.vector(3, ti.f64),
            "pos_new": ti.types.vector(3, ti.f64),
            "vel": ti.types.vector(3, ti.f64),
            "mass": ti.f64,
        }, reorder=False)
        self.mesh.edges.place({
            "v_g": ti.types.matrix(1, 3, ti.f64),   # mean curvature vector (rest)
            "v_f": ti.types.matrix(1, 3, ti.f64),   # mean curvature vector
            "n": ti.f64,                            # edge curvature value
            "neigh_verts": ti.types.vector(4, ti.i32),  # 邻接顶点
            "cot_w": ti.types.matrix(1, 4, ti.f64),  # cotangent weights
            "bend_weight": ti.f64,
            "voronoi": ti.f64,
            "voronoi_sqrt": ti.f64,
            "border": bool,     # 是否为边界边
        }, reorder=False)
        self.mesh.faces.place({
            "area": ti.f64,
            "stretch_w": ti.f64,
            "stretch_lim_w": ti.f64,
        }, reorder=False)

        # 直接从obj文件读会丢精度
        # self.mesh.verts.pos_init.from_numpy(self.mesh.get_position_as_numpy().astype(np.float64))
        self.mesh.verts.pos_init.from_numpy(node_np.astype(np.float64))
        self.mesh.verts.pos.copy_from(self.mesh.verts.pos_init)

        self.edge = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)     # edge connectivity
        self.ele  = ti.Vector.field(3, dtype=ti.i32, shape=self.ELEMENT_N)  # element connectivity
        self.mass_sum = ti.field(dtype=ti.f64, shape=())

        self.Xg_inv         = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N)         # rest configuration
        self.F              = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N)              # deformation gradient
        self.F_A            = ti.Matrix.field(2, 3, dtype=ti.f64, shape=self.ELEMENT_N)            # deformation gradient linearisation coefficient matrix
        self.ele_u          = ti.Matrix.field(3, 3, dtype=ti.f64, shape=self.ELEMENT_N) # svd U
        self.ele_v          = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.ELEMENT_N) # svd V
        self.stretch_stress = ti.Vector.field(2, dtype=ti.f64, shape=self.ELEMENT_N)    # stretch stress
        self.Bp_stretch     = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N) # stretch part
        self.Bp_stretch_lim = ti.Matrix.field(3, 2, dtype=ti.f64, shape=self.ELEMENT_N) # strain-limit part

        self.stretch_energy     = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)
        self.stretch_lim_energy = ti.field(dtype=ti.f64, shape=self.ELEMENT_N)

        self.cot_w       = ti.Matrix.field(1, 4, dtype=ti.f64, shape=self.EDGE_N)   # cotangent weights
        self.Bp_bend     = ti.Matrix.field(1, 3, dtype=ti.f64, shape=self.EDGE_N)
        self.bend_energy = ti.field(dtype=ti.f64, shape=self.EDGE_N)
        self.v_f         = ti.Matrix.field(1, 3, dtype=ti.f64, shape=self.EDGE_N)   # rest curvature vector

        self.sn              = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.lhs             = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_stretch     = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_stretch_lim = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.lhs_bend_ti     = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))
        self.damping         = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N, self.PARTICLE_N))  # damping term
        self.rhs             = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.rhs_stretch     = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.rhs_bend        = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)

        self.dBp_stretch = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.PARTICLE_N*3))
        self.dBp_bend    = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.PARTICLE_N*3))
        self.g_hessian   = ti.field(dtype=ti.f64, shape=(self.PARTICLE_N*3, self.PARTICLE_N*3)) # deformation energy hessian
        self.dA          = None      # dim: 3N*3N, 用于初始化
        self.dx_const    = ti.field(ti.f64, shape=self.dim*self.PARTICLE_N)    # M/h2 + velocity constraint weights
        self.dL_dq_y     = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.z           = ti.field(dtype=ti.f64, shape=self.PARTICLE_N*3)
        self.dL_dq_y.fill(0.)
        self.z.fill(0.)

        self.pre_fact_lhs_solve = None

        # self.fix_particle_list = list(range(6)) # + [30]
        # self.contact_particle_list = [30] + [35]
        self.fix_particle_list = fix
        self.contact_particle_list = contact
        self.FIX_N = len(self.fix_particle_list)
        self.CON_N = len(self.contact_particle_list)
        self.fix_particle_ti = ti.field(dtype=ti.i32, shape=self.FIX_N)
        self.contact_particle_ti = ti.field(dtype=ti.i32, shape=self.CON_N)
        self.contact_vel = ti.Vector.field(3, dtype=ti.f64, shape=self.CON_N)
        self.fix_particle_ti.from_numpy(np.array(self.fix_particle_list).astype(np.int32))
        self.contact_particle_ti.from_numpy(np.array(self.contact_particle_list).astype(np.int32))
        self.contact_vel.fill(0.)

        self.marker_list = [11, 17, 23, 29]
        self.marker_N = len(self.marker_list)
        self.marker_ti = ti.field(dtype=ti.i32, shape=self.marker_N)
        self.marker_pos_desired = ti.Vector.field(3, dtype=ti.f64, shape=(self.marker_N))
        self.error = ti.Vector.field(3, dtype=ti.f64, shape=(self.marker_N))
        self.marker_ti.from_numpy(np.array(self.marker_list).astype(np.int32))

        self.construct_mass()
        self.construct_cotangent()
        self.construct_Xg_inv()
        self.positional_weight = 1.e3 * self.mass_sum[None] / self.PARTICLE_N / self.dt**2
        self.construct_dx_const()

        print(f"Particle numer: {self.PARTICLE_N}; Edge number: {self.EDGE_N}; Element number: {self.ELEMENT_N}")
        print(f"Positional weight: {self.positional_weight}")
        print(f"stretch weight: {self.mesh.faces.stretch_w[0]}")

    @ti.kernel
    def construct_mass(self):
        ti.mesh_local(self.mesh.verts.mass, self.mesh.verts.pos_init, self.mesh.edges.voronoi,
                      self.mesh.faces.area, self.mesh.faces.stretch_w, self.mesh.faces.stretch_lim_w)
        for f in self.mesh.faces:
            v1, v2, v3 = f.verts[0], f.verts[1], f.verts[2]
            e1, e2, e3 = f.edges[0], f.edges[1], f.edges[2]
            faces_area = 0.5 * ((v2.pos_init - v1.pos_init).cross(v3.pos_init - v1.pos_init)).norm()

            v1.mass += self.density * faces_area / 3.
            v2.mass += self.density * faces_area / 3.
            v3.mass += self.density * faces_area / 3.
            e1.voronoi += faces_area / 3.
            e2.voronoi += faces_area / 3.
            e3.voronoi += faces_area / 3.
            f.area = faces_area

            f.stretch_w = 2 * self.mu * faces_area
            f.stretch_lim_w = 0.e3 * f.stretch_w
            self.ele[f.id] = ti.Vector([v1.id, v2.id, v3.id])

        ti.mesh_local(self.mesh.edges.voronoi, self.mesh.edges.voronoi_sqrt)
        for e in self.mesh.edges:
            if e.faces.size > 1:
                e.voronoi_sqrt = ti.sqrt(e.voronoi)
            else:
                e.voronoi_sqrt = 0.
            
            v1, v2 = e.verts[0], e.verts[1]
            self.edge[e.id] = ti.Vector([v1.id, v2.id])

        ti.mesh_local(self.mesh.verts.mass)
        for v in self.mesh.verts:
            self.mass_sum[None] += v.mass

    @ti.kernel
    def construct_cotangent(self):
        ti.mesh_local(self.mesh.edges.bend_weight, self.mesh.edges.voronoi, self.mesh.edges.border)
        for e in self.mesh.edges:
            if e.faces.size > 1:
                e.bend_weight = self.bend_weight # * e.voronoi, no need to multiply by voronoi, because bend_weight is already normalized
                v1, v2 = e.verts[0], e.verts[1]
                tri1, tri2 = e.faces[0], e.faces[1]

                m, n = 0, 0
                for i in range(3):
                    if tri1.verts[i].id != v1.id and tri1.verts[i].id != v2.id:
                        m = i
                    if tri2.verts[i].id != v1.id and tri2.verts[i].id != v2.id:
                        n = i
                v3_1, v3_2 = tri1.verts[m], tri2.verts[n]

                cot01 = cotangent_ti(v2.pos_init - v1.pos_init, v3_1.pos_init - v1.pos_init)
                cot02 = cotangent_ti(v2.pos_init - v1.pos_init, v3_2.pos_init - v1.pos_init)
                cot03 = cotangent_ti(v1.pos_init - v2.pos_init, v3_1.pos_init - v2.pos_init)
                cot04 = cotangent_ti(v1.pos_init - v2.pos_init, v3_2.pos_init - v2.pos_init)

                e.cot_w = ti.Matrix([[cot03+cot04, cot01+cot02, -cot01-cot03, -cot02-cot04]], dt=ti.f64)
                e.neigh_verts = ti.Vector([v1.id, v2.id, v3_1.id, v3_2.id])
                e.v_g = e.cot_w @ ti.Matrix.rows([v1.pos_init, v2.pos_init, v3_1.pos_init, v3_2.pos_init]) / e.voronoi_sqrt
                e.n = e.v_g.norm()
            else:
                e.bend_weight = 0.
                e.border = True

    @ti.kernel
    def construct_Xg_inv(self):
        for f in self.mesh.faces:
            v1, v2, v3 = f.verts[0], f.verts[1], f.verts[2]
            a = ti.Vector([v1.pos_init.x, v1.pos_init.y])
            b = ti.Vector([v2.pos_init.x, v2.pos_init.y])
            c = ti.Vector([v3.pos_init.x, v3.pos_init.y])
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.Xg_inv[f.id] = B_i_inv.inverse()

    @ti.kernel
    def construct_lhs_mass(self):
        ti.mesh_local(self.mesh.verts.mass)
        for v in self.mesh.verts:
            self.lhs[v.id, v.id] += v.mass / self.dt**2

    @ti.kernel
    def construct_lhs_stretch(self):
        # https://medium.com/@victorlouisdg/jax-cloth-tutorial-part-1-e7a0e285864f
        for f_i in range(self.ELEMENT_N):
            Xg_inv = self.Xg_inv[f_i]
            a, b, c, d = Xg_inv[0, 0], Xg_inv[0, 1], Xg_inv[1, 0], Xg_inv[1, 1]

            # F's dim=6*9，flatten(F)按照列优先
            self.F_A[f_i][0, 0] = -a - c
            self.F_A[f_i][0, 1] = a
            self.F_A[f_i][0, 2] = c
            self.F_A[f_i][1, 0] = -b - d
            self.F_A[f_i][1, 1] = b
            self.F_A[f_i][1, 2] = d

        ti.mesh_local(self.mesh.faces.stretch_w, self.mesh.faces.stretch_lim_w)
        for f in self.mesh.faces:
            idx1, idx2, idx3 = f.verts[0].id, f.verts[1].id, f.verts[2].id
            q_idx_vec = ti.Vector([idx1, idx2, idx3])
            F_A = self.F_A[f.id]
            # print(f.id, "F_A:\n", F_A)
            ATA = F_A.transpose() @ F_A
            # print(f.id, "ATA:\n", ATA)

            stretch_weight = f.stretch_w
            stretch_lim_weight = f.stretch_lim_w
            # print(f.id, "Stretch weight:", stretch_weight)
            for A_row_idx, A_col_idx in ti.ndrange(3, 3):
                lhs_row_idx, lhs_col_idx = q_idx_vec[A_row_idx], q_idx_vec[A_col_idx]
                self.lhs[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]
                self.lhs_stretch[lhs_row_idx, lhs_col_idx] += stretch_weight * ATA[A_row_idx, A_col_idx]

                # Strain-limit
                self.lhs[lhs_row_idx, lhs_col_idx] += stretch_lim_weight * ATA[A_row_idx, A_col_idx]
                self.lhs_stretch_lim[lhs_row_idx, lhs_col_idx] += stretch_lim_weight * ATA[A_row_idx, A_col_idx]

    @ti.kernel
    def construct_lhs_bend(self):
        """ v_g: corresponding to normal curvature vector of dscrete edges (= curvature H*3)
        """
        ti.mesh_local(self.mesh.verts.pos_init, self.mesh.edges.voronoi_sqrt, self.mesh.edges.bend_weight,
                      self.mesh.edges.neigh_verts, self.mesh.edges.cot_w)
        for e in self.mesh.edges:
            if e.border == False:
                v1, v2 = e.verts[0], e.verts[1]
                v1_id, v2_id, v3_1_id, v3_2_id = e.neigh_verts[0], e.neigh_verts[1], e.neigh_verts[2], e.neigh_verts[3]

                A = e.cot_w
                ATA = A.transpose() @ A

                q_idx_vec = ti.Vector([v1_id, v2_id, v3_1_id, v3_2_id])
                for row_idx, col_idx in ti.ndrange(4, 4):
                    lhs_row_idx, lhs_col_idx = q_idx_vec[row_idx], q_idx_vec[col_idx]
                    self.lhs[lhs_row_idx, lhs_col_idx] += e.bend_weight * ATA[row_idx, col_idx]
                    self.lhs_bend_ti[lhs_row_idx, lhs_col_idx] += e.bend_weight * ATA[row_idx, col_idx]

    @ti.kernel
    def construct_lhs_positional(self):
        for i in range(self.FIX_N):
            q_i = self.fix_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight

        for i in range(self.CON_N):
            q_i = self.contact_particle_ti[i]
            self.lhs[q_i, q_i] += self.positional_weight

    @ti.kernel
    def construct_lhs_damp(self):
        """ Construct damping matrix for the system.
        """
        for i, j in self.damping:
            self.damping[i, j] += self.damp_alpha2 * (
                self.lhs_stretch[i, j] + self.lhs_stretch_lim[i, j] + self.lhs_bend_ti[i, j])
        # for i in range(self.FIX_N):
        #     q_i = self.fix_particle_ti[i]
        #     self.damping[q_i, q_i] += self.positional_weight * self.damp_alpha2
        # for i in range(self.CON_N):
        #     q_i = self.contact_particle_ti[i]
        #     self.damping[q_i, q_i] += self.positional_weight * self.damp_alpha2

        for i, j in self.damping:
            self.lhs[i, j] += self.damping[i, j] / self.dt

    def precomputation(self):
        self.construct_lhs_mass()
        self.construct_lhs_stretch()
        self.construct_lhs_bend()
        self.construct_lhs_positional()
        self.construct_lhs_damp()

    @ti.kernel
    def construct_sn(self):
        dim = self.dim
        dt = self.dt
        ti.mesh_local(self.mesh.verts.pos, self.mesh.verts.vel)
        for v in self.mesh.verts:
            idx1, idx2, idx3 = dim*v.id, dim*v.id+1, dim*v.id+2
            self.sn[idx1] = v.pos.x + v.vel.x * dt
            self.sn[idx2] = v.pos.y + v.vel.y * dt
            self.sn[idx3] = v.pos.z + v.vel.z * dt + self.g * dt**2

        # Contact particles update
        for idx in range(self.CON_N):
            q_id = self.contact_particle_ti[idx]
            q_pos = self.mesh.verts.pos[q_id]
            self.sn[q_id*3]     = q_pos.x + self.contact_vel[idx].x * dt
            self.sn[q_id*3 + 1] = q_pos.y + self.contact_vel[idx].y * dt
            self.sn[q_id*3 + 2] = q_pos.z + self.contact_vel[idx].z * dt + self.g * dt**2

    @ti.kernel
    def warm_start(self):
        ti.mesh_local(self.mesh.verts.pos_new)
        for v in self.mesh.verts:
            v.pos_new.x = self.sn[v.id*3]
            v.pos_new.y = self.sn[v.id*3 + 1]
            v.pos_new.z = self.sn[v.id*3 + 2]

    @ti.kernel
    def construct_rhs_mass(self):
        ti.mesh_local(self.mesh.verts.mass)
        for v in self.mesh.verts:
            idx1, idx2, idx3 = v.id*3, v.id*3+1, v.id*3+2
            self.rhs[idx1] += v.mass * self.sn[idx1] / self.dt**2
            self.rhs[idx2] += v.mass * self.sn[idx2] / self.dt**2
            self.rhs[idx3] += v.mass * self.sn[idx3] / self.dt**2

    @ti.kernel
    def construct_rhs_stretch(self):
        ti.mesh_local(self.mesh.verts.pos_new, self.mesh.faces.stretch_w, self.mesh.faces.stretch_lim_w)
        for f in self.mesh.faces:
            v1, v2, v3 = f.verts[0], f.verts[1], f.verts[2]
            a, b, c = v1.pos_new, v2.pos_new, v3.pos_new
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f.id], ti.f64)
            self.F[f.id] = F_i
            # print(f"{f.id} F_i:{F_i:e}")

            U, sig, V = svd_3x2_new(F_i)
            self.ele_u[f.id], self.ele_v[f.id], self.stretch_stress[f.id] = U, V, sig
            self.Bp_stretch[f.id] = U @ ti.Matrix([[1., 0], [0., 1], [0, 0]], ti.f64) @ V.transpose()
            self.stretch_energy[f.id] = 0.5 * f.stretch_w * ((sig[0]-1.)**2 + (sig[1]-1.)**2)

            s_lim = ti.Vector.zero(ti.f64, 2)
            for dim in range(2):
                if sig[dim] > (1 + self.strain_lim_rate):
                    s_lim[dim] = 1 + self.strain_lim_rate
                elif sig[dim] < (1 - self.strain_lim_rate):
                    s_lim[dim] = 1 - self.strain_lim_rate
                else:
                    s_lim[dim] = sig[dim]
            self.Bp_stretch_lim[f.id] = U @ ti.Matrix([[s_lim[0], 0], [0., s_lim[1]], [0, 0]], ti.f64) @ V.transpose()
            self.stretch_lim_energy[f.id] = 0.5 * f.stretch_lim_w * ((sig[0]-s_lim[0])**2 + (sig[1]-s_lim[1])**2)

        ti.mesh_local(self.mesh.faces.stretch_w, self.mesh.faces.stretch_lim_w)
        for f in self.mesh.faces:
            Bp_stretch_i = self.Bp_stretch[f.id]
            Bp_stretch_lim_i = self.Bp_stretch_lim[f.id]
            F_AT = self.F_A[f.id].transpose()

            # Bp_stretch_i做transpose，因为AT需要与Bp的x，y，z分别矩阵乘法
            F_ATBp = F_AT @ Bp_stretch_i.transpose() * f.stretch_w
            F_ATBp_lim = F_AT @ Bp_stretch_lim_i.transpose() * f.stretch_lim_w

            for q_i, dim_idx in ti.ndrange(3, 3):
                q_idx = f.verts[q_i].id
                self.rhs[q_idx*3+dim_idx] += F_ATBp[q_i, dim_idx]
                self.rhs[q_idx*3+dim_idx] += F_ATBp_lim[q_i, dim_idx]
                # self.rhs_stretch[q_idx*3+dim_idx] += F_ATBp_lim[q_i, dim_idx]

    @ti.kernel
    def construct_rhs_bend(self):
        ti.mesh_local(self.mesh.edges.v_g, self.mesh.edges.v_f, self.mesh.edges.n, 
                      self.mesh.edges.bend_weight, self.mesh.edges.voronoi_sqrt,
                      self.mesh.edges.neigh_verts, self.mesh.edges.cot_w)
        for e in self.mesh.edges:
            if e.border == False:
                v_f_old = e.v_f
                v1, v2 = e.verts[0], e.verts[1]
                v1_id, v2_id, v3_1_id, v3_2_id = e.neigh_verts[0], e.neigh_verts[1], e.neigh_verts[2], e.neigh_verts[3]
                v3_1_pos, v3_2_pos = self.mesh.verts.pos_new[v3_1_id], self.mesh.verts.pos_new[v3_2_id]

                v_f = e.cot_w @ ti.Matrix.rows([v1.pos_new, v2.pos_new, v3_1_pos, v3_2_pos]) / e.voronoi_sqrt # dim: 3*1
                e.v_f = v_f

                if v_f.norm() > 0.:
                    self.Bp_bend[e.id] = e.n * v_f / v_f.norm()
                else:
                    normal = ti.Matrix.rows([(v1.pos_new - v2.pos_new).cross(v3_1_pos - v2.pos_new).normalized()])
                    if v_f_old.norm() > 0.:
                        self.Bp_bend[e.id] = e.n * v_f_old / v_f_old.norm()
                    else:
                        self.Bp_bend[e.id] = e.n * normal
                    e.v_f = v_f_old
                # print(f"{e.id} Bp_bend:\n", self.Bp_bend[e.id])

                self.bend_energy[e.id] = 0.5 * e.bend_weight * (v_f - self.Bp_bend[e.id]).norm()**2

                ATBp = e.cot_w.transpose() @ self.Bp_bend[e.id] * e.bend_weight

                q_idx_vec = ti.Vector([v1_id, v2_id, v3_1_id, v3_2_id])
                for q_i, dim_idx in ti.ndrange(4, 3):
                    q_idx = q_idx_vec[q_i]
                    self.rhs[q_idx*3+dim_idx] += ATBp[q_i, dim_idx]
                    # self.rhs_bend[q_idx*3+dim_idx] += ATBp[q_i, dim_idx]

    @ti.kernel
    def construct_rhs_positional(self):
        for q_i in range(self.FIX_N):
            weight = self.positional_weight
            q_idx = self.fix_particle_ti[q_i]
            q_pos_init = self.mesh.verts.pos_init[q_idx]
            q_i_x, q_i_y, q_i_z = q_idx*3, q_idx*3+1, q_idx*3+2
            self.rhs[q_i_x] += weight * q_pos_init.x
            self.rhs[q_i_y] += weight * q_pos_init.y
            self.rhs[q_i_z] += weight * q_pos_init.z

        for i in range(self.CON_N):
            q_idx = self.contact_particle_ti[i]
            q_pos = self.mesh.verts.pos[q_idx]
            self.rhs[q_idx*3]   += self.positional_weight * (q_pos.x + self.contact_vel[i].x * self.dt)
            self.rhs[q_idx*3+1] += self.positional_weight * (q_pos.y + self.contact_vel[i].y * self.dt)
            self.rhs[q_idx*3+2] += self.positional_weight * (q_pos.z + self.contact_vel[i].z * self.dt)

    @ti.kernel
    def construct_rhs_damp(self):
        for i in range(self.PARTICLE_N):
            for j in range(self.PARTICLE_N):
                self.rhs[3*i]   += self.damping[i, j] * self.mesh.verts.pos[j].x / self.dt
                self.rhs[3*i+1] += self.damping[i, j] * self.mesh.verts.pos[j].y / self.dt
                self.rhs[3*i+2] += self.damping[i, j] * self.mesh.verts.pos[j].z / self.dt

    def local_solve(self):
        self.rhs.fill(0.)
        self.rhs_stretch.fill(0.)
        self.rhs_bend.fill(0.)
        self.construct_rhs_mass()
        self.construct_rhs_stretch()
        self.construct_rhs_bend()
        self.construct_rhs_positional()
        self.construct_rhs_damp()

    @ti.kernel
    def update_pos_new(self, sol_x:ti.types.ndarray(), sol_y:ti.types.ndarray(), sol_z:ti.types.ndarray()):
        ti.mesh_local(self.mesh.verts.pos_new)
        for v in self.mesh.verts:
            v.pos_new.x = sol_x[v.id]
            v.pos_new.y = sol_y[v.id]
            v.pos_new.z = sol_z[v.id]

    @ti.kernel
    def update_vel_pos(self):
        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.mesh.verts.pos_new[q_idx] = self.mesh.verts.pos[q_idx] + self.contact_vel[idx] * self.dt

        ti.mesh_local(self.mesh.verts.pos, self.mesh.verts.pos_new, self.mesh.verts.vel)
        for v in self.mesh.verts:
            # if v.pos_new.z < 0.:    # 设置底板支撑
            #     v.pos_new.z = 0.
            v.vel = (v.pos_new - v.pos) / self.dt
            v.pos = v.pos_new

        for i in range(self.FIX_N):
            q_idx = self.fix_particle_ti[i]
            self.mesh.verts.pos[q_idx] = self.mesh.verts.pos_init[q_idx]
            self.mesh.verts.vel[q_idx] = ti.Vector([0., 0., 0.], dt=ti.f64)

        for idx in range(self.CON_N):
            q_idx = self.contact_particle_ti[idx]
            self.mesh.verts.vel[q_idx] = ti.Vector([0., 0., 0.], dt=ti.f64)

    @ti.kernel
    def hessian_stretch(self):
        self.dBp_stretch.fill(0.)
        for i in range(self.ELEMENT_N):
            F_Ai = self.F_A[i]
            U, sig, V = self.ele_u[i], self.stretch_stress[i], self.ele_v[i]

            dBp_dF = ti.Matrix.zero(ti.f64, 6, 6)
            for m in range(3):
                for n in range(2):
                    Omega_uv = ti.Matrix.zero(ti.f64, 3, 2)
                    Omega_uv[0, 1] = (U[m,0]*V[n,1] - U[m,1]*V[n,0]) / (sig[0] + sig[1])
                    Omega_uv[1, 0] = -Omega_uv[0, 1]
                    Omega_uv[2, 0] = U[m,2]*V[n,0] / sig[0]
                    Omega_uv[2, 1] = U[m,2]*V[n,1] / sig[1]
                    dBp_df = U @ Omega_uv @ V.transpose()
                    dBp_dF[2*m+n, :] = ti.Vector([dBp_df[0, 0], dBp_df[0, 1], dBp_df[1, 0], dBp_df[1, 1], dBp_df[2, 0], dBp_df[2, 1]])

            idx1, idx2, idx3 = self.ele[i]
            # print(i, "dBp_dF_i:", dBp_dF)
            # print("Stretch weight:", self.stretch_weight[i])
            for m, n in ti.ndrange(3, 3):       # 3*3表示维数, 
                dBp_dF_i = ti.Matrix.zero(ti.f64, 2, 2)
                for k, l in ti.ndrange(2, 2):
                    dBp_dF_i[k, l] = dBp_dF[2*m+k, 2*n+l]
                AT_dBp_dq_i = F_Ai.transpose() @ dBp_dF_i @ F_Ai    # A.T @ (dBp_x / dF_x) @ A
                AT_dBp_dq_i = AT_dBp_dq_i * self.mesh.faces.stretch_w[i]

                row_idx_vec = ti.Vector([idx1*3+m, idx2*3+m, idx3*3+m])
                col_idx_vec = ti.Vector([idx1*3+n, idx2*3+n, idx3*3+n])
                for k, l in ti.ndrange(3, 3):       # AT_dBp_dq_i's dim
                    row_idx = row_idx_vec[k]
                    col_idx = col_idx_vec[l]
                    self.dBp_stretch[row_idx, col_idx] += AT_dBp_dq_i[k, l]

    @ti.kernel
    def hessian_bend(self):
        # since the rest curvature is zero, the dBp/dq is constant
        self.dBp_bend.fill(0.)

    def construct_g_hessian(self):
        self.hessian_stretch()
        self.hessian_bend()
        self.dA = self.dBp_stretch.to_numpy() + self.dBp_bend.to_numpy()
        # np.savetxt("Data/dBp_stretch.csv", self.dBp_stretch.to_numpy(), delimiter=",", fmt="%.8e")
        # np.savetxt("Data/dBp_bend.csv", self.dBp_bend.to_numpy(), delimiter=",", fmt="%.8e")

    @ti.kernel
    def construct_dx_const(self):
        """解决Movement Constraint中dBp/dq为常数的情况
        """
        ti.mesh_local(self.mesh.verts.mass)
        for v in self.mesh.verts:
            self.dx_const[v.id*self.dim]   += v.mass / self.dt**2
            self.dx_const[v.id*self.dim+1] += v.mass / self.dt**2
            self.dx_const[v.id*self.dim+2] += v.mass / self.dt**2

        for q_i in ti.static(self.contact_particle_list):
            for d in ti.static(range(self.dim)):
                self.dx_const[q_i*self.dim+d] += self.positional_weight

    def compute_grad(self):
        """ Compute gradient of system state "q" w.r.t. input "y" = \partial q/ \partial y
        """
        dx_const_np = self.dx_const.to_numpy()          # include contact dA
        Mh2 = np.diag(dx_const_np)        # dim: 3N*3N
        lhs_np = self.lhs.to_numpy()

        nabla_g = np.kron(lhs_np, np.eye(3)) - self.dA

        dq_dy_np = np.linalg.solve(nabla_g, Mh2)

        # np.savetxt("Data/lhs_.csv", lhs_np, delimiter=",", fmt="%.8e")
        # np.savetxt("Data/dq_dy_.csv", dq_dy_np, delimiter=",", fmt="%.5f")

    def compute_tangent_stiffness(self):
        """ Compute the tangent stiffness matrix K = \partial^2 E / \partial q^2.
        """
        self.construct_g_hessian()
        self.compute_grad()

    @ti.kernel
    def construct_L_act(self)->ti.f64:
        """ construct Loss: based on desired node (only one) position
        """
        self.dL_dq_y.fill(0.)
        loss = 0.
        for i in range(self.marker_N):
            q_i = self.marker_ti[i]
            desired_pos = self.marker_pos_desired[i]
            current_pos = self.mesh.verts.pos[q_i]
            self.error[i] = current_pos - desired_pos

            self.dL_dq_y[q_i*3]     = 2 * self.error[i].x
            self.dL_dq_y[q_i*3 + 1] = 2 * self.error[i].y
            self.dL_dq_y[q_i*3 + 2] = 2 * self.error[i].z

            loss += self.error[i].norm_sqr()

        return loss

    def compute_z_act(self, itr_num:int):
        """ Compute z = \partial q / \partial y
        """
        dL_dq_y_np = self.dL_dq_y.to_numpy()        # 此处的z是关于y的导数
        z_np = self.z.to_numpy()
        for itr in range(itr_num):
            rhs_dA = self.dA @ z_np + dL_dq_y_np
            rhs_dA_x = rhs_dA[0::3]
            rhs_dA_y = rhs_dA[1::3]
            rhs_dA_z = rhs_dA[2::3]

            z_new_x = self.pre_fact_lhs_solve(rhs_dA_x)
            z_new_y = self.pre_fact_lhs_solve(rhs_dA_y)
            z_new_z = self.pre_fact_lhs_solve(rhs_dA_z)
            z_np = np.ravel(np.column_stack((z_new_x, z_new_y, z_new_z)))
        self.z.from_numpy(z_np)

    def compute_dL_dy(self):
        """ \partial L / \partial y
        """
        loss_tmp = self.construct_L_act()
        self.construct_g_hessian()
        self.compute_z_act(10)

        print(f"Loss: {loss_tmp}")
        print(f"Error: {self.error.to_numpy()}")
        z_np = self.z.to_numpy()
        self.dL_dy = np.multiply(z_np, self.dx_const.to_numpy())
        return loss_tmp

    def preset_gui(self, pos:List[float], target:List[float], up_orient:List[float]):
        """ Taichi GUI pre-setting
        Args:
            pos (List[float])      : Camera position
            target (List[float])   : Camera visual target
            up_orient (List[float]): Camera orientation
        """
        self.window, self.camera, self.scene = gui_set(pos, target, up_orient)
        self.canvas = self.window.get_canvas()
        self.show_preset()

    def show_preset(self):
        self.node_show    = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_N)
        self.node_color   = ti.Vector.field(3, dtype=ti.f32, shape=self.PARTICLE_N)
        self.node_desired = ti.Vector.field(3, dtype=ti.f32, shape=self.marker_N)
        self.edge_show    = ti.Vector.field(2, dtype=ti.i32, shape=self.EDGE_N)
        self.edge_show.from_numpy(self.edge.to_numpy().astype(np.int32))
        
        for q_i in range(self.PARTICLE_N):
            self.node_color[q_i] = ti.Vector([0., 0., 0.])
        for q_i in self.fix_particle_list:
            self.node_color[q_i] = ti.Vector([1., 0., 0.])
        for q_i in self.contact_particle_list:
            self.node_color[q_i] = ti.Vector([0., 0., 1.])
        for q_i in self.marker_list:
            self.node_color[q_i] = ti.Vector([0., 1., 0.])

    def gui_show(self, SHOW_FLAG:bool=True, WRITE_FLAG:bool=False, itr_num:int=0):
        if SHOW_FLAG is False:
            return
        self.scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
        self.scene.ambient_light((.8, .8, .8))

        self.node_show.from_numpy(self.mesh.verts.pos.to_numpy().astype(np.float32))
        # self.scene.particles(self.node_desired, radius=0.002, color=(0., 0.4, 0.))
        self.scene.particles(self.node_show, radius=0.001, per_vertex_color=self.node_color)
        self.scene.lines(self.node_show, width=1., indices=self.edge_show, color=(0., 0., 0.), vertex_count=0)

        self.canvas.scene(self.scene)
        self.canvas.set_background_color((1., 1., 1.))

        if WRITE_FLAG is True:
            self.window.save_image(f"Figure/{itr_num:05d}.png")
        self.window.show()

    def substep(self, step_num:int):
        self.construct_sn()
        self.warm_start()
        for itr in ti.static(range(self.solve_itr)):
            # print(f"Iteration: {itr} ------------------------------------")
            self.local_solve()
            rhs_np = self.rhs.to_numpy()
            # print(f"Rhs:\n{self.rhs_stretch.to_numpy().reshape(-1, 3)}")
            # Split rhs_np into x,y,z components
            rhs_np_x = rhs_np[0::3]
            rhs_np_y = rhs_np[1::3]
            rhs_np_z = rhs_np[2::3]

            node_pos_new_np_x = self.pre_fact_lhs_solve(rhs_np_x)
            node_pos_new_np_y = self.pre_fact_lhs_solve(rhs_np_y)
            node_pos_new_np_z = self.pre_fact_lhs_solve(rhs_np_z)

            self.update_pos_new(node_pos_new_np_x, node_pos_new_np_y, node_pos_new_np_z)
            # print(f"Node pos new:\n", self.node_pos_new.to_numpy().reshape(-1, 3))
        
        self.update_vel_pos()

def main():
    gain = 8.e1

    class Soft(SoftBend2D):
        def __init__(self, shape:list, fix, contact, E:float, nu:float, dt:float, density:float, g=-9.8):
            super().__init__(shape, fix, contact, E, nu, dt, density, g)

    soft = Soft([0.1, 0.1], range(6), [30, 35], 1.e4, 0.4, 0.01, 10e2)
    soft.preset_gui([-0.2, 0.05, 0.15], [0.05, 0.1, 0.], [0., 0., 1.])

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = spla.factorized(s_lhs_np)

    soft.contact_vel[0] = ti.Vector([0.0, -0.01, 0.0])
    soft.contact_vel[1] = ti.Vector([0.0, -0.01, 0.01])

    # pos_sample = []
    for itr in range(200):
        # print(f"Simulation Time: {itr*soft.dt:.2f} ======================================")
        soft.substep(itr)
        soft.gui_show(True, False, itr)
        # pos_sample.append(soft.mesh.verts.pos[32].to_numpy())
    # np.savetxt("Data/pos32_4.csv", np.array(pos_sample), delimiter=",", fmt="%.8e")

    soft.contact_vel.fill(0.)

    for itr in range(100):
        soft.substep(itr)
        soft.gui_show(True, False, itr)
        print(f"Time Step: {itr} ======================================")
        print(f"Marker Node Pos: \n{soft.mesh.verts.pos.to_numpy()[soft.marker_list]}")
    exit()

    soft.marker_pos_desired[0] = soft.mesh.verts.pos[soft.marker_list[0]] + ti.Vector([0., 0., 0.01])

    print(f"Arrive the stable state, start to compute tangent stiffness...")

    soft.substep(1)
    soft.compute_tangent_stiffness()

    for itr in range(100):
        print(f"Time Step: {itr} ======================================")
        soft.substep(itr)
        soft.compute_dL_dy()
        dL_dy = soft.dL_dy.reshape(-1, 3)
        end_speed = -gain * dL_dy[soft.contact_particle_list]
        end_speed_compress = compress_vectors(end_speed, 0.04)
        soft.contact_vel.from_numpy(end_speed_compress)
        print(f"End speed (compressed): {end_speed_compress.flatten()}")
        print(f"Stretch energy: {np.sum(soft.stretch_energy.to_numpy()):e}")
        print(f"Bend energy: {np.sum(soft.bend_energy.to_numpy()):e}")

        soft.gui_show(True, False, itr)


if __name__ == "__main__":
    main()