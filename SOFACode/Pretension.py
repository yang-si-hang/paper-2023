""" 用于为MPC控制器验证的2D模拟, 场景是切割前的预拉伸, 使用两个接触点
目的是为了验证*MPC*控制器的约束, 并证明不同的接触点选择影响任务完成效率
Sofa 环境似乎不能承受大拉伸, 难以变形到期望距离
created at 2025-03-29 by hsy
"""
import Sofa
import SofaRuntime
import Sofa.Gui
import Sofa.SofaGL
import os, sys
from typing import List, Dict, DefaultDict, Tuple
import numpy as np
import numpy.typing as npt 
from scipy import sparse
import casadi as ca
import copy
import meshio
import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)                                            # 改变当前工作目录到脚本文件所在目录

root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from SOFACode._DiffPD2D import SoftObject2D, line_from_points_2d, compress_vectors
from Utilize.GenMsh import read_mshv2_triangle, mesh_obj_tri, write_mshv2_tri


def convert_node_indice(old_node_num, domain_length=0.1, old_res=0.01, new_res=0.005):
    # Compute the number of nodes in one row for each grid.
    # We add 1 because the grid goes from 0 to domain_length inclusive.
    old_n = int(domain_length / old_res) + 1  # e.g., 11
    new_n = int(domain_length / new_res) + 1  # e.g., 21

    # Find the x (column) and y (row) indices in the original grid.
    i_old = old_node_num % old_n
    j_old = old_node_num // old_n

    # Compute the corresponding indices in the new grid.
    # Since the new resolution is twice as fine, each old index multiplies by factor 2.
    i_new = i_old * int(old_res / new_res)  # or simply 2 * i_old
    j_new = j_old * int(old_res / new_res)  # or simply 2 * j_old

    # Compute the new serial number (using X-first order)
    new_node_num = i_new + j_new * new_n
    return int(new_node_num)


def createScene(root, contact:List[int]):
    root.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                 'Sofa.Component.Collision',
                                                 'Sofa.Component.Constraint.Projective',
                                                 'Sofa.Component.IO.Mesh',
                                                 'Sofa.Component.LinearSolver',
                                                 'Sofa.GL.Component.Rendering3D'])
    
    root.dt = 0.01
    root.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    root.gravity = [0., 0., 0.]
    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')

    root.addObject('DefaultAnimationLoop', )
    root.addObject('CollisionPipeline', depth="6", verbose="0", draw="0")
    root.addObject('BruteForceBroadPhase', )
    root.addObject('BVHNarrowPhase', )
    root.addObject('NewProximityIntersection', name="Proximity", alarmDistance="0.5", contactDistance="0.2")
    root.addObject('CollisionResponse', name="Response", response="PenalityContactForceField")

    # FEM的设置
    # root.addObject('FreeMotionAnimationLoop')
    # root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)

    # root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    # root.addObject('BruteForceBroadPhase', name='BroadPhase')
    # root.addObject('BVHNarrowPhase', name='NarrowPhase')
    # root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    # root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    # obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MeshGmshLoader', name='loader', filename='Mesh/shape_split.msh', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.1')#, massDensity='0.1')

    X_EPS = 5.e-3
    obj.addObject('BoxROI', name='box', box=f"-0.1 {-X_EPS} -0.1 0.11 {X_EPS} 0.1")
    obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='@box.indices')

    obj.addObject('MeshSpringForceField', name="springs", trianglesStiffness=90, trianglesDamping=0.3)
    # obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='5.e2', poissonRatio='0.3', method='large')
    obj.addObject('TriangleCollisionModel')
    # obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    obj_move_list = []
    for q_i in contact:
        obj_move_list.append(obj.addObject('LinearMovementConstraint', name='cnt'+str(q_i), template="Vec3", indices=[q_i]))

    # 输出Sofa设置信息
    # Sofa.msg_info("Scene", f"Contact indices: {obj_linear_move.indices.value}")
    # Sofa.msg_info("User", f"Fixed indices: {obj_fixed.indices.value}")

    return obj, obj_move_list


def add_move(handle_list:list, dt:float, movement:npt.NDArray):
    """ Use `LinearMovemetConstraint` to add a simulation step-wise movement
    Args:
        handle: The node of the object
        dt: The time step
        movement: The additional movement
    """
    if movement.shape[1] == 2:
        movement = np.concatenate((movement, np.zeros((movement.shape[0], 1))), axis=1)
    for i, handle in enumerate(handle_list):
        times_array = handle.findData('keyTimes').value
        movements_array = handle.findData('movements').value

        last_time = times_array[-1]
        last_movement = movements_array[-1, :]

        handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
        handle.findData('movements').value = np.append(movements_array, [movement[i,:] + last_movement], axis=0)


def get_marker_pos(handle, marker_idx:list)->npt.NDArray:
    """从sofa中获取指定节点的位置
    """
    marker_pos = np.zeros((len(marker_idx), 3))
    # node_pos = handle.findData('position').value
    for i, idx in enumerate(marker_idx):
        pos_tmp = copy.deepcopy(handle.findData('position').value[idx])
        marker_pos[i] = pos_tmp
    return marker_pos


def save_vtu(mesh_file:str, pos:npt.NDArray, write_name:str):
    """Save the node position to a .vtu file

    Args:
        mesh_file (str): The initial mesh file name
        pos (npt.NDArray): The node position
        write_name (istrnt): The write file name
    """
    _, triangles = read_mshv2_triangle(mesh_file)

    cells_write = [("triangle", triangles)]
    mesh = meshio.Mesh(points=pos, cells=cells_write)
    mesh.write(f"data/{write_name}")


class MyObject(SoftObject2D):
    def __init__(self, shape, fix, contact, dots_list, E, nu, dt, density, **kwargs):
        super().__init__(shape, fix, contact, E, nu, dt, density, **kwargs)
        self.loss = 0.
        self.contact_j = np.zeros((self.PARTICLE_N, self.CON_N))
        self.dots_idx = dots_list
        dots_num = len(dots_list)
        self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=dots_num)
        self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=dots_num)

        print(f"Marker index: {self.dots_idx}")

        self.line_params = self.construct_line()

    
    def construct_line(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos_init[i] = self.node_pos_init[idx]

        pos1, pos2 = self.dot_pos_init[0].to_numpy(), self.dot_pos_init[1].to_numpy()
        line_params = line_from_points_2d(pos1, pos2)

        return line_params
    

    def update_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos[i] = self.node_pos[idx]

        
    def construct_L_sofa(self, dot_sofa:npt.NDArray, factor:float):
        """将两个标记点拉伸到指定间距
        """
        self.dL_dq_contact.fill(0.)
        if len(self.dots_idx) != 2:
            raise ValueError("The number of marker is not 2")
        else:
            idx1, idx2 = self.dots_idx
        pos1, pos2 = dot_sofa[:,:2]         # sofa中是三维坐标
        dis_tmp = np.linalg.norm(pos1 - pos2)
        dis_desired = np.linalg.norm(
            self.node_pos_init[idx1].to_numpy() - self.node_pos_init[idx2].to_numpy()
            ) * factor

        a, b, c = self.line_params
        line_normal = np.array([a, b], dtype=np.float64)

        # 两个点到直线的距离
        line_distance1 = line_normal.dot(pos1) + c
        line_distance2 = line_normal.dot(pos2) + c

        loss = (dis_tmp - dis_desired) ** 2 + line_distance1 ** 2 + line_distance2 ** 2
        grad1 = 2*(dis_tmp - dis_desired) * (pos1 - pos2) / dis_tmp + line_distance1 * line_normal
        grad2 = 2*(dis_tmp - dis_desired) * (pos2 - pos1) / dis_tmp + line_distance2 * line_normal

        self.dL_dq_contact[idx1*2]     = grad1[0]
        self.dL_dq_contact[idx1*2 + 1] = grad1[1]
        self.dL_dq_contact[idx2*2]     = grad2[0]
        self.dL_dq_contact[idx2*2 + 1] = grad2[1]

        return dis_tmp, loss
        

    def compute_dcontact(self, dot_sofa:npt.NDArray):
        """ \partial L / \partial y with contact action, 计算关于contact的导数, 用于控制
        """
        dist_tmp, loss_tmp = self.construct_L_sofa(dot_sofa, 1.15)
        self.construct_g_hessian()
        self.compute_z(10)

        print(f"Distance: {dist_tmp}; Loss: {loss_tmp}")

        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())

        return loss_tmp
    

    def compute_jacobian(self):
        """计算全局雅可比矩阵, 得到dq/dy(2n*2n)
        """
        A = self.lhs.to_numpy()
        matrix_l = np.kron(A, np.eye(2)) - self.dA
        matrix_r = np.diag(self.dx_const.to_numpy())
        self.dq_dy_np = np.linalg.solve(matrix_l, matrix_r)


    def compute_contact_j(self):
        """从dq/dy中提取全局节点关于contact的雅可比矩阵
        """
        self.construct_g_hessian()
        self.compute_jacobian()

        contact = np.asarray(self.contact_particle_list, dtype=np.int32)
        marker = np.asarray(self.dots_idx, dtype=np.int32)
        contact_dims = np.column_stack((2 * contact, 2 * contact + 1)).ravel()
        marker_dims = np.column_stack((2 * marker, 2 * marker + 1)).ravel()

        contact_j = self.dq_dy_np[:, contact_dims]
        self.contact_j = contact_j.copy()
        return contact_j
    

    # def build_mpc(self, H:int, dots_sofa:npt.NDArray, factor:float):
        """构建NLP问题
        """
        U = ca.MX.sym("U", 2 * self.CON_N * H)
        total_loss = ca.MX(0.0)  # 初始化为 ca.MX 对象
        decrease_w = ca.MX(1.)
        u_max = 0.02 * self.dt
        node_pos = ca.MX(self.node_pos.to_numpy())         # dim:N*2
        contact_j_casadi = ca.MX(self.contact_j)

        dis_desired = ca.MX(np.linalg.norm(
            self.node_pos_init[self.dots_idx[0]] - self.node_pos_init[self.dots_idx[1]]) * factor)

        a, b, c = self.line_params
        g = []
        lbg = []
        ubg = []

        pos1 = ca.MX(dots_sofa[0, :2].reshape(1, 2))
        pos2 = ca.MX(dots_sofa[1, :2].reshape(1, 2))
        # 不加此刻的loss, 是因为是定值, 无需交给nlp优化

        for t in range(H):
            u_t = U[2 * self.CON_N * t: 2 * self.CON_N * (t + 1)]

            # Action norm constraint
            g.append(ca.sumsqr(u_t[0:2]))
            lbg.append(0)
            ubg.append(u_max)

            g.append(ca.sumsqr(u_t[2:4]))
            lbg.append(0)
            ubg.append(u_max)

            dnode_pos = contact_j_casadi @ u_t
            dnode_pos = ca.reshape(dnode_pos, self.PARTICLE_N, 2)
            node_pos += dnode_pos

            pos1 += dnode_pos[self.dots_idx[0], :]
            pos2 += dnode_pos[self.dots_idx[1], :]
            dis_now = ca.norm_2(pos1 - pos2)

            dist_loss = (dis_now - dis_desired)**2

            line1 = a * pos1[0] + b * pos1[1] + c
            line2 = a * pos2[0] + b * pos2[1] + c
            line_loss = line1**2 + line2**2

            total_loss += np.power(decrease_w, t) * (dist_loss + line_loss)

        # Set up NLP
        nlp = {
            'x': U,
            'f': total_loss,
            'g': ca.vertcat(*g)
        }

        solver = ca.nlpsol("solver", "ipopt", nlp, {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.tol': 1e-10
        })

        U_init = np.zeros(2 * self.CON_N * H)
        sol = solver(x0=U_init, lbg=lbg, ubg=ubg)
        U_opt = np.array(sol['x']).flatten()
        print(f"MPC Calculation Loss: {sol['f']}")

        return U_opt[:2*self.CON_N]


            # deformation strain constraint
            # for e_i in range(self.ELEMENT_N):
            #     idx1, idx2, idx3 = self.ele[e_i]
            #     q1, q2, q3 = node_pos[idx1], node_pos[idx2], node_pos[idx3]
            #     X_f = ca.MX(ca.Matrix.cols([q2 - q1, q3 - q1]))


    def build_mpc(self, H: int, contact_j:npt.NDArray, dots_sofa: npt.NDArray, factor: float):
        """ 构建NLP问题
        """
        opti = ca.Opti()  # 创建Opti实例

        # 定义决策变量
        U = opti.variable(2 * self.CON_N * H)

        # 初始化损失函数
        total_loss = ca.MX(0.0)
        decrease_w = 1.0
        u_max = 0.02 * self.dt
        node_pos_np = np.asarray(self.node_pos.to_numpy(), dtype=np.float64)  # dim:N*2
        node_pos = ca.MX(node_pos_np.flatten())  # dim:2N
        contact_j_casadi = ca.MX(contact_j)
        ele_Xg_inv = self.Xg_inv.to_numpy()

        # 计算期望距离
        dis_desired = np.linalg.norm(
            self.node_pos_init[self.dots_idx[0]] - self.node_pos_init[self.dots_idx[1]]) * factor

        a, b, c = self.line_params

        # 初始化位置
        pos1 = ca.MX(dots_sofa[0, :2].reshape(1, 2))
        pos2 = ca.MX(dots_sofa[1, :2].reshape(1, 2))

        F_i_det_con = ca.MX(np.zeros(self.ELEMENT_N * H))

        for t in range(H):
            u_t = U[2 * self.CON_N * t: 2 * self.CON_N * (t + 1)]

            # Action norm constraint
            opti.subject_to(ca.sumsqr(u_t[0:2]) <= u_max**2)
            opti.subject_to(ca.sumsqr(u_t[2:4]) <= u_max**2)

            # 更新节点位置
            dnode_pos = contact_j_casadi @ u_t
            node_pos += dnode_pos

            # deformation strain constraint
            for f_i in range(self.ELEMENT_N):
                idx1, idx2, idx3 = self.ele[f_i]
                q1, q2, q3 = node_pos[2*idx1:2*idx1 + 2], node_pos[2*idx2:2*idx2 + 2], node_pos[2*idx3:2*idx3 + 2]
                X_f = ca.horzcat((q2 - q1), (q3 - q1))
                F_i = X_f @ ele_Xg_inv[f_i]
                F_i_det = F_i[0, 0] * F_i[1, 1] - F_i[0, 1] * F_i[1, 0]
                F_i_det_con[f_i + t * self.ELEMENT_N] = F_i_det
                opti.subject_to(F_i_det <= 1.4)

            # 更新位置
            pos1 += dnode_pos[2*self.dots_idx[0]: 2*self.dots_idx[0] + 2].T
            pos2 += dnode_pos[2*self.dots_idx[1]: 2*self.dots_idx[1] + 2].T

            # 计算当前距离
            dis_now = ca.norm_2(pos1 - pos2)

            # 计算距离损失
            dist_loss = (dis_now - dis_desired)**2

            # 计算线损失
            line1 = a * pos1[0] + b * pos1[1] + c
            line2 = a * pos2[0] + b * pos2[1] + c
            line_loss = line1**2 + line2**2

            # 更新总损失
            total_loss += np.power(decrease_w, t) * (dist_loss + line_loss)

        # 定义目标函数
        opti.minimize(total_loss)

        # 设置求解器选项
        opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-8})

        # 初始猜测
        opti.set_initial(U, np.zeros(2 * self.CON_N * H))

        # 求解问题
        sol = opti.solve()

        # 获取优化结果
        U_opt = sol.value(U)
        max_strain = sol.value(F_i_det_con).max()
        print(f"MPC Calculation Loss: {sol.value(total_loss)}")

        print(f"Max strain: {max_strain}")

        return U_opt[:2*self.CON_N], max_strain


@ti.data_oriented
class SofaObject:
    def __init__(self, points:npt.NDArray, triangles:npt.NDArray):
        self.points_num = points.shape[0]
        self.triangles_num = triangles.shape[0]

        self.node_pos = ti.Vector.field(2, dtype=ti.f64, shape=self.points_num)
        self.node_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=self.points_num)
        self.node_pos.from_numpy(points[:,:2])
        self.node_pos_init.from_numpy(points[:,:2])

        self.triangles = ti.Vector.field(3, dtype=ti.i32, shape=self.triangles_num)
        self.Xg_inv = ti.Matrix.field(2, 2, dtype=ti.f64, shape=self.triangles_num)
        self.stretch_energy = ti.field(dtype=ti.f64, shape=self.triangles_num)
        self.triange_area = ti.field(dtype=ti.f64, shape=self.triangles_num)
        self.triangles.from_numpy(triangles)

        self.compute_area()
        self.construct_Xg_inv()

    
    @ti.kernel
    def compute_area(self):
        for f_i in range(self.triangles_num):
            ia, ib, ic = self.triangles[f_i]
            qa, qb, qc = self.node_pos_init[ia], self.node_pos_init[ib], self.node_pos_init[ic]
            self.triange_area[f_i] = 0.5 * ti.abs(((qb - qa).cross(qc - qa)))


    @ti.kernel
    def construct_Xg_inv(self):
        for i in range(self.triangles_num):
            ia, ib, ic = self.triangles[i]
            a = ti.Vector([self.node_pos_init[ia].x, self.node_pos_init[ia].y])
            b = ti.Vector([self.node_pos_init[ib].x, self.node_pos_init[ib].y])
            c = ti.Vector([self.node_pos_init[ic].x, self.node_pos_init[ic].y])
            B_i_inv = ti.Matrix.cols([b - a, c - a])
            self.Xg_inv[i] = B_i_inv.inverse()


    @ti.kernel
    def construct_rhs_stretch(self):
        for f_i in range(self.triangles_num):
            idx1, idx2, idx3 = self.triangles[f_i]
            a, b, c = self.node_pos[idx1], self.node_pos[idx2], self.node_pos[idx3]
            X_f = ti.Matrix.cols([b - a, c - a])
            F_i = ti.cast(X_f @ self.Xg_inv[f_i], ti.f64)

            U, sig, V = ti.svd(F_i, ti.f64)
            self.stretch_energy[f_i] = 0.5 * self.triange_area[f_i] * ((sig[0,0]-1.)**2 + (sig[1,1]-1.)**2)


def main(contact_list:List[int], marker_list:List[int]):
    # contact_sofa = [399, 420, 421, 439, 440, 419]
    # contact_sofa = [231, 252, 273, 439, 440, 419]
    # contact_list = [66, 120]
    # marker_sofa = [346, 390]
    # marker_list = [93, 105]

    contact_angle = {11:90, 22:90, 33:90, 44:90, 55:90, 66:90, 77:90, 88:90, 99:90,
                    21:-90, 32:-90, 43:-90, 54:-90, 65:-90, 76:-90, 87:-90, 98:-90, 109:-90,
                    111:0, 112:0, 113:0, 114:0, 115:0, 116:0, 117:0, 118:0, 119:0}

    shape = [0.1, 0.1]
    fix = range(11)
    H:int = 3           # MPC的Horizon
    # gain = 5.e2
    contact_sofa, marker_sofa = [], []

    for contact in contact_list:
        contact_sofa2 = convert_node_indice(contact)
        if contact_angle[contact] == 90 or contact_angle[contact] == -90:
            contact_sofa1, contact_sofa3 = contact_sofa2 - 21, contact_sofa2 + 21
        elif contact_angle[contact] == 0:
            contact_sofa1, contact_sofa3 = contact_sofa2 - 1, contact_sofa2 + 1
        else:
            print("Error of contact indice.")
        contact_sofa += [contact_sofa1, contact_sofa2, contact_sofa3]

    for marker in marker_list:
        marker_sofa.append(convert_node_indice(marker))
    contact_sofa = [int(x) for x in contact_sofa]
    marker_sofa = [int(x) for x in marker_sofa]
    # print(contact_sofa, marker_sofa)

    node_np, _, ele_np = mesh_obj_tri(shape, 0.01/2)
    msh_file:str = "Mesh/shape_split.msh"
    write_mshv2_tri(msh_file, node_np, ele_np)

    # Setup Sofa scene -----------------------------------------
    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, contact_sofa)
    Sofa.Simulation.init(root)

    soft_sofa = SofaObject(node_np, ele_np)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    contact_pos_np = np.zeros((len(contact_list), 3))

    # Setup deformation model (PD) ------------------------------
    params = {"E": 1.e4, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = MyObject(shape, fix, contact_list, marker_list, **params)
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    loss_list = []
    max_strain_list = []

    for step in range(100):
        print(f"Time Step: {step} ======================================")
        dots_pos_sofa_new = get_marker_pos(dofs, marker_sofa)
        print(f'Detected marker position: {dots_pos_sofa_new.flatten()}')

        sofa_pos_tmp = dofs.findData('position').value
        soft_sofa.node_pos.from_numpy(sofa_pos_tmp[:,:2])
        soft_sofa.construct_rhs_stretch()
        # save_vtu('Mesh/shape_split.msh', sofa_pos_tmp, f'shape_{step:04d}.vtu')

        soft.substep(step)
        contact_j = soft.compute_contact_j()        # 以此作为MPC的线性模型
        # np.savetxt(f"contact.csv", cantact_j, fmt="%.6f", delimiter=",")
        vel_flatten, max_strain = soft.build_mpc(1, contact_j, dots_pos_sofa_new[:,:2], 1.1)
        _, loss_tmp = soft.construct_L_sofa(dots_pos_sofa_new[:,:2], 1.1)

        soft.update_dot_pos()
        print(f"Model marker position: {soft.dot_pos.to_numpy().flatten()}")

        end_speed = vel_flatten.reshape(-1, 2) / soft.dt
        soft.contact_vel.from_numpy(end_speed)

        # dy_dcontact = soft.dy_contact.reshape(-1, 2)        # reshape到与接触点个数相同
        # end_speed = -gain * dy_dcontact[soft.contact_particle_list]
        # end_speed_compress = compress_vectors(end_speed, 0.02)
        # soft.contact_vel.from_numpy(end_speed_compress)
        
        print(f"Loss: {loss_tmp}")
        print(f"End speed: {end_speed.flatten()}")
        print(f"Model Deformation Strain: {soft.stretch_energy.to_numpy().sum():e}")
        print(f"Sofa Defomation Strain: {soft_sofa.stretch_energy.to_numpy().sum():e}")

        add_move(move_handle, dt, np.repeat(end_speed * dt, 3, axis=0))
        Sofa.Simulation.animate(root, dt)

        loss_list.append(loss_tmp)
        max_strain_list.append(max_strain)

    np.savetxt("loss_list.csv", loss_list, fmt="%e", delimiter=",")
    np.savetxt("max_strain_list.csv", max_strain_list, fmt="%e", delimiter=",")

if __name__ == "__main__":
    # main([55, 98], [93, 105])
    main([77, 118], [93, 105])