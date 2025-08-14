""" 用于为MPC控制器验证的2D模拟, 场景是切割前的预拉伸, 使用两个接触点
配合验证`Prentension.py`中*MPC*控制器的约束, 证明MPC方法可以满足约束, 而Broyden方法不行
Sofa 环境似乎不能承受大拉伸, 难以变形到期望距离
created by hsy on 2025-07-30
"""
from pathlib import Path
import Sofa
import SofaRuntime
import Sofa.Gui
import Sofa.SofaGL
import os, sys
from typing import List, Dict, DefaultDict, Tuple
import numpy as np
import numpy.typing as npt 
import taichi as ti
from scipy import sparse
import copy
import meshio
ti.init(arch=ti.cpu, default_fp=ti.f64, default_ip=ti.i32, debug=True)

from Utilize.GenMsh import read_mshv2_triangle, mesh_obj_tri, write_mshv2_tri
from Utilize.sofa_utilize import add_move, save_vtu, get_marker_pos
from Utilize.MathNp import line_from_points_2d, compress_vectors

dir_path = Path(__file__).parent

def convert_node_indice(old_node_num, domain_length=0.1, old_res=0.01, new_res=0.005):
    """ 为了将模型中的节点索引转换为sofa中的索引 """
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

def construct_line(dots_pos_init):
    pos1, pos2 = dots_pos_init[0, :], dots_pos_init[1, :]
    line_params = line_from_points_2d(pos1, pos2)

    return line_params

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

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    # obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MeshGmshLoader', name='loader', filename=f'{dir_path}/Mesh/shape_split.msh', scale='1', flipNormals='0')
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

    return obj, obj_move_list

# ----- Broyden Method -----
def update_jacobian(factor:float, action, delta_pos, Ja:npt.NDArray):
    """
    Args:
        factor (float): Update factor for the Jacobian.
        delta_action (npt.NDArray(,2)): Change in action vector.
        delta_pos (npt.NDArray(,2)): Change in position vector.
        Ja (npt.NDArray): Current Jacobian matrix.
    """
    dim = 2
    m, n = Ja.shape
    point_N, action_N = m // 2, n // 2

    a = Ja.flatten()
    W = np.zeros((dim*point_N, Ja.size))
    for idx in range(dim*point_N):
        W[idx, action_N*dim*idx:action_N*dim*(idx+1)] = action.flatten()
        # for a_i in range(action_N):
        #     W[idx, action_N*dim*idx + a_i]     = action[a_i, 0]
        #     W[idx, action_N*dim*idx + a_i + 1] = action[a_i, 1]

    e = W @ a - delta_pos.flatten()
    a -= factor * W.transpose() @ e

    # a: shape:(N*2, 2), e: shape:(N*4,)
    return a.reshape((m, n)), e, W @ a

def cal_loss(dot_pos_soft:npt.NDArray, factor:float, dot_pos_init, line_params):
    """ 将两个标记点拉伸到指定间距 """
    pos1, pos2 = dot_pos_soft[:, :2]
    dis_tmp = np.linalg.norm(pos1 - pos2)
    dis_desired = np.linalg.norm(dot_pos_init[0,:] - dot_pos_init[1,:]) * factor

    a, b, c = line_params
    line_normal = np.array([a, b], dtype=np.float64)

    # 两个点到直线的距离
    line_distance1 = line_normal.dot(pos1) + c
    line_distance2 = line_normal.dot(pos2) + c

    loss = (dis_tmp - dis_desired) ** 2 + line_distance1 ** 2 + line_distance2 ** 2
    grad1 = 2*(dis_tmp - dis_desired) * (pos1 - pos2) / dis_tmp + line_distance1 * line_normal
    grad2 = 2*(dis_tmp - dis_desired) * (pos2 - pos1) / dis_tmp + line_distance2 * line_normal

    dL_dq = np.zeros(2*2)
    dL_dq[:2]  = grad1
    dL_dq[2:4] = grad2

    return dis_tmp, loss, dL_dq

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
        self.F_i_det = ti.field(dtype=ti.f64, shape=self.triangles_num)
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
            self.F_i_det[f_i] = F_i[0, 0] * F_i[1, 1] - F_i[0, 1] * F_i[1, 0]

            U, sig, V = ti.svd(F_i, ti.f64)
            self.stretch_energy[f_i] = 0.5 * self.triange_area[f_i] * ((sig[0,0]-1.)**2 + (sig[1,1]-1.)**2)


def main(contact_list:List[int], marker_list:List[int]):
    contact_angle = {11:90, 22:90, 33:90, 44:90, 55:90, 66:90, 77:90, 88:90, 99:90,
                    21:-90, 32:-90, 43:-90, 54:-90, 65:-90, 76:-90, 87:-90, 98:-90, 109:-90,
                    111:0, 112:0, 113:0, 114:0, 115:0, 116:0, 117:0, 118:0, 119:0}

    shape = [0.1, 0.1]
    fix = range(11)

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

    node_np, _, ele_np = mesh_obj_tri(shape, 0.01/2)
    msh_file:str = dir_path / "Mesh/shape_split.msh"
    write_mshv2_tri(msh_file, node_np, ele_np)

    dots_pos_init = node_np[marker_sofa]
    line_params = construct_line(dots_pos_init)
    print(f"Initial marker positions: {dots_pos_init}")
    # print(f"Line parameters: {line_params}")
    # exit()

    # ----- Setup Sofa scene -----
    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, contact_sofa)
    Sofa.Simulation.init(root)

    soft_sofa = SofaObject(node_np, ele_np)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    gain = 1.e1
    # ja = np.tile(np.ones((2, 2)), (2, 2))
    # ja = np.tile(np.eye(2, 2), (len(marker_list), len(contact_list)))
    ja = np.eye(4)

    dots_pos_soft = get_marker_pos(dofs, marker_sofa)[:, :2]
    action = np.zeros((2, 2))

    loss_list = []
    max_strain_list = []

    for step in range(100):
        print(f"Time Step: {step*0.01} ======================================")
        dots_pos_sofa_new = get_marker_pos(dofs, marker_sofa)[:, :2]
        print(f'Detected marker position: {dots_pos_sofa_new.flatten()}')

        sofa_pos_tmp = dofs.findData('position').value
        soft_sofa.node_pos.from_numpy(sofa_pos_tmp[:,:2])
        soft_sofa.construct_rhs_stretch()
        max_strain = np.max(soft_sofa.F_i_det.to_numpy())

        delta_pos = dots_pos_sofa_new - dots_pos_soft
        dots_pos_soft = dots_pos_sofa_new

        _, loss_tmp, dL_dq = cal_loss(dots_pos_soft[:,:2], 1.1, dots_pos_init, line_params)
        dL_da = dL_dq @ ja
        ja_new, ja_error, delta_pos_ada = update_jacobian(1.e5, action, delta_pos, ja)
        ja = ja_new

        action_flat = -gain * dL_da
        action = action_flat.reshape(-1, 2)
        action = compress_vectors(action, 2.e-4)

        print(f"Loss: {loss_tmp}")
        print(f"End speed: {action.flatten()}")
        print(f"Sofa Defomation Strain: {soft_sofa.stretch_energy.to_numpy().sum():e}")

        add_move(move_handle, dt, np.repeat(action, 3, axis=0))
        Sofa.Simulation.animate(root, dt)

        loss_list.append(loss_tmp)
        max_strain_list.append(max_strain)

    np.savetxt(f"{dir_path}/Data/loss_list.csv", loss_list, fmt="%e", delimiter=",")
    np.savetxt(f"{dir_path}/Data/max_strain_list.csv", max_strain_list, fmt="%e", delimiter=",")


if __name__ == "__main__":
    main([55, 98], [93, 105])