""" 使用Adaptive Controller (Broyden) 完成多特征点控制
基于Sofa环境
created by ysh on 2025-08-11
"""
from pathlib import Path
from typing import List
import numpy.typing as npt
import time
import Sofa
import SofaRuntime
import Sofa.Gui
import os
import numpy as np
import copy

from Utilize.GenMsh import mesh_obj_tri, write_mshv2_tri
from Utilize.sofa_utilize import add_move, get_marker_pos, move_desire
from Utilize.MathNp import compress_vectors

dir_path = Path(__file__).parent

def createScene(root, contact:list):
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

    obj.addObject('MeshGmshLoader', name='loader', filename=f"{dir_path}/Mesh/shape.msh", scale='1', flipNormals='0')
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

def cal_loss(dot_pos_soft:npt.NDArray, dot_pos_target:npt.NDArray):
    """ 将多个标记点变形到目标位置 """
    loss = 0.
    dL_dq = np.zeros_like(dot_pos_soft)
    for i, dot in enumerate(dot_pos_soft):
        target_pos = dot_pos_target[i, :]
        error = dot - target_pos
        loss += np.sum(error**2)
        dL_dq[i, :] = 2 * error
    return loss, dL_dq.flatten()

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

def main(contact_list:List[int], marker_list:List[int]):
    shape = [0.1, 0.1]
    fix = range(11)

    node_np, _, ele_np = mesh_obj_tri(shape, 0.01)
    msh_file:str = dir_path / "Mesh/shape.msh"
    write_mshv2_tri(msh_file, node_np, ele_np)

    # ----- Setup Sofa scene -----
    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, contact_list)
    Sofa.Simulation.init(root)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')
    
    # move_desire(root, move_handle, 2.0, np.array([[0.005, 0.01]] * len(contact_list)))
    # dots_pos_target = get_marker_pos(dofs, marker_list)
    # print(f"Target marker position: {dots_pos_target[:,2].flatten()}")
    # exit()

    # Sofa.Simulation.reset(root)
    # # 清空LinearMovementConstraint
    # for handle in move_handle:
    #     handle.findData('keyTimes').value = np.array([0.])
    #     handle.findData('movements').value = np.array([[0., 0., 0.]])

    dots_pos_target = np.array([[0.01340374, 0.08088684],
                                [0.02344932, 0.09084837],
                                [0.03346793, 0.09087583],
                                [0.08404906, 0.09356144]])

    gain = 1.e0
    ja = np.ones((len(marker_list)*2, len(contact_list)*2))
    # ja = np.kron(np.ones((len(marker_list), len(contact_list))), np.eye(2))

    dots_pos_soft = get_marker_pos(dofs, marker_list)[:, :2]
    action = np.zeros((len(contact_list), 2))

    loss_list = []
    delta_pos_list = []
    delta_pos_ada_list = []

    for step in range(200):
        print(f"Time Step: {step*0.01:.2f} ======================================")
        dots_pos_sofa_new = get_marker_pos(dofs, marker_list)[:, :2]
        print(f'Detected marker position: {dots_pos_sofa_new.flatten()}')

        delta_pos = dots_pos_sofa_new - dots_pos_soft
        dots_pos_soft = dots_pos_sofa_new

        loss_tmp, dL_dq = cal_loss(dots_pos_soft[:,:2], dots_pos_target[:,:2])
        dL_da = dL_dq @ ja
        ja_new, ja_error, delta_pos_ada = update_jacobian(1.e5, action, delta_pos, ja)
        ja = ja_new
        print(f"ja: {ja.flatten()}")

        action_flat = -gain * dL_da
        action = action_flat.reshape(-1, 2)
        action = compress_vectors(action, 2.e-4)

        print(f"Loss: {loss_tmp}")
        print(f"End speed: {action}; {np.linalg.norm(action)}")
        # print(f"Contact pos: {get_marker_pos(dofs, contact_list)[:, :2]}")

        add_move(move_handle, dt, action)
        Sofa.Simulation.animate(root, dt)

        loss_list.append(loss_tmp)
        delta_pos_list.append(delta_pos.flatten())
        delta_pos_ada_list.append(delta_pos_ada.flatten())

    np.savetxt(f"{dir_path}/Data/loss_list_ada_{contact_list[0]}.csv", loss_list, fmt="%e", delimiter=",")
    np.savetxt(f'{dir_path}/Data/delta_pos_ada_{contact_list[0]}.csv', np.array(delta_pos_list), fmt='%e', delimiter=',')
    np.savetxt(f'{dir_path}/Data/delta_pos_ada_pred_{contact_list[0]}.csv', np.array(delta_pos_ada_list), fmt='%e', delimiter=',')

if __name__ == "__main__":
    main([120], [89, 101, 102, 107])

    # candidate_contact = [110, 120, 111, 112, 114, 115, 116, 117, 119]
    # for contact in candidate_contact:
    #     main([contact], [89, 101, 102])

