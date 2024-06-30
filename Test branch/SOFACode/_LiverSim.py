"""
使用liver.msh作为几何网格文件,施加预先指定的变形,模拟变形效果

"""

import Sofa
import SofaRuntime
import Sofa.Gui
import Sofa.SofaGL
import os
import numpy as np
import copy


script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)                                            # 改变当前工作目录到脚本文件所在目录


def createScene(root):
    root.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                 'Sofa.Component.Collision',
                                                 'Sofa.Component.Constraint.Projective',
                                                 'Sofa.Component.IO.Mesh',
                                                 'Sofa.Component.LinearSolver',
                                                 'Sofa.GL.Component.Rendering3D'])

    root.dt = 0.01
    root.bbox = [[-0.1, -0.1, 0], [0.2, 0.2, 0.2]]
    root.gravity = [0, 0, 0]

    # root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')
    root.add('VisualStyle', displayFlags='showVisual showBehaviorModels')
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)

    root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    root.addObject('BruteForceBroadPhase', name='BroadPhase')
    root.addObject('BVHNarrowPhase', name='NarrowPhase')
    root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.5', rayleighMass='0.5')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    obj.addObject('MeshGmshLoader', name='loader', filename='Mesh/liver.msh', scale='0.05', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TetrahedronSetTopologyModifier', name='modifier')
    obj.addObject('TetrahedronSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.01')#, massDensity='0.1')

    obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='3 39 64')
    obj_fixed.drawSize = 0.005

    obj.addObject('TetrahedronFEMForceField', name='FEM', youngModulus='5.e4', poissonRatio='0.45', method='large')
    obj.addObject('TriangleCollisionModel')
    # obj.addObject('LinearSolverConstraintCorrection', solverName='@linearsolver')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    obj_linear_move = obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices=[85])

    visu = obj.addChild('visual')
    visu.addObject('MeshOBJLoader', name='loader', filename='Mesh/liver-smooth.obj', scale='0.05', flipNormals='0')
    visu.addObject('OglModel', name='visual_model', src='@loader', color=[1, 0, 0, 1])
    visu.addObject('BarycentricMapping', name='visual_mapping', input='@../dofs', output='@visual_model')

    surf = obj.addChild('surface')


def add_move(handle, dt, movement):
    """
    Use `LinearMovemetConstraint` to add a simulation step-wise movement
    :param handle: The node of the object
    :param dt:
    :param movement: The additional movement
    :return:
    """
    times_array = handle.findData('keyTimes').value
    movements_array = handle.findData('movements').value

    last_time = times_array[-1]
    last_movement = movements_array[-1, :]

    handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
    handle.findData('movements').value = np.append(movements_array, [movement + last_movement], axis=0)


def save_pos(handle, path):
    node_pos = handle.findData('position').value
    np.savetxt(f'{path}', node_pos, '%.6f', delimiter=',')


def main():
    contact_idx = [85]
    fix_idx = [3, 39, 64]

    root = Sofa.Core.Node('root')
    createScene(root)
    c = root.addObject('Camera', name='c', position=[0, 1, 0], lookAt=[0, 0, 0])

    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 800)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')
    linear_mov = obj.getObject('cnt')

    velocity = np.array([0.005, 0., 0.])
    action = velocity * dt

    contact_pos_np = np.zeros((len(contact_idx), 3))

    for itr in range(500):
        # print(f'Time：{root.time.value:.3f}---------------------------------------')
        add_move(linear_mov, dt, action)
        Sofa.Simulation.animate(root, dt)

        for i, q_idx in enumerate(contact_idx):
            contact_pos_np[i, :] = copy.deepcopy(dofs.findData('position').value[q_idx])
            print(f'Contact position {q_idx}: {contact_pos_np[i, :]}')

    save_pos(dofs, script_dir + '/node_end_pos.csv')

    # Sofa.Gui.GUIManager.MainLoop(root)
    # Sofa.Gui.GUIManager.closeGUI()
    # print("End of simulation.")

if __name__ == '__main__':
    main()