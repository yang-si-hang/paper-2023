"""
Construct a 2D square simulation scene by SOFA
Apply node movement in every simulation step
"""


import Sofa
import SofaRuntime
import Sofa.Gui
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


def createScene(root):
    root.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                 'Sofa.Component.Collision',
                                                 'Sofa.Component.Constraint.Projective',
                                                 'Sofa.Component.IO.Mesh',
                                                 'Sofa.Component.LinearSolver',
                                                 'Sofa.GL.Component.Rendering3D'])

    root.dt = 0.01
    root.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    root.gravity = [0., 0., 0.]
    # root.addObject('InteractiveCamera', name='cam', position=[0., 0.3, 0.], orientation=[1., 0., 0., 0.], zoomSpeed=0.1)

    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=100)

    root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    root.addObject('BruteForceBroadPhase', name='BroadPhase')
    root.addObject('BVHNarrowPhase', name='NarrowPhase')
    root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    config_node = root.addChild("Config")
    config_node.addObject("OglSceneFrame", style="Arrows", alignment="TopRight")

    obj = root.addChild('object')
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.01')#, massDensity='0.1')
    obj.addObject('FixedConstraint', indices="0 1 2 3 4 5 6 7 8 9 10")
    obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='100', poissonRatio='0.3', method='large')
    # obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='5.e4', poissonRatio='0.3', method='large')
    # obj.addObject('TriangleBendingSprings', name='FEM-Bend', stiffness='100', damping='1.0')
    obj.addObject('TriangleCollisionModel')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    # obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices='120', keyTimes=[0., 5., 10.],
    #               movements=[0., 0., 0., 0.01, 0., 0., 0.02, 0., 0.])
    obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices='120')
    # obj.addObject('PartialLinearMovementConstraint', name='cnt', template="Vec3", indices='120')

    obj_visu = obj.addChild('VisualModel')
    obj_visu.loader = obj_visu.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', triangulate='true',
                                         scale=1.)
    obj_visu.addObject('OglModel', name='model', src='@loader', scale3d=[1.]*3, color=[0., 1., 0.], updateNormals=False)
    # obj_visu.addObject('RigidMapping')
    obj_visu.addObject('IdentityMapping')


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
    last_movement = movements_array[-1,:]

    handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
    handle.findData('movements').value = np.append(movements_array, [movement+last_movement], axis=0)


def save_pos(handle, path):
    node_pos = handle.findData('position').value
    np.savetxt(f'{path}', node_pos, '%.6f')


def main():
    root = Sofa.Core.Node('root')

    createScene(root)

    Sofa.Simulation.init(root)

    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 800)

    dt = root.dt.value
    obj = root.getChild('object')
    linear_mov = obj.getObject('cnt')
    dofs = obj.getObject('dofs')

    # pp = 0.01 * np.random.random()
    # add_move(linear_mov, dt, np.array([pp, 0., 0.]))

    for itr in range(1, 5, 1):
        print(f'{root.time.value}:{root.object.dofs.position.value[120]}')
        pp = pp = 0.01 * np.random.random()
        add_move(linear_mov, dt, np.array([pp, 0., 0.]))

        save_pos(dofs, script_dir+f'/node_data/node_pos_{itr}.csv')

        Sofa.Simulation.animate(root, dt)

    print(f'{root.time.value}:{root.object.dofs.position.value[120]}')

    times = linear_mov.keyTimes.value
    movements = linear_mov.movements.value
    print(f'{times}, {movements}')

    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    print("End of simulation.")


if __name__ == '__main__':
    main()