"""
使用Sofa进行布料模拟的脚本, based on `TriangularFEMForceFieldOptim`
created at 2025-01-03 by hsy
"""

import Sofa
import SofaRuntime
import Sofa.Gui
import os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


def createScene(root):
    plugins = root.addChild('plugins')
    plugins.addObject('RequiredPlugin', name="Sofa.Component")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Algorithm")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Detection.Intersection")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Geometry")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Collision.Response.Contact")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Constraint.Projective")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver.Iterative")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.IO.Mesh")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Mapping.Linear")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Mass")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.ODESolver.Backward")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.SolidMechanics.FEM.Elastic")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.StateContainer")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Topology.Container.Grid")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.Visual")
    plugins.addObject('RequiredPlugin', name="Sofa.Component.LinearSolver")
    plugins.addObject('RequiredPlugin', name="Sofa.GL.Component.Rendering3D")

    root.dt = 0.01
    root.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    root.gravity = [0., 0., -9.8]

    root.addObject('VisualStyle', displayFlags="showBehaviorModels showForceFields")
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=100)

    root.addObject('BruteForceBroadPhase')
    root.addObject('BVHNarrowPhase')
    root.addObject('CollisionPipeline', verbose="0", name="CollisionPipeline")
    root.addObject('CollisionResponse', response="PenalityContactForceField", name="collision response")
    root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1', vdamping=0)
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='100', tolerance='1.e-6', threshold='1.e-6')

    obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo', template='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.01')
    obj.addObject('FixedConstraint', indices="0 1 2 3 4 5 6 7 8 9 10 110 120")
    # obj.addObject('TriangularFEMForceFieldOptim', name="FEM", youngModulus="600", poissonRatio="0.3", method="large", template="Vec3")
    obj.addObject('TriangularFEMForceFieldOptim', name='FEM', 
                  youngModulus='5.e2', poissonRatio='0.4', method='small')
    obj.addObject('TriangularBendingSprings', name="BS", stiffness=3, damping=0.1)
    obj.addObject('DiagonalVelocityDampingForceField', rayleighStiffness=0.5, dampingCoefficient=0.8)
    obj.addObject('TriangleCollisionModel')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    # obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices='120')

    obj_visu = obj.addChild('VisualModel')
    obj_visu.loader = obj_visu.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', triangulate='true',
                                         scale=1.)
    obj_visu.addObject('OglModel', name='model', src='@loader', scale3d=[1.]*3, color=[0., 1., 0.], updateNormals=False)
    obj_visu.addObject('IdentityMapping')


def add_move(handle, dt, movement):
    """Use `LinearMovementConstraint` to add a simulation step-wise movement

    :param handle: The node of the object
    :param dt:
    :param movement: The additional movement
    """
    times_array = handle.findData('keyTimes').value
    movements_array = handle.findData('movements').value

    last_time = times_array[-1]
    last_movement = movements_array[-1,:]

    handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
    handle.findData('movements').value = np.append(movements_array, [movement+last_movement], axis=0)


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

    # for itr in range(1, 10, 1):
    #     print(f'{root.time.value}:{root.object.dofs.position.value[120]}')
    #     add_move(linear_mov, dt, np.array([0., 0.002, 0.]))

    #     # save_pos(dofs, script_dir+f'/node_data/node_pos_{itr}.csv')

    #     Sofa.Simulation.animate(root, dt)

    # for itr in range(1, 20, 1):
    #     print(f'{root.time.value}:{root.object.dofs.position.value[120]}')
    #     add_move(linear_mov, dt, np.array([0., 0., 0.]))

    #     Sofa.Simulation.animate(root, dt)

    Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    print("End of simulation.")


if __name__ == '__main__':
    main()