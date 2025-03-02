""" 用于为Contact Selection的2D模拟, 场景是切割前的预拉伸
created at 2025-03-02 by hsy
"""
import Sofa
import SofaRuntime
import Sofa.Gui
import Sofa.SofaGL
import os
import numpy as np
import numpy.typing as npt
import copy
from SOFACode.DiffPD2D import SoftObject2D

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
    root.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    root.gravity = [0., 0., 0.]
    # root.addObject('InteractiveCamera', name='cam', position=[0., 0.3, 0.], orientation=[1., 0., 0., 0.], zoomSpeed=0.1)

    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)

    root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    root.addObject('BruteForceBroadPhase', name='BroadPhase')
    root.addObject('BVHNarrowPhase', name='NarrowPhase')
    root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.5', rayleighMass='0.5')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    # 需要替换网格文件!
    obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.01')#, massDensity='0.1')

    X_EPS = 1.e-3
    obj.addObject('BoxROI', name='box', box=[-X_EPS, -0.06, -0.1, X_EPS, 0.06, 0.1])
    obj.addObject('FixedConstraint', name='fixed', indices='@box.indices')

    obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='5.e2', poissonRatio='0.4', method='large')
    obj.addObject('TriangleCollisionModel')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    # Need change the indices to be equal with manipualtion index ######################################################
    obj.addObject('LinearMovementConstraint', name='cnt1', template="Vec3", indices=[10])
    obj.addObject('LinearMovementConstraint', name='cnt2', template="Vec3", indices=[11])

    # obj_visu = obj.addChild('VisualModel')
    # obj_visu.addObject('OglModel', name='visual')
    # obj_visu.addObject('IdentityMapping', input='@..', output='@visual')


def add_move(handle, dt, movement):
    """ Use `LinearMovemetConstraint` to add a simulation step-wise movement
    Args:
        handle: The node of the object
        dt: The time step
        movement: The additional movement
    """
    times_array = handle.findData('keyTimes').value
    movements_array = handle.findData('movements').value

    last_time = times_array[-1]
    last_movement = movements_array[-1, :]

    handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
    handle.findData('movements').value = np.append(movements_array, [movement + last_movement], axis=0)


def get_marker_pos(handle, marker_idx):
    marker_pos = np.zeros((len(marker_idx), 3))
    # node_pos = handle.findData('position').value
    for i, idx in enumerate(marker_idx):
        pos_tmp = copy.deepcopy(handle.findData('position').value[idx])
        marker_pos[i] = pos_tmp
    return marker_pos


class MyObject(SoftObject):




def main():
    root = Sofa.Core.Node('root')
    createScene(root)

    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 800)


if __file__ == "__main__":
    main()