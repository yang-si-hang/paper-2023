""" 比较PD模型与SOFA的仿真速率, 针对单一任务来比较，而不单纯比较正向模拟和反向模拟
created by ysh on 2025-08-14
"""
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, DefaultDict, Tuple
from cv2 import line
import numpy as np
import numpy.typing as npt
from collections import defaultdict
import Sofa
from scipy import sparse
import taichi as ti

from Utilize.GenMsh import mesh_obj_tri, write_mshv2_tri
from Utilize.sofa_utilize import add_move, get_marker_pos, move_desire
from _DiffPD2D import SoftObject2D
ti.init(arch=ti.gpu, debug=True, default_fp=ti.f64, device_memory_GB=8)

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

    # obj.addObject('MeshSpringForceField', name="springs", trianglesStiffness=90, trianglesDamping=0.3)
    obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='5.e2', poissonRatio='0.3', method='large')
    obj.addObject('TriangleCollisionModel')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    obj_move_list = []
    for q_i in contact:
        obj_move_list.append(obj.addObject('LinearMovementConstraint', name='cnt'+str(q_i), template="Vec3", indices=[q_i]))

    return obj, obj_move_list

def main():
    shape = [0.1, 0.1]
    contact_list = [40400]
    marker_list = [110]
    fix = range(201)

    node_np, _, ele_np = mesh_obj_tri(shape, 0.01/20)
    msh_file:str = dir_path / "Mesh/shape.msh"
    write_mshv2_tri(msh_file, node_np, ele_np)

    # ----- Setup Sofa scene -----
    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, contact_list)
    Sofa.Simulation.init(root)
    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    start_time = time.time()
    move_desire(root, move_handle, 2.0, np.array([[0.005, 0.01]] * len(contact_list)))
    print(f"Time of SOFA: {time.time() - start_time:.4f} seconds")

    params = {"E": 5.e4, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = SoftObject2D(msh_file, fix, contact_list, **params)
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    end_speed = np.array([[0.005, 0.01]]) / 2.

    start_time = time.time()
    for step in range(200):
        soft.substep(step)
        soft.contact_vel.from_numpy(end_speed)
    print(f"Time of PD: {time.time() - start_time:.4f} seconds")

if __name__ == "__main__":
    main()