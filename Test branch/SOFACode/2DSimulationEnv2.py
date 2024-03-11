"""
Construct a 2D square simulation scene by SOFA
Apply node movement in every simulation step
"""


import Sofa
import SofaRuntime
import Sofa.Gui
import os
import numpy as np
import copy
from ControlSimulation import *

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
    root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)

    root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    root.addObject('BruteForceBroadPhase', name='BroadPhase')
    root.addObject('BVHNarrowPhase', name='NarrowPhase')
    root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    obj.addObject('MeshVTKLoader', name='loader', filename='trian.vtk', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.01')#, massDensity='0.1')

    X_EPS = 1.e-3
    obj.addObject('BoxROI', name='box', box=[-X_EPS, -0.06, -0.1, X_EPS, 0.06, 0.1])
    obj.addObject('FixedConstraint', name='fixed', indices='@box.indices')

    obj.addObject('TriangularFEMForceField', name='FEM', youngModulus='5.e5', poissonRatio='0.4', method='large')
    obj.addObject('TriangleCollisionModel')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    # Need change the indices to be equal with manipualtion index ######################################################
    obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices=[10])

    # obj_visu = obj.addChild('VisualModel')
    # obj_visu.addObject('OglModel', name='visual')
    # obj_visu.addObject('IdentityMapping', input='@..', output='@visual')


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
    manipulate_idx = 10
    marker_idx = 42

    class MyObject(SoftObject):
        def __init__(self, shape, seed_size, manipulate, marker):
            super().__init__(shape, seed_size)
            self.grasp_particle_list = manipulate
            self.marker_idx = marker
            self.marker_pos_desired[0] = self.node_init_pos[self.marker_idx] + ti.Vector([0.2, 0.])*0.01

            print('Particle number:', self.PARTICLE_NUM, '|', 'Element number:', self.ELEMENT_NUM, '|','Edge number:', self.EDGE_NUM)
            print('Fixed node idx:', self.fix_particle_list)
            print('marker node desired pos:', self.marker_pos_desired[0])


        def construct_L(self, sofa_pos):
            dim = self.dim
            idx = self.marker_idx
            desired_pos = self.marker_pos_desired[0]
            current_pos = sofa_pos
            error = current_pos - desired_pos
            L = error.norm() ** 2
            self.dL[idx * dim] = 2 * (current_pos[0] - desired_pos.x)
            self.dL[idx * dim + 1] = 2 * (current_pos[1] - desired_pos.y)
            return error, L


        def control_grasp(self, learning_rate):
            incre_ti = super().control_grasp(learning_rate)
            return incre_ti.to_numpy()


        # Without GGUI
        def substep(self, sofa_pos):
            self.construct_sn()
            self.warm_up()
            # Local sovle needs iteration
            for itr in ti.static(range(self.solve_iteration)):
                self.local_solve()
                self.construct_rhs()
                rhs_np = self.rhs.to_numpy()
                node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
                self.update_pos_new(node_pos_new_np)

            self.update_vel_pos()

            error, loss_tmp = self.construct_L(sofa_pos)
            self.loss = loss_tmp
            print('Error:', error, 'Loss:', loss_tmp)
            self.diff_pd(10)


    soft = MyObject(shape=[0.1, 0.1], seed_size=0.1/10, manipulate=[manipulate_idx], marker=marker_idx)

    print('grasp node idx:', soft.grasp_particle_list, '|','marker node idx:', soft.marker_idx)
    print('marker node initial pos:', soft.node_init_pos[soft.marker_idx])

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    loss_list = []
    marker_pos_list = []
    grasp_pos_list = []

    #######################################################################

    root = Sofa.Core.Node('root')
    createScene(root)

    Sofa.Simulation.init(root)
    Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    Sofa.Gui.GUIManager.createGUI(root, __file__)
    Sofa.Gui.GUIManager.SetDimension(1080, 800)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')
    linear_mov = obj.getObject('cnt')

    for itr in range(1, 1000, 1):
        print(f'Time：{root.time.value:.3f}---------------------------------------')
        print(f'Marker pos:{root.object.dofs.position.value[marker_idx]}')
        print(f'Grasp pos:{root.object.dofs.position.value[manipulate_idx]}')
        marker_pos = dofs.position.value[marker_idx]
        marker_pos_2d = marker_pos[0:2]

        soft.substep(marker_pos_2d)
        action_2d = soft.control_grasp(2.e0)
        action = np.append(action_2d, 0.)
        loss_list.append(soft.loss)
        marker_pos_tmp = copy.deepcopy(dofs.findData('position').value[marker_idx][0:2])
        manipulate_pos_tmp = copy.deepcopy(dofs.findData('position').value[manipulate_idx][:])
        marker_pos_list.append(marker_pos_tmp)
        grasp_pos_list.append(manipulate_pos_tmp)

        add_move(linear_mov, dt, action)

        Sofa.Simulation.animate(root, dt)

        save_pos(dofs, f'./data/pos_{itr}.csv')

    np.savetxt('loss.csv', np.array(loss_list), fmt='%e', delimiter=',')
    np.savetxt('marker_pos.csv', np.array(marker_pos_list), fmt='%e', delimiter=',')
    np.savetxt('grasp_pos.csv', np.array(grasp_pos_list), fmt='%e', delimiter=',')

    times = linear_mov.keyTimes.value
    movements = linear_mov.movements.value
    # print(f'{times}, {movements}')

    # Sofa.Gui.GUIManager.MainLoop(root)
    Sofa.Gui.GUIManager.closeGUI()
    print("End of simulation.")


if __name__ == '__main__':
    main()