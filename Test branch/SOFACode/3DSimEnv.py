"""
在SOFA中创建3D仿真环境
created at 2024-10-21 by hsy
"""
import copy
import Sofa
import SofaRuntime
import Sofa.Gui
import os
import numpy as np
from DiffPDModel3d import *

script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录

points_num:int = 1

class MyObject(SoftObject):
    def __init__(self, shape:list, seed_size:list, mesh_file:str, contact_list:list, fixed_list:list):
        super().__init__(shape, seed_size, mesh_file, contact_list)
        self.dt = 1. / 100
        self.loss = np.zeros(points_num, dtype=np.float64)
        # 注意此处的与`SoftObject`中需要一致，暂时不支持动态改变
        self.fixed_list = fixed_list

        self.dot_pos = ti.Vector.field(3, dtype=ti.f64, shape=points_num)
        self.dot_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=points_num)
        self.dot_pos_desired = ti.Vector.field(3, dtype=ti.f64, shape=points_num)

        # 直接取指定节点
        self.marker_list = [121]
        dot_pos_init = np.expand_dims(self.node_init_pos[121].to_numpy(), axis=0)
        self.dot_pos_init.from_numpy(dot_pos_init)
        self.dot_pos.from_numpy(dot_pos_init)

        self.define_desired_pos()


    def define_desired_pos(self):
        for i in range(points_num):
            self.dot_pos_desired[i] = self.dot_pos_init[i] + ti.Vector([0., 0., 0.03])


    def get_marker_model_pos(self):
        for i in range(points_num):
            self.dot_pos[i] = self.node_pos[self.marker_list[i]]


    def construct_L_model(self):
        dim:int = self.dim
        error = np.zeros((points_num, dim), dtype=np.float64)
        self.dL_dq.fill(0.)
        for i, idx in enumerate(self.marker_list):
            desired_pos = self.dot_pos_desired[i]
            current_pos = self.dot_pos[i]
            error[i] = (desired_pos - current_pos).to_numpy()
            self.dL_dq[idx*dim+0] += 2 * (current_pos[0] - desired_pos[0])
            self.dL_dq[idx*dim+1] += 2 * (current_pos[1] - desired_pos[1])
            self.dL_dq[idx*dim+2] += 2 * (current_pos[2] - desired_pos[2])

        return error, np.linalg.norm(error, axis=1) ** 2


    def construct_L_sofa(self, dot_sofa:npt.NDArray):
        dim:int = self.dim
        error = np.zeros((points_num, dim), dtype=np.float64)
        self.dL_dq.fill(0.)
        for i, idx in enumerate(self.marker_list):
            desired_pos = self.dot_pos_desired[i]
            current_pos = dot_sofa[i]
            error[i] = (desired_pos - current_pos).to_numpy()
            self.dL_dq[idx*dim+0] += 2 * (current_pos[0] - desired_pos[0])
            self.dL_dq[idx*dim+1] += 2 * (current_pos[1] - desired_pos[1])
            self.dL_dq[idx*dim+2] += 2 * (current_pos[2] - desired_pos[2])

        return error, np.linalg.norm(error, axis=1) ** 2


    def diff_pd(self, itr_num:int):
        self.partial_p()
        dA = self.rhs_dA.to_numpy()
        z_np = self.z.to_numpy()
        par_L = self.dL_dq.to_numpy()
        for itr in ti.static(range(itr_num)):
            rhs_diff_np = dA @ z_np + par_L
            z_new_np = self.pre_fact_lhs_solve(rhs_diff_np)
            z_np = z_new_np
        self.z.from_numpy(z_np)


    def substep(self, step_num:int=1):
        self.construct_sn()
        self.warm_start()
        for itr in range(self.solve_iteration):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        self.get_marker_model_pos()


    def compute_gradient(self, dot_sofa:npt.NDArray):
        error, loss_tmp = self.construct_L_sofa(dot_sofa)
        self.loss = loss_tmp
        self.diff_pd(10)
        self.cal_ygrad()            # 得到dL/dy

        return error, loss_tmp



def createScene(root):
    root.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                 'Sofa.Component.Collision',
                                                 'Sofa.Component.Constraint.Projective',
                                                 'Sofa.Component.IO.Mesh',
                                                 'Sofa.Component.LinearSolver',
                                                 'Sofa.GL.Component.Rendering3D',
                                                 'Sofa.Component.Topology.Container.Dynamic'])
    root.dt = 0.01
    root.bbox = [[-0.7, -0.7, -0.2], [0.7, 0.7, 0.2]]
    root.gravity = [0., 0., 0.]

    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=100)

    root.addObject('CollisionPipeline', name='Pipeline', verbose='0')
    root.addObject('BruteForceBroadPhase', name='BroadPhase')
    root.addObject('BVHNarrowPhase', name='NarrowPhase')
    root.addObject('CollisionResponse', name='Response', response='PenalityContactResponse')
    root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.1')
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='200', tolerance='1.e-9', threshold='1.e-9')

    obj.addObject('MeshGmshLoader', name='loader', filename='Mesh/cube_new.msh', scale='1.', rotation='0 0 0', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TetrahedronSetTopologyContainer', name="tetratopo", src='@loader')
    obj.addObject('TetrahedronSetTopologyModifier', name="modifier")
    obj.addObject('TetrahedronSetGeometryAlgorithms', template="Vec3", name="geomalgo")
    obj.addObject('MeshMatrixMass', totalMass="0.0325", name="SparseMass", topology="@tetratopo")
    obj.addObject('TetrahedronFEMForceField', name="FEM", youngModulus="1000", poissonRatio="0.3",
                  computeGlobalMatrix="false", method="large", computeVonMisesStress="2", showVonMisesStressPerElement="true")
    obj.addObject('TetrahedronCollisionModel')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    obj.addObject('FixedConstraint', template='Vec3', indices='206 207 208 209 228 229 230 231')

    # obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", showMovement=False, indices='10 11 12 13 32 33 34 35')

    # obj_visu = obj.addChild('VisualModel')
    # obj_visu.loader = obj_visu.addObject('MeshGmshLoader', name='loader', filename='Mesh/cube_new2.msh', triangulate='true', scale=1.)
    # obj_visu.addObject('OglModel', name='model', src='@loader', scale3d=[1.]*3, color=[1., 0., 0.8], updateNormals=False)
    # obj_visu.addObject('BarycentricMapping', input="@..", output="@model")


def get_marker_pos_sofa(handle, indices:list):
    # 获取SOFA中对象的某一些节点的位置
    pos = np.zeros((len(indices), 3), dtype=np.float64)
    for i, idx in enumerate(indices):
        pos[i] = handle.position.value[idx]

    return pos


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
    np.savetxt(f'{path}', node_pos, '%.8f', delimiter=',')


def main():
    obj_shape = [0.5, 0.5, 0.05]
    obj_seed_size = 0.05
    learning_rate = 5.e-1
    contact_list = [10, 11, 12, 13] + [32, 33, 34, 35]
    fixed_list = [206, 207, 208, 209] + [228, 229, 230, 231]

    soft = MyObject(obj_shape, obj_seed_size, 'Mesh/cube_new.msh', contact_list, fixed_list)
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    print(f"Contact nodes indices: {soft.contact_particles_list}; Fixed nodes indices: {soft.fix_particle_list}")
    print(f"Marker node initial pos: {soft.dot_pos_init}; Marker node desired pos: {soft.dot_pos_desired}")

    root = Sofa.Core.Node('root')
    createScene(root)
    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    cnt_mov_handle = []
    for i, idx in enumerate(soft.contact_particles_list):
        obj.addObject('LinearMovementConstraint', name=f'cnt{i:d}', template="Vec3", showMovement=False, indices=f'{idx:d}')
        cnt_mov_handle.append(obj.getObject(f'cnt{i:d}'))
    Sofa.Simulation.init(root)

    # Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    # Sofa.Gui.GUIManager.createGUI(root, __file__)
    # Sofa.Gui.GUIManager.SetDimension(1080, 800)

    loss_list = []
    marker_pos_list = []
    rob_mov_list = []
    contact_pos_list = []

    for itr in range(1, 200, 1):
        print(f"Time：{root.time.value:.3f}---------------------------------------")
        marker_pos = get_marker_pos_sofa(dofs, soft.marker_list)
        contact_pos = copy.deepcopy(dofs.findData('position').value[soft.contact_particles_list])
        print(f"Sofa marker pos: {marker_pos}")
        soft.substep()
        error_tmp, loss_tmp = soft.compute_gradient(marker_pos)
        print(f"Error: {error_tmp}; Loss: {loss_tmp}")

        drob = soft.compute_action()
        compressed_action = action_compress(-drob*learning_rate, 8.e-3)
        soft.apply_action(compressed_action)

        for i in range(soft.contact_num):
            mov_tmp = soft.contact_vel[i].to_numpy() * soft.dt
            add_move(cnt_mov_handle[i], dt, mov_tmp.tolist())
            print(f"Node {soft.contact_particles_list[i]} movement: {mov_tmp}")
        Sofa.Simulation.animate(root, dt)

        loss_list.append(loss_tmp)
        marker_pos_list.append(marker_pos.flatten())
        contact_pos_list.append(contact_pos.flatten())
        rob_mov_list.append(compressed_action)

    save_pos(dofs, 'node_pos_final.csv')

    # Sofa.Gui.GUIManager.MainLoop(root)
    # Sofa.Gui.GUIManager.closeGUI()
    print("End of simulation.")

    np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
    np.savetxt('marker_pos_list.csv', np.array(marker_pos_list), fmt='%.10f', delimiter=',')
    np.savetxt('contact_pos_list.csv', np.array(contact_pos_list), fmt='%.10f', delimiter=',')
    np.savetxt('rob_mov_list.csv', np.array(rob_mov_list), fmt='%.10f', delimiter=',')

if __name__ == '__main__':
    main()