"""
使用liver.msh作为几何网格文件,控制其中某个节点的运动,以到达期望位置,并记录该节点的位置
created at 2024-09-15 by hsy
"""
import Sofa
import SofaRuntime
import Sofa.Gui
import Sofa.SofaGL
import os, sys
import numpy as np
import numpy.typing as npt
import copy
import meshio
from DiffPDModelLiver import *

script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)                                            # 改变当前工作目录到脚本文件所在目录

root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import read_elements_from_msh2


POINTS_NUM = 1

def action_compress(vec:npt.NDArray, max_length:float=3.e-4)->npt.NDArray:
    """Compress action vector in a safe range
    Args:
        vec (npt.NDArray): the action vector
        max_length (float, optional): the maximum length of the action vector. Defaults to 3.e-4.
    
    Returns:
        vec (npt.NDArray): the compressed action vector
    """
    length = np.linalg.norm(vec)

    if length > max_length:
        factor = max_length / length
        return vec * factor
    else:
        return vec


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

    # obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='3 39 64')
    obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='15 83 80 16 78 17 77 11 79 10 18 7 19 23 20 24 125 59 68 63 57 60 64 66 65 67 124 9 12 8')
    obj_fixed.drawSize = 0.005

    obj.addObject('TetrahedronFEMForceField', name='FEM', youngModulus='5.e4', poissonRatio='0.45', method='large')
    obj.addObject('TriangleCollisionModel')
    # obj.addObject('LinearSolverConstraintCorrection', solverName='@linearsolver')
    obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    # obj_linear_move = obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices=[85])
    obj_linear_move = obj.addObject('LinearMovementConstraint', name='cnt', template="Vec3", indices=[101])

    visu = obj.addChild('visual')
    visu.addObject('MeshOBJLoader', name='loader', filename='Mesh/liver-smooth.obj', scale='0.05', flipNormals='0')
    visu.addObject('OglModel', name='visual_model', src='@loader', color=[1, 0, 0, 1])
    visu.addObject('BarycentricMapping', name='visual_mapping', input='@../dofs', output='@visual_model')

    surf = obj.addChild('surface')

    # 输出Sofa设置信息
    Sofa.msg_info("Scene", f"Contact indices: {obj_linear_move.indices.value}")
    Sofa.msg_info("Scene", f"Fixed indices: {obj_fixed.indices.value}")


def add_move(handle, dt, movement):
    """Use `LinearMovemetConstraint` to add a simulation step-wise movement

    Args:
        handle (_type_): The node of the object
        dt (_type_): _description_
        movement (_type_): The additional movement
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


def save_pos(handle, path):
    node_pos = handle.findData('position').value
    np.savetxt(f'{path}', node_pos, '%.6f', delimiter=',')


def save_vtu(mesh_file:str, pos:npt.NDArray, write_name:str):
    """Save the node position to a .vtu file

    Args:
        mesh_file (str): The initial mesh file name
        pos (npt.NDArray): The node position
        write_name (istrnt): The write file name
    """
    _, eles = read_elements_from_msh2(mesh_file)
    cells = np.array([node_ids for (_, node_ids) in eles]) - 1

    cells_write = [("tetra", cells)]
    mesh = meshio.Mesh(points=pos, cells=cells_write)
    mesh.write(f"data/{write_name}")


class MyObject(SoftObject):
    def __init__(self, shape, seed_size, file, dots_idx, contact_list):
        super().__init__(shape, seed_size, file, contact_list)
        self.dt = 1. / 100
        self.dots_idx = dots_idx
        self.loss = np.zeros(POINTS_NUM)
        self.dot_pos = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_desired = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)

        self.dL_dy_m = ti.Vector.field(3, dtype=ti.f64, shape=self.PARTICLE_NUM)
        self.read_desired_pos()

        print(f"Marker index: {self.dots_idx}")


    def read_desired_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos_init[i] = self.node_init_pos[idx]
            self.dot_pos[i] = self.node_init_pos[idx]
            self.dot_pos_desired[i] = self.node_init_pos[idx] + ti.Vector([-0.04, 0.06, 0.01])
        print(f"Initial position: {self.dot_pos_init.to_numpy()}")
        print(f"Desired position: {self.dot_pos_desired.to_numpy()}")


    def update_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos[i] = self.node_pos[idx]


    def construct_L_sofa(self, dot_sofa:npt.NDArray):
        """_summary_

        Args:
            dot_sofa (npt.NDArray): 从SOFA中读取的节点位置

        Returns:
            Tuple:
                - error (npt.NDArray): 误差
                - loss (npt.NDArray): 损失
        """
        dim = self.dim
        error = np.zeros((POINTS_NUM, dim), dtype=np.float64)
        self.dL_dq.fill(0.)
        for i, idx in enumerate(self.dots_idx):
            desired_pos = self.dot_pos_desired[i]
            current_pos = dot_sofa[i]
            error_ti = current_pos - desired_pos
            error[i] = error_ti.to_numpy()
            for d in range(dim):
                self.dL_dq[idx*dim+d] += 2 * error_ti[d]
            
        return error, np.linalg.norm(error, axis=1)**2


    def diff_pd(self, itr_num:int):
        self.partial_p()
        dA = self.rhs_dA.to_numpy()
        par_L = self.dL_dq.to_numpy()
        z_np = self.z.to_numpy()
        for itr in ti.static(range(itr_num)):
            rhs_diff_np = dA @ z_np + par_L
            z_new_np = self.pre_fact_lhs_solve(rhs_diff_np)
            z_np = z_new_np
        self.z.from_numpy(z_np)


    @ti.kernel
    def compute_grad_y(self):
        for idx in range(self.PARTICLE_NUM):
            idx1, idx2, idx3 = idx*self.dim, idx*self.dim+1, idx*self.dim+2
            self.dL_dy_m[idx][0] = self.z[idx1] * self.dx_const[idx1]
            self.dL_dy_m[idx][1] = self.z[idx2] * self.dx_const[idx2]
            self.dL_dy_m[idx][2] = self.z[idx3] * self.dx_const[idx3]


    def substep(self, step_num:int):
        self.construct_desired_pos()
        self.construct_sn()
        self.warm_start()
        for itr in ti.static(range(self.solve_iteration)):
            self.local_solve()
            self.construct_rhs()
            rhs_np = self.rhs.to_numpy()
            node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
            self.update_pos_new(node_pos_new_np)

        self.update_vel_pos()
        self.update_dot_pos()
        # self.gui_show(self.window, self.canvas, self.scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=step_num)


    def actuate_action(self, contact_speed):
        self.contact_vel[0] = contact_speed


    def compute_gradient(self, dot_sofa:npt.NDArray):
        error, loss_tmp = self.construct_L_sofa(dot_sofa)
        self.loss = loss_tmp
        self.diff_pd(10)
        self.compute_grad_y()

        return error


def main():
    # 测试过的接触点数据: 101, 102
    contact_idx = [101]
    marker_idx = [0]
    fix_idx = [39, 61, 63, 64]
    fixed_list = [15, 83, 80, 16, 78, 17, 77, 11 ,79, 10, 18, 7, 19, 23, 20, 24, 
                  125, 59, 68, 63, 57, 60, 64, 66, 65, 67, 124, 9, 12, 8]
    obj_shape = [0.1, 0.02, 0.1]          # 作为一个占位符，此处无用
    learning_rate = 8.e1

    soft_obj = MyObject(obj_shape, 0.01, 'Mesh/liver.msh', marker_idx, contact_idx)
    # soft_obj.preset_gui(camera_pos=[0.1, 0.2, 0.05], camera_target=[-0.1, 0., 0.05])

    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csr_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

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

    dots_pos_sofa_init = get_marker_pos(dofs, marker_idx)

    contact_pos_np = np.zeros((len(contact_idx), 3))
    dots_pos_sofa = dots_pos_sofa_init
    dots_pos_model = soft_obj.dot_pos.to_numpy()
    # print(f"dot pos: {dots_pos_model}")
    dots_sofa_list = []
    delta_pos_list = []
    delta_pos_model_list = []
    loss_list = []
    rob_movement_list = []
    contact_pos_list = []
    sofa_contact_pos_list = []

    for step in range(150):
        print(f"Step: {step}---------------------------------------")
        dots_pos_sofa_new = get_marker_pos(dofs, marker_idx)
        print(f'Detected marker position: {dots_pos_sofa_new}')

        sofa_pos_tmp = dofs.findData('position').value
        save_vtu('Mesh/liver.msh', sofa_pos_tmp, f'sofa_pos_{step:04d}.vtu')

        delta_pos = dots_pos_sofa_new - dots_pos_sofa
        dots_pos_sofa = dots_pos_sofa_new

        soft_obj.substep(1)
        dots_pos_model_new = soft_obj.dot_pos.to_numpy()
        delta_pos_model = dots_pos_model_new - dots_pos_model
        dots_pos_model = copy.deepcopy(dots_pos_model_new)
        error_tmp = soft_obj.compute_gradient(dots_pos_sofa_new)
        loss_tmp = np.linalg.norm(error_tmp) ** 2
        print(f"Model marker position: {dots_pos_model}")

        node_pos_tmp = soft_obj.node_pos.to_numpy()
        # save_vtu('Mesh/liver.msh', node_pos_tmp, f'model_pos_{step}.vtu')

        end_speed_np = -learning_rate * soft_obj.dL_dy_m[soft_obj.contact_particles_list[0]].to_numpy()
        end_movement_np = action_compress(end_speed_np * soft_obj.dt, 1.5e-3)
        # end_movement_np = np.array([0., 0., 0.])

        soft_obj.actuate_action(end_movement_np / soft_obj.dt)
        print(f'Error items: {error_tmp}; Loss sum: {loss_tmp}')
        print(f'The tool movement: {end_movement_np.tolist()}; movement length: {np.linalg.norm(end_movement_np)}')

        add_move(linear_mov, dt, end_movement_np)
        Sofa.Simulation.animate(root, dt)

        for i, q_idx in enumerate(contact_idx):
            contact_pos_np[i, :] = copy.deepcopy(dofs.findData('position').value[q_idx])
            print(f'Contact position {q_idx}: {contact_pos_np[i, :]}')

        # 写入数据
        dots_sofa_list.append(dots_pos_sofa.flatten())
        delta_pos_list.append(delta_pos.flatten())
        delta_pos_model_list.append(delta_pos_model.flatten())
        loss_list.append(loss_tmp)
        rob_movement_list.append(end_movement_np.tolist())
        contact_pos_list.append(soft_obj.node_pos[contact_idx[0]].to_numpy())       # 对于只有一个接触点的情况
        sofa_contact_pos_list.append(contact_pos_np.flatten())

    np.savetxt('dots_sofa_list.csv', np.array(dots_sofa_list), fmt='%.10f', delimiter=',')
    np.savetxt('delta_pos_list.csv', np.array(delta_pos_list), fmt='%.10f', delimiter=',')
    np.savetxt('delta_pos_model_list.csv', np.array(delta_pos_model_list), fmt='%.10f', delimiter=',')
    np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
    np.savetxt('rob_movement_list.csv', np.array(rob_movement_list), fmt='%.10f', delimiter=',')
    np.savetxt('contact_pos_list.csv', np.array(contact_pos_list), fmt='%.10f', delimiter=',')
    np.savetxt('sofa_contact_pos_list.csv', np.array(sofa_contact_pos_list), fmt='%.10f', delimiter=',')
    # save_pos(dofs, script_dir + '/node_end_pos.csv')

    # Sofa.Gui.GUIManager.MainLoop(root)
    # Sofa.Gui.GUIManager.closeGUI()
    # print("End of simulation.")

if __name__ == '__main__':
    main()