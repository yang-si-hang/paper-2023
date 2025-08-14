""" 使用DiffPD可微模型完成多特征点控制
基于Sofa环境
created by ysh on 2025-08-11
"""
from pathlib import Path
from typing import List
import numpy.typing as npt
import Sofa
import SofaRuntime
import Sofa.Gui
import taichi as ti
import numpy as np
from scipy import sparse
np.set_printoptions(linewidth=150)

from SOFACode._DiffPD2D import SoftObject2D
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

class MyObject(SoftObject2D):
    def __init__(self, shape, fix, contact, dots_list, E, nu, dt, density, **kwargs):
        super().__init__(shape, fix, contact, E, nu, dt, density, **kwargs)
        self.loss = 0.
        self.dots_idx = dots_list
        dots_num = len(dots_list)
        self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=dots_num)
        self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=dots_num)
        self.dots_pos_target = None

        print(f"Marker index: {self.dots_idx}")

        self.construct_dot_pos()
        self.update_dot_pos()

    def construct_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos_init[i] = self.node_pos_init[idx]
    

    def update_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos[i] = self.node_pos[idx]

    def construct_L_sofa(self, dots_sofa:npt.NDArray):
        """ multi-marker points control"""
        self.dL_dq_contact.fill(0.)
        self.loss = 0.
        for i, idx in enumerate(self.dots_idx):
            error = dots_sofa[i, :] - self.dots_pos_target[i, :]
            self.loss += np.sum(error**2)
            self.dL_dq_contact[idx*self.dim] = 2 * error[0]
            self.dL_dq_contact[idx*self.dim + 1] = 2 * error[1]
        return self.loss

    def compute_dcontact(self, dots_sofa:npt.NDArray):
        """ \partial L / \partial y with contact action, 计算关于contact的导数, 用于控制
        """
        loss_tmp = self.construct_L_sofa(dots_sofa)
        self.construct_g_hessian()
        self.compute_z(10)

        # print(f"Loss: {loss_tmp}")

        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())
        return loss_tmp

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
    rest_contact_pos = dofs.findData('position').value[contact_list, :].squeeze()

    move_desire(root, move_handle, 2.0, np.array([[0.005, 0.01]] * len(contact_list)))
    dots_pos_target = get_marker_pos(dofs, marker_list)
    print(f"Target marker position: {dots_pos_target[:, :2].flatten()}")

    Sofa.Simulation.reset(root)
    # 清空LinearMovementConstraint
    for handle in move_handle:
        handle.findData('keyTimes').value = np.array([0.])
        handle.findData('movements').value = np.array([[0., 0., 0.]])

    params = {"E": 5.e4, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = MyObject(shape, fix, contact_list, marker_list, **params)
    soft.dots_pos_target = dots_pos_target[:, :2]
    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    gain = 1.e0

    dots_pos_soft = get_marker_pos(dofs, marker_list)[:, :2]
    dots_pos_model = soft.dot_pos_init.to_numpy()
    loss_list = []
    delta_pos_list = []
    delta_pos_model_list = []

    for step in range(200):
        print(f"Time Step: {step*0.01:.2f} ======================================")
        dots_pos_sofa_new = get_marker_pos(dofs, marker_list)[:, :2]
        print(f'Detected marker position: {dots_pos_sofa_new.flatten()}')

        delta_pos = dots_pos_sofa_new - dots_pos_soft
        dots_pos_soft = dots_pos_sofa_new

        soft.substep(step)
        soft.update_dot_pos()
        dots_pos_model_new = soft.dot_pos.to_numpy()
        delta_pos_model = dots_pos_model_new - dots_pos_model
        dots_pos_model = dots_pos_model_new
        loss_tmp = soft.compute_dcontact(dots_pos_sofa_new)
        print(f"Model marker position: {soft.dot_pos.to_numpy().flatten()}")
        print(f"Loss: {loss_tmp}")

        dy_dcontact = soft.dy_contact.reshape(-1, 2)
        end_speed = -gain * dy_dcontact[soft.contact_particle_list]
        end_speed_compress = compress_vectors(end_speed, 2.e-4)
        soft.contact_vel.from_numpy(end_speed_compress / soft.dt)
        print(f"End speed: {end_speed_compress}; {np.linalg.norm(end_speed_compress)}")
        print(f"Contact pos: {get_marker_pos(dofs, contact_list)[:, :2]}")

        add_move(move_handle, dt, end_speed_compress)
        Sofa.Simulation.animate(root, dt)

        loss_list.append(loss_tmp)
        delta_pos_list.append(delta_pos.flatten())
        delta_pos_model_list.append(delta_pos_model.flatten())

    np.savetxt(f"{dir_path}/Data/loss_list_model_{contact_list[0]}.csv", loss_list, fmt="%e", delimiter=",")
    np.savetxt(f'{dir_path}/Data/delta_pos_{contact_list[0]}.csv', np.array(delta_pos_list), fmt='%e', delimiter=',')
    np.savetxt(f"{dir_path}/Data/delta_pos_model_{contact_list[0]}.csv", np.array(delta_pos_model_list), fmt="%e", delimiter=",")

if __name__ == "__main__":
    main([119], [89, 101, 102, 107])

    # candidate_contact = [110, 120, 111, 112, 114, 115, 116, 117, 119]
    # candidate_contact = [109, 117, 118, 119, 120]
    # for contact in candidate_contact:
    #     main([contact], [89, 101, 102])