""" Construct 2d surface simulation in 3d
sofa simulation algorithm is `MeshSpring` and `TriangularBendingSpring`
created by hsy on 2025-07-22
"""
import time
import Sofa
import SofaRuntime
import Sofa.Simulation
import Sofa.Gui
from pathlib import Path
import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse import linalg as spla
import copy
import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

from _DiffPDBend import SoftBend2D
from Utilize.GenMsh import mesh_obj_tri, write_mshv2_tri, save_vtu
from Utilize.MathNp import compress_vectors
from Utilize.sofa_utilize import add_move, get_marker_pos, save_pos, move_desire
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
    root.gravity = [0., 0., -9.8]

    root.addObject('VisualStyle', displayFlags='showBehaviorModels showVisual showForceFields showInteractionForceFields showWireframe')
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('DefaultAnimationLoop', )

    root.addObject('GenericConstraintSolver', tolerance=1e-9, maxIterations=200)

    root.addObject('CollisionPipeline', depth="6", verbose="0", draw="0")
    root.addObject('BruteForceBroadPhase', )
    root.addObject('BVHNarrowPhase', )
    root.addObject('CollisionResponse', name="Response", response="PenalityContactForceField")

    root.addObject('NewProximityIntersection', name="Proximity", alarmDistance="0.5", contactDistance="0.2")
    # root.addObject('MinProximityIntersection', name='Proximity', alarmDistance=0.8, contactDistance=0.5)

    obj = root.addChild('object')
    # Rayleigh阻尼影响了软体振动
    obj.addObject('EulerImplicitSolver', name='odesolver', rayleighStiffness='0.1', rayleighMass='0.3', vdamping=0.1)
    obj.addObject('CGLinearSolver', name='linearsolver', iterations='100', tolerance='1.e-9', threshold='1.e-9')

    obj.addObject('MeshGmshLoader', name='loader', filename=f'{dir_path}/Mesh/plane_dense.msh', scale='1', flipNormals='0')
    obj.addObject('MechanicalObject', src='@loader', name='dofs', template='Vec3', translation2=[0., 0., 0.], scale3d=[1.]*3)
    obj.addObject('TriangleSetTopologyContainer', src='@loader', name='container')
    obj.addObject('TriangleSetTopologyModifier', name='modifier')
    obj.addObject('TriangleSetGeometryAlgorithms', name='geomalgo')#, tempate='Vec3')
    obj.addObject('DiagonalMass', name='mass', totalMass='0.1')#, massDensity='0.1')

    X_EPS = 2.e-3
    obj.addObject('BoxROI', name='box', box=f"-0.1 {-X_EPS} -0.1 0.11 {X_EPS} 0.1")
    obj_fixed = obj.addObject('FixedConstraint', name='fixed', indices='@box.indices')
    # obj.addObject('FixedConstraint', name='fixed2', indices='110 120')

    # obj.addObject('MeshSpringForceField', name="springs", trianglesStiffness=3, trianglesDamping=0.3)
    obj.addObject('MeshSpringForceField', name="springs", stiffness=5, damping=0.05)
    # obj.addObject('TriangularFEMForceFieldOptim', name='FEM', youngModulus='5.e2', poissonRatio='0.3', method='small')
    obj.addObject('TriangularBendingSprings', name="BS", stiffness=5, damping=0.05)
    obj.addObject('TriangleCollisionModel')
    # obj.addObject('UncoupledConstraintCorrection', defaultCompliance="0.001")

    obj_move_list = []
    for q_i in contact:
        obj_move_list.append(obj.addObject('LinearMovementConstraint', name='cnt'+str(q_i), template="Vec3", indices=[q_i], relativeMovements="1"))

    # 输出Sofa设置信息
    # Sofa.msg_info("Scene", f"Contact indices: {obj_linear_move.indices.value}")
    # Sofa.msg_info("User", f"Fixed indices: {obj_fixed.indices.value}")

    return obj, obj_move_list


class SoftBend(SoftBend2D):
    def __init__(self, shape, fix, contact, dots_list, E, nu, dt, density):
        super().__init__(shape, fix, contact, E, nu, dt, density)
        self.loss = 0.
        self.dots_idx = dots_list
        dots_num = len(dots_list)
        self.dot_pos = ti.Vector.field(3, dtype=ti.f64, shape=dots_num)
        self.dot_pos_init = ti.Vector.field(3, dtype=ti.f64, shape=dots_num)

        print(f"Marker index: {self.dots_idx}")
        self.construct_dot_pos()

    def construct_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos_init[i] = self.mesh.verts.pos_init[idx]

    def update_dot_pos(self):
        for i, idx in enumerate(self.dots_idx):
            self.dot_pos[i] = self.mesh.verts.pos[idx]

    def construct_L_sofa(self, dot_sofa:npt.NDArray):
        """ desired multi feature point on the surface
        """
        self.dL_dq_y.fill(0.)
        loss = 0.
        for i in range(self.marker_N):
            q_i = self.marker_ti[i]
            desired_pos = self.marker_pos_desired[i]
            current_pos = dot_sofa[i, :]
            self.error[i] = current_pos - desired_pos

            self.dL_dq_y[q_i*3]     = 2 * self.error[i].x
            self.dL_dq_y[q_i*3 + 1] = 2 * self.error[i].y
            self.dL_dq_y[q_i*3 + 2] = 2 * self.error[i].z

            loss += self.error[i].norm_sqr()
        return loss

    def compute_dcontact(self, dot_sofa:npt.NDArray):
        loss = self.construct_L_sofa(dot_sofa)
        self.construct_g_hessian()
        self.compute_z_act(10)

        z_np = self.z.to_numpy()
        self.dy_contact = np.multiply(z_np, self.dx_const.to_numpy())
        return loss

def main():
    gain = 5.e1
    shape = [0.1, 0.1]
    fix = range(6)
    contact = [30, 35]
    dots_list = [11, 17, 23, 29]

    # contact_sofa = [110, 120]
    # dots_sofa = [32, 54, 76, 98]
    contact_sofa = [420, 440]
    dots_sofa = [104, 188, 272, 356]

    node_np, _, ele_np = mesh_obj_tri(shape, 0.02/4)
    msh_file:str = f"{dir_path}/Mesh/plane_dense.msh"                   # sofa使用更细分的网格模型
    write_mshv2_tri(msh_file, node_np, ele_np)

    root = Sofa.Core.Node('root')
    _, move_handle = createScene(root, contact_sofa)
    Sofa.Simulation.init(root)

    # Sofa.Gui.GUIManager.Init("myscene", "qglviewer")
    # Sofa.Gui.GUIManager.createGUI(root, __file__)
    # Sofa.Gui.GUIManager.SetDimension(1080, 800)

    dt = root.dt.value
    obj = root.getChild('object')
    dofs = obj.getObject('dofs')

    # Sofa.Gui.GUIManager.MainLoop(root)
    # Sofa.Gui.GUIManager.closeGUI()
    # print("End of simulation.")
    # exit()

    move_desire(root, move_handle, 2.0, np.zeros((len(move_handle), 3)))    # 等待到达稳定状态

    sofa_pos_tmp = dofs.findData('position').value
    save_vtu(f'{dir_path}/Mesh/plane_dense.msh', sofa_pos_tmp, f'{dir_path}/Data/shape_init.vtu')

    # # Move to desired position
    # contact_desire = np.array([[-0.01, -0.03, 0.0], [-0.01, -0.03, 0.02]])
    # move_desire(root, move_handle, 3.0, contact_desire)

    # move_desire(root, move_handle, 10., np.zeros((len(move_handle), 3)))

    # sofa_pos_tmp = dofs.findData('position').value
    # save_vtu(f'{dir_path}/Mesh/plane_dense.msh', sofa_pos_tmp, f'{dir_path}/Data/shape_desire_stable.vtu')
    # exit()

    params = {"E": 1.e4, "nu": 0.4, "dt": 0.01, "density": 10.e2}
    soft = SoftBend(shape, fix, contact, dots_list, **params)
    soft.marker_pos_desired.from_numpy(np.array([[0.0995999, 0.013363, -0.0149682],
                                                 [0.0983548, 0.031184, -0.0231223],
                                                 [0.0952519, 0.0463307, -0.0122749],
                                                 [0.0924176, 0.058808, 0.00310562]]))
    # soft.marker_pos_desired.from_numpy(np.array([[0.08, 0.0, 0.0],
    #                                              [0.06, 0.0, 0.0],
    #                                              [0.04, 0.0, 0],
    #                                              [0.02, 0.0, 0]]))

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = spla.factorized(s_lhs_np)
    for step in range(200):
        soft.substep(step)

    for step in range(300):
        print(f"Time: {step*soft.dt:.2f} ======================================")
        dots_pos_sofa_new = get_marker_pos(dofs, dots_sofa)
        print(f'Detected marker position: {dots_pos_sofa_new.flatten()}')        

        sofa_pos_tmp = dofs.findData('position').value
        save_vtu(f'{dir_path}/Mesh/plane_dense.msh', sofa_pos_tmp, f'{dir_path}/Data/plane_{step:04d}.vtu')

        soft.substep(step)
        loss = soft.compute_dcontact(dots_pos_sofa_new)
        soft.update_dot_pos()
        print(f"Model marker position: {soft.dot_pos.to_numpy().flatten()}")
        print(f"Loss: {loss}")

        dy_dcontact = soft.dy_contact.reshape(-1, 3)        # reshape到与接触点个数相同
        end_speed = -gain * dy_dcontact[soft.contact_particle_list]
        end_speed_compress = compress_vectors(end_speed, 0.08)
        soft.contact_vel.from_numpy(end_speed_compress)

        print(f"End speed: {end_speed_compress.flatten()}. Norm: {np.linalg.norm(end_speed_compress, axis=1)}")
        # print(f"Sofa Node 440 pos: {dofs.findData('position').value[440]}")

        add_move(move_handle, dt, end_speed_compress * dt)
        Sofa.Simulation.animate(root, dt)

if __name__ == "__main__":
    main()