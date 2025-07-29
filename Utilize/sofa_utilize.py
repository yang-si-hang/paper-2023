""" Sofa中的一些通用函数
created by hsy on 2025-07-22
"""
import numpy as np
import numpy.typing as npt
import copy
import meshio
import Sofa

from .GenMsh import read_mshv2_triangle

def add_move(handle_list:list, dt:float, movement:npt.NDArray):
    """ Use `LinearMovementConstraint` to add a simulation step-wise movement
    
    Args:
        handle: The node of the object
        dt: The time step
        movement: The additional movement
    """
    if movement.shape[1] == 2:
        movement = np.concatenate((movement, np.zeros((movement.shape[0], 1))), axis=1)
    for i, handle in enumerate(handle_list):
        times_array = handle.findData('keyTimes').value
        movements_array = handle.findData('movements').value

        last_time = times_array[-1]
        last_movement = movements_array[-1, :]

        handle.findData('keyTimes').value = np.append(times_array, last_time + dt)
        handle.findData('movements').value = np.append(movements_array, [movement[i,:] + last_movement], axis=0)

def move_desire(root, handle_list:list, time:float, desire:npt.NDArray):
    """ Use `LinearMovementConstraint` to add a simulation step-wise desired movement

    Args:
        root: The root node of the Sofa scene
        handle_list: The list of nodes to be moved
        time: total time
        desire: The desired movement
    """
    dt = root.dt.value
    for step in range(int(time/dt)):
        mov_step = desire * dt / time   # 线性插值
        add_move(handle_list, dt, mov_step)
        Sofa.Simulation.animate(root, dt)   # 一定要放在循环最后

def get_marker_pos(handle, marker_idx:list)->npt.NDArray:
    """从sofa中获取指定节点的位置
    """
    marker_pos = np.zeros((len(marker_idx), 3))
    # node_pos = handle.findData('position').value
    for i, idx in enumerate(marker_idx):
        pos_tmp = copy.deepcopy(handle.findData('position').value[idx])
        marker_pos[i] = pos_tmp
    return marker_pos

def save_vtu(mesh_file:str, pos:npt.NDArray, write_name:str):
    """Save the node position to a .vtu file

    Args:
        mesh_file (str): The initial mesh file name
        pos (npt.NDArray): The node position
        write_name (str): The write file name
    """
    _, triangles = read_mshv2_triangle(mesh_file)

    cells_write = [("triangle", triangles)]
    mesh = meshio.Mesh(points=pos, cells=cells_write)
    mesh.write(f"{write_name}")

def save_msh(mesh_file:str, pos:npt.NDArray, write_name:str):
    pass

def save_pos(handle, path):
    node_pos = handle.findData('position').value
    np.savetxt(f'{path}', node_pos, '%.6f')