"""
只使用拉伸约束模拟2D材料的3D空间弯曲形变
created by hsy on 2025-7-15
"""
import os
import sys
import time
from typing import List, Dict, DefaultDict, Tuple
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
import taichi as ti
import meshtaichi_patcher as Patcher
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.GenMsh import mesh_obj_tri, write_obj

def read_mesh():
    return

@ti.data_oriented
class PDPointBend2D:
    def __init__(self, shape, E, nu, dt):
        if isinstance(shape, str):
            nodes, edges, faces = read_mesh(shape)
        else:
            nodes, edges, faces = mesh_obj_tri(shape, 0.01)

        nodes = np.hstack((nodes, np.zeros((nodes.shape[0], 1))))

        obj_file = "Mesh/plane.obj"
        write_obj(obj_file, nodes, faces)

        self.solve_itr:int = 10
        self.strain_lim_rate:float = 0.1
        self.E, self.nu, self.dt, self.density, self.g = E, nu, dt, 1.e3, -9.8
        self.dim = 3
        self.mu, self.lam = self.E / (2 * (1 + self.nu)), self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
        
        self.mesh = Patcher.load_mesh(obj_file, relations=[])

        self.mesh.verts.place({
            "pos": ti.types.vector(3, ti.f64),
            "pos_init": ti.types.vector(3, ti.f64),
            "vel": ti.types.vector(3, ti.f64),
        }, reorder=False)
        self.mesh.edges.place({
            "border": bool,
        }, reorder=False)
        self.mesh.faces.place({
            "area": ti.f64,
        }, reorder=False)

    @ti.kernel
    def construct_mass(self):
        