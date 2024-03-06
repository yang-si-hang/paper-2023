"""
Generate a Tetrahedral mesh file in .vtk with given meshed data
"""

import numpy as np
import pygmsh
import meshio
import vtk
import os
script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


def mesh_data(shape, seed_size):
    L = shape[0]
    W = shape[1]
    H = 0.02
    center_pos = [0., 0., 0.]

    with pygmsh.geo.Geometry() as geom:
        geom.add_box(0., L, 0., W, 0., H, mesh_size=seed_size)
        mesh = geom.generate_mesh()
        meshio.write('mesh.vtk', mesh, file_format='vtk', binary=False)


def main():
    mesh_data(shape=[0.1, 0.1], seed_size=0.01)


if __name__ == '__main__':
    main()