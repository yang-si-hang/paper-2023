"""
Generate a mesh file with given meshed data
"""


import numpy as np
from scipy.spatial import Delaunay
import os
script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
os.chdir(script_dir)            # 改变当前工作目录


def mesh_data(shape, seed_size):
    L = shape[0]
    W = shape[1]
    # If the shape can be divided by seed_size, the remainder is 1, otherwise 0
    LN_remain = int(1) if np.mod(L, seed_size) < 1.e-8 else int(0)  # 1e-8 due to the precision problem
    WN_remain = int(1) if np.mod(W, seed_size) < 1.e-8 else int(0)
    LN = int(np.ceil(L / seed_size)) + LN_remain
    WN = int(np.ceil(W / seed_size)) + WN_remain

    xx, yy = np.meshgrid(np.linspace(0, L, LN), np.linspace(0, W, WN))
    xx_pad = xx.flatten()
    yy_pad = yy.flatten()
    node = np.array([xx_pad, yy_pad]).T

    tri = Delaunay(node)
    
    element = tri.simplices
    element += 1

    edge_set = set()
    for simplices in element:
        for i in range(3):
            edge_tmp = tuple(sorted([i, (i + 1) % 3]))
            edge_set.add(edge_tmp)

    edge = np.array(list(edge_set))

    data = {'v': node, 'e': edge, 'f': element}

    return data


def mesh_gen(data, filename):
    vertices = data['v']
    faces = data['f']
    height = 0.
    with open(filename, 'w') as file:
        for v in vertices:
            file.write(f'v {v[0]:.6f} {height:.6f} {v[1]:.6f}\n')
        for f in faces:
            file.write(f"f {f[0]} {f[1]} {f[2]}\n")


def main():
    data = mesh_data(shape=[0.1, 0.1], seed_size=0.01)
    mesh_gen(data, filename="object.obj")


if __name__ == "__main__":
    main()