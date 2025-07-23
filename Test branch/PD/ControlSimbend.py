""" 使用taichi环境构建仿真环境, 实现2d曲面的期望点位置控制、多点控制以及曲率控制
created by hsy on 2025-07-20
"""
import os
import sys
import time
from typing import List, Dict, DefaultDict, Tuple
import numpy as np
import numpy.typing as npt
from collections import defaultdict
from scipy import sparse
from scipy.sparse import linalg as spla
import taichi as ti
import meshtaichi_patcher as Patcher
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

root_path = os.path.abspath(os.path.join(script_dir, '..')) # 添加根目录到 sys.path（跨目录导入模块）
sys.path.append(root_path)
from Utilize.MathNp import compress_vectors

from _DiffPD2dBend import SoftBend2D

def main():
    gain = 2.e2

    soft = SoftBend2D([0.1, 0.1], 1.e4, 0.4, 0.01, 10e2)
    # soft.preset_gui([-0.2, 0.05, 0.15], [0.05, 0.1, 0.], [0., 0., 1.])
    soft.preset_gui([0.2, 0.07, 0.15], [0.05, 0.05, 0.], [0., 0., 1.])

    soft.marker_list = [11, 17, 23, 29]
    soft.marker_ti.from_numpy(np.array(soft.marker_list).astype(np.int32))

    soft.precomputation()
    lhs_np = soft.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft.pre_fact_lhs_solve = spla.factorized(s_lhs_np)

    for itr in range(200):
        soft.substep(itr)

    marker_desired_np = np.array([[0.10046816,  0.00942272, -0.01836638],
                                  [0.10185394,  0.02874743, -0.02520925],
                                  [0.1014244,   0.04679911, -0.01465023],
                                  [0.09991732,  0.06291499,  0.0003617]], dtype=np.float64)
    soft.marker_pos_desired.from_numpy(marker_desired_np)
    soft.node_desired.from_numpy(marker_desired_np.astype(np.float32))

    end_speed_compress = np.zeros((soft.CON_N, 3), dtype=np.float64)
    # loss_list = []
    for itr in range(200):
        print(f"Simulation Time: {itr*soft.dt:.2f} ======================================")
        soft.substep(itr)
        # time.sleep(0.05)
        loss_tmp = soft.compute_dL_dy()
        dL_dy = soft.dL_dy.reshape(-1, 3)
        for i in range(soft.CON_N):
            end_speed = -gain * dL_dy[soft.contact_particle_list[i]]
            end_speed_compress[i, :] = compress_vectors(end_speed, 0.05)
        soft.contact_vel.from_numpy(end_speed_compress.astype(np.float64))
        print(f"End speed (compressed): {end_speed_compress.flatten()}")
        print(f"Stretch energy: {np.sum(soft.stretch_energy.to_numpy()):e}")
        print(f"Bend energy: {np.sum(soft.bend_energy.to_numpy()):e}")

        # loss_list.append(loss_tmp)

        soft.gui_show(True, True, itr)

    # np.savetxt("Data/loss_list.csv", loss_list, fmt='%e', delimiter=",")

if __name__ == "__main__":
    main()