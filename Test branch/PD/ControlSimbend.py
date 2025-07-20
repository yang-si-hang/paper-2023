""" 使用taichi环境构建仿真环境, 实现2d曲面的期望点位置控制
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
from Utilize.GenMsh import mesh_obj_tri, write_obj
from Utilize.GuiTaichi import gui_set
from Utilize.MathTaichi import svd_3x2_new, cotangent_ti

