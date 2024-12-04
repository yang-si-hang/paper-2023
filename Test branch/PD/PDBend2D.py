"""
使用弯曲约束和拉伸约束构建的2D Projective Dynamics模型
"""

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

