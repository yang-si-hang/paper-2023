# The program bases the article Quasi-Newton Methods for Real-Time Simulation of Hyperelastic
# Materials

import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, default_fp=ti.f64, debug=True)

dim = 2

