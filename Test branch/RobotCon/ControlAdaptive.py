"""
使用Adaptive Controller的方法控制软体上的图案变形
created at 2024-09-02 by hsy
"""


import numpy as np


POINTS_NUM = 4

j_init = np.eye(2*POINTS_NUM, 2)
a = j_init.flatten()

delta_action = np.zeros(2)
W = np.zeros((2*POINTS_NUM, 4*POINTS_NUM))
for idx in range(2*POINTS_NUM):
    W[idx, 2*idx] = delta_action[0]
    W[idx, 2*idx+1] = delta_action[1]