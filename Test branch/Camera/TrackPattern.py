"""
追踪图像上的一群点特征
created at 2024-07-15 by hsy
"""

import cv2
import numpy as np
from DotsPatternDetect import *


# 初始化Kalman Filter
def initialize_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                     [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], 
                                    [0, 1, 0, 1], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = np.array([[1, 0, 0, 0], 
                                   [0, 1, 0, 0], 
                                   [0, 0, 1, 0], 
                                   [0, 0, 0, 1]], np.float32) * 0.03
    return kf


def main():



if __name__ == '__main__':
    main()