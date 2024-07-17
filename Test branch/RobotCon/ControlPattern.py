"""
实物实验,使用Strain Constraint & Volume Constraint的DiffPD进行控制
created at 2024-07-17 by hsy
"""


import time
import os
import cv2
import numpy as np
from ControlSimulation import *
from RobAction import URROb
from DotsPatternDetect import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# pattern中有几个点，重新赋值
POINTS_NUM:int = 2

# Define the lower and upper bounds for the red color in HSV
lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 43, 46])
upper_red_2 = np.array([180, 255, 255])
red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

image_width, image_height = int(1280), int(720)

kalman = kalman_filter_init()

