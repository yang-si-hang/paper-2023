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
from CVVideo import *
from CoordinateTransform import *

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
kalmans = [kalman_filter_init() for _ in range(POINTS_NUM)]

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def action_compress(vec:np.ndarray, max_length:float=3.e-4):
    """
    Compress action vector in a safe range
    """
    length = np.linalg.norm(vec)

    if length > max_length:
        factor = max_length / length
        return vec * factor
    else:
        return vec
    

def main():
    obj_shape = [0.1, 0.1]
    obj_seed_size = 0.01
    learning_rate = 1.e1

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    camera_id, image_init, window_name = init_camera()
    dots_init = init_region_range(camera_id, image_init, red_range)
    dot_pos_init = dot_in_soft(dots_init, trans_soft, intrinsic)
    print("The initial position of the dot in soft object: ", dot_pos_init)

    for idx, dot in enumerate(dots_init):
        kalmans[idx].statePre = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)
        kalmans[idx].statePost = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)

    class MyObject(SoftObject):
        def __init__(self, shape, seed_size, contact_idx: list):
            super().__init__(shape, seed_size, contact_idx)


if __name__ == '__main__':
    main()