"""
获得期望变形下的标记点位置
"""


import numpy as np
import numpy.typing as npt
import cv2
from DotsPatternDetect import init_camera, init_region_range, get_image
from CoordinateTransform import dot_in_soft

def main():
    # Define the lower and upper bounds for the red color in HSV
    lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([156, 43, 46])
    upper_red_2 = np.array([180, 255, 255])
    red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    zed_id, image_init, window_name = init_camera()
    dots_init = init_region_range(zed_id, image_init, red_range)
    print(f'Init position of dot: {dots_init}')

    color_image_bgra = get_image(zed_id, image_init)
    for dot in dots_init:
        color_image_bgra = cv2.circle(color_image_bgra, (int(dot[0]), int(dot[1])), 5, (0, 255, 0, 255), -1)
    cv2.imwrite('data/desired_pos.png', color_image_bgra)

    dots_desired_pos = dot_in_soft(dots_init, trans_soft, intrinsic)
    np.savez('data/desired_pos.npz', desired_pos=dots_desired_pos, desired_pixel=dots_init)

    while True:
        cv2.imshow(window_name, color_image_bgra)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()