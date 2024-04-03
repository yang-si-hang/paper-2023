import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config


def get_color_image_from_kinect():
    # 创建Kinect对象并打开设备
    k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P))
    k4a.start()

    # 获取彩色相机内参
    cam_calib = k4a.calibration
    color_camera_intrinsics = cam_calib.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    print("彩色相机内参:", color_camera_intrinsics)

    color_intrinsics_str = "["
    for i, row in enumerate(color_camera_intrinsics):
        color_intrinsics_str += "[" + ", ".join(map(str, row)) + "],"
        if i != 2:
            color_intrinsics_str += "\n"
    color_intrinsics_str += "]"

    with open('Intrinsic.txt', 'w') as f:
        f.write(color_intrinsics_str)

    # 获取下一帧的捕获
    capture = k4a.get_capture()

    # 从捕获中提取彩色图像
    color_image = capture.color

    # 停止设备并关闭
    k4a.stop()

    return color_image


if __name__ == "__main__":
    img = get_color_image_from_kinect()
    cv2.imwrite("kinect_image.jpg", img)
