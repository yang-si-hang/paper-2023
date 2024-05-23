"""
Obtain the transformation matrix from the camera coordinate system to the chess board coordinate system.
"""

import cv2
import numpy as np
import pyzed.sl as sl


def get_camera_intrinsic():
    # 创建一个相机对象
    zed = sl.Camera()

    # 设置相机配置
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 设置分辨率
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # 设置深度模式

    # 打开相机
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # 获取相机内参
    camera_info = zed.get_camera_information()
    left_cam_params = camera_info.camera_configuration.calibration_parameters.left_cam
    right_cam_params = camera_info.camera_configuration.calibration_parameters.right_cam

    camera_used_params = right_cam_params

    # 内参矩阵
    camera_matrix = np.array([[camera_used_params.fx, 0, camera_used_params.cx],
                              [0, camera_used_params.fy, camera_used_params.cy],
                              [0, 0, 1]], dtype=np.float32)

    # 畸变系数
    dist_coeffs = np.array([camera_used_params.disto[0], camera_used_params.disto[1], camera_used_params.disto[2],
                            camera_used_params.disto[3], camera_used_params.disto[4]], dtype=np.float32)

    image = sl.Mat()

    # 捕获一帧
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # 从相机中检索图像
        zed.retrieve_image(image, sl.VIEW.RIGHT)  # 你可以选择 LEFT 或 RIGHT

        # 将图像转换为 OpenCV 格式
        image_cv = image.get_data()
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGRA2BGR)

        # 保存图像到文件
        cv2.imwrite("zed_image.jpg", image_cv)
        print("Right image captured and saved as zed_image.jpg")

    # 关闭相机
    zed.close()

    return camera_matrix, dist_coeffs, image_cv


def get_border(chessboard_size, square_size, image_chess, intrisic_matrix, dist):
    # 世界坐标系中的棋盘格点, X向右，Y向下，Z向里
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # 读取棋盘格图像
    # image = cv2.imread('chessboard.jpg')
    image = image_chess

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格的角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # 如果找到了角点
    if ret:
        # 提高角点的准确度
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))

        # 计算旋转和平移向量，相机是Z向外的标准规定坐标系
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, intrisic_matrix, dist)

        # 将旋转向量转换为旋转矩阵
        rotation_matrix, _ = cv2.Rodrigues(rvecs)

        # 打印变换矩阵
        transformation_matrix = np.hstack((rotation_matrix, tvecs))
        transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))
        print("Transformation Matrix:\n", transformation_matrix)
    else:
        print("Chessboard corners not found")
        raise Exception("Chessboard corners not found")

    return image, corners2, ret, transformation_matrix


def main():
    # 棋盘格的大小和每个棋盘格的格子宽度（单位：米）
    chessboard_size, square_size = (11, 8), 0.015

    intrisic_matrix, dist, image_chess = get_camera_intrinsic()

    image, corner_refined, ret, *_ = get_border(chessboard_size, square_size, image_chess, intrisic_matrix, dist)

    # 显示图像并标记角点
    cv2.drawChessboardCorners(image, chessboard_size, corner_refined, ret)
    cv2.imshow('Chessboard', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()