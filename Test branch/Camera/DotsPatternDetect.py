"""
This code for detecting black dots on red soft object.
(不一定采用这个流程) Filting red component --> Canny edge detection --> Find contours.
created at 2024-07-15 by hsy.
"""

import time
import cv2
from typing import Tuple 
import numpy as np
import numpy.typing as npt
import pyzed.sl as sl
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


def init_camera():
    # 创建一个相机对象
    zed = sl.Camera()

    # 设置相机配置
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 设置相机分辨率为HD720
    init_params.camera_fps = 30  # 设置相机的帧率为30 fps

    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # 创建一个图像矩阵对象
    image = sl.Mat()

    # 设置窗口
    window_name = 'ZED Camera Image'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1080, 720)

    return zed, image, window_name


def get_image(zed, image):
    # 捕获图像
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        # 将图像从ZED相机转移到图像矩阵
        zed.retrieve_image(image, sl.VIEW.RIGHT)    # 你可以选择 LEFT 或 RIGHT
        # 将图像矩阵转换为opencv格式
        frame = image.get_data()
        color_image = frame
        if color_image is not None:
            # opencv BGRA
            return color_image


def init_region_range(zed_id, image_init:npt.NDArray[np.uint8], red_range:list)->npt.NDArray[np.float64]:
    """
    获得初始图像上的点的位置
    :return: 初始点位置的均值
    """
    image_bgra = get_image(zed_id, image_init)
    mask = np.zeros(image_bgra.shape[:2], dtype=np.uint8)
    mask[:, :] = 255
    itr_num = 50
    dots_list = []
    for i in range(itr_num):
        image_bgra = get_image(zed_id, image_init)
        edges = image_process(image_bgra, red_range, mask)
        dots, area, ellipse = ellipse_fitting(edges)
        # print('Dot coordinates:', dot, area)
        dots_list += dots

    dots_array = np.array(dots_list)

    # 计算均值向量和协方差矩阵
    mean_vector = np.mean(dots_array, axis=0)
    cov_matrix = np.cov(dots_array, rowvar=False)

    # 计算 Mahalanobis 距离
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = [mahalanobis(sample, mean_vector, inv_cov_matrix) for sample in dots_array]

    # 设置阈值（例如使用 99% 置信水平）
    threshold = chi2.ppf((1-0.01), df=2)

    # 剔除离群值
    filtered_data = dots_array[np.array(distances) < threshold]

    return np.mean(filtered_data, axis=0)


def cal_center(contour):
    area = cv2.contourArea(contour)
    if area < 1:
        return None
    # 计算矩
    M = cv2.moments(contour)
    if M['m00'] != 0:
        # 计算质心坐标
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        return (cX, cY), area
    else:
        return None


def get_mask_region(init_pos, masked_region, size:int=50):
    height, width = masked_region.shape

    x1 = max(int(init_pos[0]) - size, 0)
    x2 = min(int(init_pos[0]) + size, width)
    y1 = max(int(init_pos[1]) - size, 0)
    y2 = min(int(init_pos[1]) + size, height)

    # 将指定区域的值设置为255（白色）
    masked_region[y1:y2, x1:x2] = 255
    return masked_region


def image_process(image_bgra:npt.NDArray, color_range:list, masekd_region:npt.NDArray):
    """
    处理图像，得到待跟踪黑点的外轮廓
    :param image_bgra: 输入的图像(W, H, 4)
    :param color_range: 红色范围[array*4]
    :param masekd_region: 感兴趣区域(W, H)
    """
    # Convert color image to OpenCV format
    color_image_rgb = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
    color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('color_image_bgr.png', color_image_bgr)

    # Convert the image to the HSV color space (for red detection)
    hsv = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2HSV)

    # Create a mask to extract the red tissue
    lower_red_1, upper_red_1, lower_red_2, upper_red_2 = color_range
    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

    # 定义一个小的闭运算核
    kernel = np.ones((9, 9), np.uint8)  # 9x9大小的正方形核。你可以根据空洞的大小调整这个。

    # 使用闭运算，可能会破坏掉边界
    closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    # cv2.imwrite('closed_mask.png', closed_mask)

    # Apply the mask to the original image
    # 将两个mask进行按位与操作
    combined_mask = cv2.bitwise_and(closed_mask, masekd_region)
    masked_image = cv2.bitwise_and(color_image_bgr, color_image_bgr, mask=combined_mask)
    # cv2.imwrite('masked_image.png', masked_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    # cv2.imwrite('gray.png', gray)

    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)           # 应用高斯模糊，减少图像中的噪声

    # Canny边缘检测
    edges = cv2.Canny(blurred_image, 50, 60)            # 这里的两个数字表示低阈值和高阈值
    # cv2.imwrite('edges.png', edges)

    # 腐蚀变形体边界去除外轮廓
    kernel = np.ones((7, 7), np.uint8)
    closed_mask_reduced = cv2.erode(combined_mask, kernel, iterations=1)
    edges = cv2.bitwise_and(edges, closed_mask_reduced)
    # cv2.imwrite('edges_erode.png', edges)

    return edges


def ellipse_fitting(edges:npt.NDArray[np.uint8])->Tuple[list, list, list]:
    points_num:int = 2
    dot_coordinates = []
    areas = []
    filtered_contours = []
    # CHAIN_APPROX_NONE 存储所有轮廓点，CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线段，只保留其端点
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 检查层级信息，层级信息存储在hierarchy[0]中，格式为[next, previous, first_child, parent]
    if hierarchy is not None:
        for idx, contour in enumerate(contours):
            # # 检查是否有父轮廓，parent index = hierarchy[0, idx, 3] 不为-1则为内轮廓
            # if hierarchy[0, idx, 3] == -1:
            center_result = cal_center(contour)
            if center_result is not None:
                (cX, cY), area = center_result
                dot_coordinates.append((cX, cY))
                areas.append(area)
                filtered_contours.append(contour)

    if len(areas) == 0:                                 # 没有找到edge轮廓,canny边缘检测失败(光源掉了)
        return None, None, None
    areas_np = np.array(areas)
    # 过滤出面积不大于50的边界及其索引
    filtered_indices = np.where(areas_np <= 50)[0]      # type: npt.NDArray[np.int64]
    if len(filtered_indices) == 0:                      # 没有找到符合条件的edge轮廓
        return None, None, None
    filtered_areas = areas_np[filtered_indices]
    
    # 对过滤后的areas_np进行排序，并取出最大的point_num个元素及其索引
    if len(filtered_indices) < points_num:
        sorted_indices = np.argsort(filtered_areas)
    sorted_indices = np.argsort(filtered_areas)[-points_num:]
    largest_n_elements = filtered_areas[sorted_indices]
    original_indices = filtered_indices[sorted_indices]

    centers = []
    ellipse_areas = []
    ellipses = []
    for dots_idx in original_indices:
        if len(filtered_contours[dots_idx]) < 5:             # 椭圆拟合需要至少5个点
            continue
        ellipses_tmp = cv2.fitEllipse(filtered_contours[dots_idx])
        (x, y), (MA, ma), angle = ellipses_tmp
        ellipses.append(ellipses_tmp)
        # 圆心
        centers.append([x, y])
        ellipse_areas.append(MA * ma * np.pi / 4)
    return centers, ellipse_areas, ellipses


def image_show(image_bgr, ellipse, window_name='ZED Camera Image'):
    if ellipse is None:
        cv2.imshow(window_name, image_bgr)
        return
    else:
        # 绘制拟合的椭圆
        # cv2.ellipse(image_bgr, ellipse, (0, 255, 0), 1)
        (x, y), _, _ = ellipse
        cv2.circle(image_bgr, (int(x), int(y)), 5, (0, 255, 0), 1)
        cv2.imshow(window_name, image_bgr)


def kalman_filter_init():
    kalman = cv2.KalmanFilter(4, 2)  # 状态空间为4维，测量空间为2维
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], np.float32) * 0.03
    # 设置初始误差协方差矩阵
    kalman.errorCovPre = np.eye(4, dtype=np.float32) * 1  # 根据实际情况调整

    # 设置测量噪声协方差矩阵
    kalman.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], np.float32) * 1  # 根据实际情况调整

    return kalman


def kalman_filter_process(kalman, detected_dot, position_threshold):
    # 预测
    prediction = kalman.predict()
    predicted_x, predicted_y = prediction[0][0], prediction[1][0]
    if detected_dot is not None:
        detected_x, detected_y = detected_dot
        position_deviation = np.abs(detected_x - predicted_x) + np.abs(detected_y - predicted_y)        # 计算位置偏差

        # 判断是否为错误的检测点
        if position_deviation > position_threshold:
            # 如果偏差超过阈值，则认为检测点无效，使用预测值
            measurement = None
            pass
        else:
            # 更新测量
            measurement = np.array([[np.float32(detected_dot[0])],
                                    [np.float32(detected_dot[1])]])
            kalman.correct(measurement)
    else:
        # 如果未检测到点，则使用预测值
        measurement = None

    # 预测
    prediction = kalman.predict()
    predicted_point = (prediction[0][0], prediction[1][0])
    return predicted_point


def dot_filter(dots, area):
    """
    筛选出符合条件的点，使得到的点只有一个
    :return:
    """
    if area < 10:
        raise ValueError('Too small area!')


def main():
    # Define the lower and upper bounds for the red color in HSV
    lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([156, 43, 46])
    upper_red_2 = np.array([180, 255, 255])
    red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

    kalman = kalman_filter_init()

    zed_id, image_init, window_name = init_camera()
    dot_init = init_region_range(zed_id, image_init, red_range)
    print(f'Init position of dot: {dot_init}')
    exit(0)
    color_image_bgra = get_image(zed_id, image_init)
    height, width = color_image_bgra.shape[:2]
    masked_region = np.zeros(color_image_bgra.shape[:2], dtype=np.uint8)
    masked_region = get_mask_region(dot_init, masked_region, 50)
    kalman.statePre = np.array([[dot_init[0]], [dot_init[1]], [0], [0]], np.float32)
    kalman.statePost = np.array([[dot_init[0]], [dot_init[1]], [0], [0]], np.float32)

    try:
        while True:  # 按Q键退出
            color_image_bgra = get_image(zed_id, image_init)
            edges = image_process(color_image_bgra, red_range, masked_region)
            dots, areas, ellipse = ellipse_fitting(edges)           # dots是元组，不可改变

            filtered_dot_cell = kalman_filter_process(kalman, dots, 5)

            # dot_filter(dots, areas)
            if areas is None:
                image_show(color_image_bgra, ellipse, window_name)
                print('No dots detected in this step!')
            else:
                image_show(color_image_bgra, ellipse, window_name)
                print(f'Dot coordinates:, [{filtered_dot_cell[0]:.2f}, {filtered_dot_cell[1]:.2f}], {areas:.2f}')

            masked_region = get_mask_region(filtered_dot_cell, masked_region, 50)

            # Press 'q' to exit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        zed_id.close()
        cv2.destroyAllWindows()

        # Print the coordinates of detected dots
        print("Detected Dot Coordinates on Red Tissue:")
        for i, (cX, cY) in enumerate(dots, 0):
            print(f"Dot {i}: ({cX}, {cY}). Area: {areas[i]:.2f} pixels^2")


if __name__ == '__main__':
    main()