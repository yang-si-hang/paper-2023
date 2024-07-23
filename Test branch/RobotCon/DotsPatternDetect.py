"""
This code for detecting black dots on red soft object.
Filting red component --> Canny edge detection --> Find contours.
created at 2024-07-17 by hsy.
"""

import time
import cv2
from typing import Tuple, List
import numpy as np
import numpy.typing as npt
import pyzed.sl as sl
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from sklearn.cluster import KMeans


POINTS_NUM:int = 4


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
        

def bgr2gray(image_rgb:npt.NDArray[np.uint8], weights:List[float])->npt.NDArray[np.uint8]:
    blue_weight, green_weight, red_weight = weights

    B = image_rgb[:, :, 0]
    G = image_rgb[:, :, 1]
    R = image_rgb[:, :, 2]

    image_gray = red_weight * R + green_weight * G + blue_weight * B
    
    return image_gray.astype(np.uint8)


def init_region_range(zed_id, image_init, red_range:list)->npt.NDArray[np.float32]:
    """
    获得初始图像上的点的位置
    :return: 初始点位置的均值
    """
    image_bgra = get_image(zed_id, image_init)
    itr_num = 50
    dots_list = []
    for i in range(itr_num):
        image_bgra = get_image(zed_id, image_init)
        edges = image_process(image_bgra, red_range)
        dots, area, ellipse = ellipse_fitting(edges)

        # cv2.imshow('Canny', edges)
        # # 设置按键（例如，按 'c' 继续）
        # while True:
        #     key = cv2.waitKey(1) & 0xFF
        #     if key == ord('c'):
        #         break

        if dots is None:
            # print('No dots detected!')
            continue
        # print('Dot coordinates:', dot, area)
        dots_list += dots

    if not dots_list:
        raise ValueError('Init failed!')
    dots_array = np.array(dots_list)
    # print(f'Init dots: {dots_array}')

    # 使用K-means将点聚类为POINTS_NUM个点
    kmeans = KMeans(n_clusters=POINTS_NUM, n_init='auto', random_state=0).fit(dots_array)

    dots_center = kmeans.cluster_centers_

    # 获取每个点到其簇中心的距离平方和
    sse = kmeans.inertia_/itr_num/POINTS_NUM
    print(f'Sum of Squared Errors (SSE): {sse}')
    if sse > 20:
        pass
        # raise ValueError('Init failed!')

    return dots_center.astype(np.float32)          # shape:POINTS_NUM*2


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


def get_mask_region(init_pos:npt.NDArray[np.float64], shape:List[int], size:int=50)->npt.NDArray[np.uint8]:
    height, width = shape
    new_mask = np.zeros((height, width), dtype=np.uint8)

    for pos in init_pos:
        x1 = max(int(pos[0]) - size, 0)
        x2 = min(int(pos[0]) + size, width)
        y1 = max(int(pos[1]) - size, 0)
        y2 = min(int(pos[1]) + size, height)

        # 将指定区域的值设置为255（白色）
        new_mask[y1:y2, x1:x2] = 255

    return new_mask


def image_process(image_bgra:npt.NDArray, color_range:list):
    """
    处理图像，得到待跟踪黑点的外轮廓
    :param image_bgra: 输入的图像(W, H, 4)
    :param color_range: 红色范围[array*4]
    :return edge: Canny算子得到的边界
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
    masked_image = cv2.bitwise_and(color_image_bgr, color_image_bgr, mask=closed_mask)
    # cv2.imwrite('masked_image.png', masked_image)

    # Convert the image to grayscale
    # gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    gray = bgr2gray(masked_image, [0.2, 0.2, 0.6])
    # cv2.imshow('Gray', gray)
    # 设置按键（例如，按 'c' 继续）
    # while True:
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('c'):
    #         break
    # cv2.imwrite('gray.png', gray)

    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)           # 应用高斯模糊，减少图像中的噪声

    # Canny边缘检测
    edges = cv2.Canny(blurred_image, 30, 60)            # 这里的两个数字表示低阈值和高阈值
    # cv2.imwrite('edges.png', edges)

    # 腐蚀变形体边界去除外轮廓
    kernel = np.ones((9, 9), np.uint8)
    closed_mask_reduced = cv2.erode(closed_mask, kernel, iterations=1)
    edges = cv2.bitwise_and(edges, closed_mask_reduced)
    # cv2.imwrite('edges_erode.png', edges)

    return edges


def ellipse_fitting(edges:npt.NDArray[np.uint8])->Tuple[list, list, list]:
    dot_coordinates = []
    areas = []
    filtered_contours = []
    # CHAIN_APPROX_NONE 存储所有轮廓点，CHAIN_APPROX_SIMPLE 压缩水平、垂直和对角线段，只保留其端点
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # 检查层级信息，层级信息存储在hierarchy[0]中，格式为[next, previous, first_child, parent]
    if hierarchy is not None:
        for idx, contour in enumerate(contours):
            # # 检查是否有父轮廓，parent index = hierarchy[0, idx, 3] 不为-1则为内轮廓
            if hierarchy[0, idx, 3] == -1:
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
    filtered_indices = np.where(areas_np <= 90)[0]      # type: npt.NDArray[np.int64]
    if len(filtered_indices) == 0:                      # 没有找到符合条件的edge轮廓
        return None, None, None
    filtered_areas = areas_np[filtered_indices]
    
    # 对过滤后的areas_np进行排序，并取出最大的point_num个元素及其索引
    if len(filtered_indices) < POINTS_NUM:
        sorted_indices = np.argsort(filtered_areas)
    sorted_indices = np.argsort(filtered_areas)[-POINTS_NUM:]
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
                                         [0, 1, 0, 0]], dtype=np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], dtype=np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]], dtype=np.float32) * 0.03
    # 设置初始误差协方差矩阵
    kalman.errorCovPre = np.eye(4, dtype=np.float32) * 0.1  # 根据实际情况调整

    # 设置测量噪声协方差矩阵
    kalman.measurementNoiseCov = np.array([[1, 0],
                                           [0, 1]], dtype=np.float32) * 0.1  # 根据实际情况调整

    return kalman


def kalman_predict(kalmans:List[cv2.KalmanFilter]):
    # 只做预测，statepre会变
    dots_pred = np.zeros((POINTS_NUM, 2))
    for idx, kalman in enumerate(kalmans):
        kalman_pred = kalman.predict()  # shape:4*1
        dots_pred[idx, :] = kalman_pred[:2].flatten()
        # print(f'Predictor Velocity: {kalman_pred[2:].flatten()}')

    return dots_pred


def kalman_process(kalmans:List[cv2.KalmanFilter], dots_now:list)->npt.NDArray:
    # match的点用correct，没有match的点用predict
    processed_dots_np = np.zeros((POINTS_NUM, 2), dtype=np.float32)
    for idx, (kalman, dot_now) in enumerate(zip(kalmans, dots_now)):
        if dot_now is not None:
            measurement = np.array([[dot_now[0]],
                                    [dot_now[1]]])
            correct = kalman.correct(measurement)
            corrected_x, corrected_y = correct[0][0], correct[1][0]
            processed_dots_np[idx, :] = corrected_x, corrected_y
        else:
            predicted_x, predicted_y = kalman.statePre[0][0], kalman.statePre[1][0]
            processed_dots_np[idx, :] = predicted_x, predicted_y
    
    return processed_dots_np           # shape:POINTS_NUM*2


def match_kalman(dots_new_matched, dots_pred, dots_now):
    for dot_new_matched in dots_new_matched:
        # 计算点的距离
        distances = np.linalg.norm(dots_pred - dot_new_matched, axis=1)
        if min(distances) > 20:  # 如果距离太远，不更新
            print('Too far away!')
            continue
        # 找到距离最小的点
        min_idx = np.argmin(distances)
        dots_now[min_idx] = dot_new_matched

    return dots_now


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
    
    cv2.namedWindow('Canny', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Canny', 1080, 720)

    # cv2.namedWindow('Gray', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('Gray', 1080, 720)

    kalman = kalman_filter_init()
    kalmans = [kalman_filter_init() for _ in range(POINTS_NUM)]

    # Lucas-Kanade光流法参数
    lk_params = dict(winSize=(15, 15), maxLevel=2, 
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    zed_id, image_init, window_name = init_camera()
    dots_init = init_region_range(zed_id, image_init, red_range)
    print(f'Init position of dot: {dots_init}')
    color_image_bgra = get_image(zed_id, image_init)

    for idx, dot in enumerate(dots_init):
        kalmans[idx].statePre = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)
        kalmans[idx].statePost = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)
    # masked_region = np.ones(color_image_bgra.shape[:2], dtype=np.uint8) * 255

    # old_gray = cv2.cvtColor(color_image_bgra, cv2.COLOR_BGRA2GRAY)
    old_gray = bgr2gray(color_image_bgra, [0.2, 0.2, 0.6])
    detected_dots_np = dots_init
    try:
        while True:  # 按Q键退出
            color_image_bgra = get_image(zed_id, image_init)
            # image_gray = cv2.cvtColor(color_image_bgra, cv2.COLOR_BGRA2GRAY)
            image_gray = bgr2gray(color_image_bgra, [0.2, 0.2, 0.6])
            dots_now = [None] * POINTS_NUM
            dots_pred = kalman_predict(kalmans)

            if not np.isnan(detected_dots_np).any():
                dots_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, image_gray, detected_dots_np, None, **lk_params)
                st = st.flatten()
                
                # 光流匹配成功的特征点
                dots_new_matched = dots_new[st == 1]
                # print(f'New dots matched: {dots_new_matched}')

                dots_now = match_kalman(dots_new_matched, dots_pred, dots_now)
                # print(f'Now dots: {dots_now}')

            # optical flow & kalman filter
            processed_dots = kalman_process(kalmans, dots_now)
            print(f'Processed dots: {processed_dots.tolist()}')

            old_gray = image_gray.copy()
            edges = image_process(color_image_bgra, red_range)
            detected_dots, areas, ellipse = ellipse_fitting(edges)
            detected_dots_np = np.array(detected_dots, dtype=np.float32)
            # print(f'Detected dots: {detected_dots}')

            for processed_dot in processed_dots:
                cv2.circle(color_image_bgra, (int(processed_dot[0]), int(processed_dot[1])), 5, (0, 255, 0), 1)

            cv2.imshow(window_name, color_image_bgra)
            cv2.imshow('Canny', edges)

            # Press 'q' to exit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        zed_id.close()
        cv2.destroyAllWindows()

        # Print the coordinates of detected dots
        print("Detected Dot Coordinates on Red Tissue:")
        for i, (cX, cY) in enumerate(processed_dots, 0):
            print(f"Dot {i}: ({cX}, {cY}).")


if __name__ == '__main__':
    main()