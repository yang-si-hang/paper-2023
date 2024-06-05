"""
This code for detecting black dots on red soft object.
Filting red component --> Canny edge detection --> Find contours.
"""

import time
import cv2
import numpy as np
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


def init_region_range(zed_id, image_init, red_range):
    image_bgra = get_image(zed_id, image_init)
    mask = np.zeros(image_bgra.shape[:2], dtype=np.uint8)
    mask[:, :] = 255
    itr_num = 50
    dots_list = []
    for i in range(itr_num):
        image_bgra = get_image(zed_id, image_init)
        edges = image_process(image_bgra, red_range, mask)
        dot, area, ellipse = ellipse_fitting(edges)
        # print('Dot coordinates:', dot, area)
        dots_list.append(dot)

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


# @profile
def image_process(image_bgra, color_range, masekd_region):
    # Convert color image to OpenCV format
    color_image_rgb = cv2.cvtColor(image_bgra, cv2.COLOR_BGRA2RGB)
    color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
    # cv2.imwrite('color_image.png', color_image_bgr)

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
    # cv2.imwrite('edges.png', edges)

    return edges


def ellipse_fitting(edges):
    dot_coordinates = []
    areas = []
    filtered_contours = []
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 检查层级信息，层级信息存储在hierarchy[0]中，格式为[next, previous, first_child, parent]
    if hierarchy is not None:
        for idx, contour in enumerate(contours):
            # 检查是否有父轮廓，parent index = hierarchy[0, idx, 3] 不为-1则为内轮廓
            if hierarchy[0, idx, 3] == -1:
                center_result = cal_center(contour)
                if center_result is not None:
                    (cX, cY), area = center_result
                    dot_coordinates.append((cX, cY))
                    areas.append(area)
                    filtered_contours.append(contour)

    # 找到area中最大元素的索引,使用该索引找到dot_coordinates中对应的元素
    max_index = areas.index(max(areas))

    # 拟合椭圆
    ellipse = cv2.fitEllipse(filtered_contours[max_index])
    (x, y), (MA, ma), angle = ellipse
    # 圆心
    center = (x, y)
    ellipse_area = MA * ma * np.pi / 4

    # return [dot_coordinates[max_index]], [areas[max_index]]
    return center, ellipse_area, ellipse


def image_show(image_bgr, ellipse, window_name='ZED Camera Image'):
    # 绘制拟合的椭圆
    cv2.ellipse(image_bgr, ellipse, (0, 255, 0), 1)
    cv2.imshow(window_name, image_bgr)
    # cv2.imwrite('color_image_bgr.png', image_bgr)


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

    zed_id, image_init, window_name = init_camera()
    dot_init = init_region_range(zed_id, image_init, red_range)
    print(f'Init position of dot: {dot_init}')
    color_image_bgra = get_image(zed_id, image_init)
    height, width = color_image_bgra.shape[:2]
    masked_region = np.zeros(color_image_bgra.shape[:2], dtype=np.uint8)

    masked_region = get_mask_region(dot_init, masked_region, 50)

    # size = int(50)
    # x1 = max(int(dot_init[0]) - size, 0)
    # x2 = min(int(dot_init[0]) + size, width)
    # y1 = max(int(dot_init[1]) - size, 0)
    # y2 = min(int(dot_init[1]) + size, height)
    #
    # # 将指定区域的值设置为255（白色）
    # masked_region[y1:y2, x1:x2] = 255

    try:
        while True:  # 按Q键退出
            color_image_bgra = get_image(zed_id, image_init)
            edges = image_process(color_image_bgra, red_range, masked_region)
            dots, areas, ellipse = ellipse_fitting(edges)
            image_show(color_image_bgra, ellipse, window_name)
            # dot_filter(dots, areas)
            print('Dot coordinates:', dots, areas)

            masked_region = get_mask_region(dots, masked_region, 50)

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