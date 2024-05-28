"""
This code for detecting black dots on red soft object.
Filting red component --> Canny edge detection --> Find contours.
"""


import time
import cv2
import numpy as np
import pyzed.sl as sl

def init_camera():
    # 创建一个相机对象
    zed = sl.Camera()

    # 设置相机配置
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # 设置相机分辨率为HD1080
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


def init_region_range(image_bgr):
    itr_num = 10
    for i in range(itr_num):
        pass


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


# @profile
def get_dot(image_bgra, color_range, window_name='ZED Camera Image'):
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
    masked_image = cv2.bitwise_and(color_image_bgr, color_image_bgr, mask=closed_mask)
    # cv2.imwrite('masked_image.png', masked_image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('gray.png', gray)

    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)           # 应用高斯模糊，减少图像中的噪声

    # Canny边缘检测
    edges = cv2.Canny(blurred_image, 50, 60)            # 这里的两个数字表示低阈值和高阈值
    # cv2.imwrite('edges.png', edges)

    # 腐蚀变形体边界去除外轮廓
    kernel = np.ones((7, 7), np.uint8)
    closed_mask_reduced = cv2.erode(closed_mask, kernel, iterations=1)
    edges = cv2.bitwise_and(edges, closed_mask_reduced)
    cv2.imwrite('edges.png', edges)

    dot_coordinates = []
    areas = []
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

    # 找到area中最大元素的索引,使用该索引找到dot_coordinates中对应的元素
    max_index = areas.index(max(areas))
    (dot_x,  dot_y) = dot_coordinates[max_index]
    # 在图像上标记中心点
    cv2.circle(color_image_bgr, (dot_x,  dot_y), 5, (255, 0, 0), -1)

    # time_end = time.time()
    # Display the result (with circles around detected dots on red tissue)
    cv2.imshow(window_name, color_image_bgr)
    cv2.imwrite('color_image_bgr.png', color_image_bgr)
    # print(f"Time taken: {time_end - time_start:.5f} seconds")

    return [dot_coordinates[max_index]], [areas[max_index]]


def dot_filter(dots, area):
    """
    筛选出符合条件的点，使得到的点只有一个
    :return:
    """
    if area[0] < 10:
        raise ValueError('Too small area!')


def main():
    # Define the lower and upper bounds for the red color in HSV
    lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([156, 43, 46])
    upper_red_2 = np.array([180, 255, 255])
    red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

    zed_id, image_init, window_name = init_camera()
    try:
        while True:  # 按Q键退出
            color_image_bgra = get_image(zed_id, image_init)
            dots, areas = get_dot(color_image_bgra, red_range, window_name)
            dot_filter(dots, areas)
            print('Dot coordinates:', dots[0], areas[0])

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