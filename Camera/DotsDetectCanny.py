"""
This code for detecting black dots on red soft object.
Filting red component --> Canny edge detection --> Find contours.
"""


import time
import cv2
import numpy as np
import pyzed.sl as sl


# 创建一个相机对象
zed = sl.Camera()

# 设置相机配置
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720     # 设置相机分辨率为HD1080
init_params.camera_fps = 30                             # 设置相机的帧率为30 fps

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

# Define the lower and upper bounds for the red color in HSV
lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 43, 46])
upper_red_2 = np.array([180, 255, 255])


def detect_rect(image):
    """
    检测长方形
    :param image:
    :return:
    """
    gray = image

    # 应用Canny边缘检测
    edges = cv2.Canny(gray, 50, 60, apertureSize=3)

    # 定义结构元素（kernel）
    kernel = np.ones((5, 5), np.uint8)

    # 进行闭运算以平滑轮廓
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)

    # 多边形逼近
    epsilon = 0.05 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    for point in approx:
        cv2.circle(image, tuple(point[0]), 3, (0, 0, 255), -1)

    # 如果轮廓有4个顶点，我们假设它是一个长方形
    if len(approx) == 4:
        # 画出轮廓和角点
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

        # # 打印角点坐标
        # print("角点坐标:")
        # for point in approx:
        #     print(point[0])

        return  approx
    else:
        return None


def calculate_area(x1, y1, x2, y2, x3, y3):
    return 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

def calculate_barycentric_coordinates(A, B, C, D, P):
    # 四边形的四个顶点
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = D

    # 内部点
    xp, yp = P

    # 计算四边形的总面积
    area_ABCD = calculate_area(x1, y1, x2, y2, x3, y3) + calculate_area(x1, y1, x3, y3, x4, y4)

    # 计算每个子三角形的面积
    area_PAB = calculate_area(xp, yp, x1, y1, x2, y2)
    area_PBC = calculate_area(xp, yp, x2, y2, x3, y3)
    area_PCD = calculate_area(xp, yp, x3, y3, x4, y4)
    area_PDA = calculate_area(xp, yp, x4, y4, x1, y1)

    # 计算重心坐标
    lambda_A = area_PBC / area_ABCD
    lambda_B = area_PDA / area_ABCD
    lambda_C = area_PAB / area_ABCD
    lambda_D = area_PCD / area_ABCD

    return (lambda_A, lambda_B, lambda_C, lambda_D)


# @profile
def main():
    try:
        while True:  # 按Q键退出
            # time_start = time.time()
            # 捕获图像
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                # 将图像从ZED相机转移到图像矩阵
                zed.retrieve_image(image, sl.VIEW.RIGHT)    # 你可以选择 LEFT 或 RIGHT
                # 将图像矩阵转换为OpenCV格式
                frame = image.get_data()
                color_image = frame
                # time_capture = time.time()
                if color_image is not None:
                    # Convert color image to OpenCV format
                    # color_image = color_image[:, :, 0:3]
                    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
                    color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)
                    # cv2.imwrite('color_image.png', color_image_bgr)

                    # Convert the image to the HSV color space (for red detection)
                    hsv = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2HSV)

                    # Create a mask to extract the red tissue
                    red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
                    red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

                    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

                    # 定义一个小的闭运算核
                    kernel = np.ones((9, 9), np.uint8)  # 9x9大小的正方形核。你可以根据空洞的大小调整这个。

                    # 使用闭运算，可能会破坏掉边界
                    closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
                    # cv2.imwrite('closed_mask.png', closed_mask)

                    # 检测四边形
                    rect_corner = detect_rect(red_mask)
                    corner_pos = np.squeeze(rect_corner)

                    # Apply the mask to the original image
                    masked_image = cv2.bitwise_and(color_image_bgr, color_image_bgr, mask=closed_mask)
                    # cv2.imwrite('masked_image.png', masked_image)

                    # Convert the image to grayscale
                    gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
                    # cv2.imwrite('gray.png', gray)

                    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)           # 应用高斯模糊，减少图像中的噪声

                    # Canny边缘检测
                    edges = cv2.Canny(blurred_image, 50, 60)            # 这里的两个数字表示低阈值和高阈值
                    # cv2.imwrite('edges.png', edges)

                    # 腐蚀变形体边界去除外轮廓
                    kernel = np.ones((5, 5), np.uint8)
                    closed_mask_reduced = cv2.erode(closed_mask, kernel, iterations=1)
                    edges = cv2.bitwise_and(edges, closed_mask_reduced)
                    # cv2.imwrite('edges.png', edges)

                    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    dot_coordinates = []
                    areas = []

                    # 检查层级信息，层级信息存储在hierarchy[0]中，格式为[next, previous, first_child, parent]
                    if hierarchy is not None:
                        for idx, contour in enumerate(contours):
                            # 检查是否有父轮廓，parent index = hierarchy[0, idx, 3] 不为-1则为内轮廓
                            if hierarchy[0, idx, 3] == -1:
                                area = cv2.contourArea(contour)
                                # 计算矩
                                M = cv2.moments(contour)
                                if M['m00'] != 0:
                                    # 计算质心坐标
                                    cX = int(M['m10'] / M['m00'])
                                    cY = int(M['m01'] / M['m00'])
                                    dot_coordinates.append((cX, cY))
                                    areas.append(area)

                                    # 在图像上标记中心点
                                    cv2.circle(color_image_bgr, (cX, cY), 5, (255, 0, 0), -1)

                    if rect_corner is not None:
                        for point in corner_pos:
                            cv2.circle(color_image_bgr, tuple(point), 5, (0, 255, 0), -1)

                    # time_end = time.time()
                    # Display the result (with circles around detected dots on red tissue)
                    cv2.imshow(window_name, color_image_bgr)

                    # print(f"Time taken: {time_end - time_start:.5f} seconds")

                    # Press 'q' to exit the application
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    finally:
        zed.close()
        cv2.destroyAllWindows()

        # Print the coordinates of detected dots
        print("Detected Dot Coordinates on Red Tissue:")
        for i, (cX, cY) in enumerate(dot_coordinates, 1):
            print(f"Dot {i}: ({cX}, {cY}). Area: {areas[i-1]:.2f} pixels^2")

        for i, point in enumerate(corner_pos, 1):
            print(f"Corner {i}: ({point[0]}, {point[1]})")

        cord = calculate_barycentric_coordinates(corner_pos[0], corner_pos[1], corner_pos[2], corner_pos[3], dot_coordinates[0])
        print(cord)


if __name__ == '__main__':
    main()