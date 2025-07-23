"""
检测图像中的四边形,并得到它的四个定点像素坐标
"""

import cv2
import numpy as np

# 读取图像
image = cv2.imread('closed_mask.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 应用Canny边缘检测
edges = cv2.Canny(gray, 50, 60, apertureSize=3)
cv2.imwrite('edges.png', edges)

# 定义结构元素（kernel）
kernel = np.ones((5, 5), np.uint8)

# 进行闭运算以平滑轮廓,如封闭曲线，平滑边界(凸起的地方变厚,像"土"字)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('closed.png', closed)

# 查找轮廓
contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)

# 多边形逼近
cv2.drawContours(image, [contour], 0, (255, 0, 0), 1)

epsilon = 0.05 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

for point in approx:
    cv2.circle(image, tuple(point[0]), 3, (0, 0, 255), -1)

if len(approx) > 4:
    approx = cv2.convexHull(approx)
    print(len(approx))

# 如果轮廓有4个顶点，我们假设它是一个长方形
if len(approx) == 4:
    # 画出多边形轮廓
    cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
    # pass

# 打印角点坐标
print("角点坐标:")
for point in approx:
    print(point[0])

# 显示图像
cv2.imshow('Detected Rectangle', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
