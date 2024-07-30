"""
检测柔软绳上的一些标记点
created at 2024-07-29 by hsy
"""

import numpy as np
import cv2
import pyzed.sl as sl



# 创建一个相机对象
zed = sl.Camera()

# 设置相机配置
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD1080     # 设置相机分辨率为HD1080
init_params.camera_fps = 30                             # 设置相机的帧率为30 fps

image_range = [460, 210, 1080, 720]

# 打开相机
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS:
    exit(1)

# 创建一个图像矩阵对象
image = sl.Mat()

# 设置窗口
window_name = 'ZED Camera Image'
cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
cv2.resizeWindow(window_name, image_range[2], image_range[3])

cv2.namedWindow('edges', cv2.WINDOW_NORMAL)
cv2.resizeWindow('edges', image_range[2], image_range[3])

low_yellow = np.array([26, 43, 46])
upper_yellow = np.array([34, 255, 255])


def on_mouse(event, x, y, flags, param):
    # 获取当前鼠标位置的坐标和BGR值
    image_input = param['image']
    display_image = image_input.copy()
    if event == cv2.EVENT_MOUSEMOVE:
        bgr_value = image_input[y, x]
        text = f'X: {x}, Y: {y}, B: {bgr_value[0]}, G: {bgr_value[1]}, R: {bgr_value[2]}'

        # 在图像的左下角显示文本信息
        cv2.putText(display_image, text, (10, display_image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        cv2.displayOverlay(window_name, text, 1000)
    # return display_image


def adjust_bgr_weights(image_bgr:np.uint8, blue_weight=-0.2, green_weight=0.6, red_weight=0.6):
    # Normalize weights to ensure their sum is 1
    total_weight = blue_weight + green_weight + red_weight
    blue_weight /= total_weight
    green_weight /= total_weight
    red_weight /= total_weight

    # Apply the weights to each channel
    weighted_b = image_bgr[:, :, 0] * blue_weight
    weighted_g = image_bgr[:, :, 1] * green_weight
    weighted_r = image_bgr[:, :, 2] * red_weight

    weighted_image = weighted_b + weighted_g + weighted_r
    weighted_image = np.clip(weighted_image, 0, 255).astype(np.uint8)

    # # Apply the weights to each channel
    # weighted_image = cv2.addWeighted(image_bgr[:, :, 0], blue_weight, image_bgr[:, :, 1], green_weight, 0)
    # weighted_image = cv2.addWeighted(weighted_image, 1.0, image_bgr[:, :, 2], red_weight, 0)

    return weighted_image


def main():
    try:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()
                color_image = frame[image_range[1]:image_range[1]+image_range[3], image_range[0]:image_range[0]+image_range[2], :]

                if color_image is not None:
                    color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
                    color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

                    hsv = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2HSV)

                    yellow_mask = cv2.inRange(hsv, low_yellow, upper_yellow)
                    masked_image = cv2.bitwise_and(color_image_bgr, color_image_bgr, mask=yellow_mask)

                    gray = adjust_bgr_weights(color_image_bgr)
                    blurred_image = cv2.GaussianBlur(gray, (5, 5), 0)           # 应用高斯模糊，减少图像中的噪声
                    edges = cv2.Canny(blurred_image, 60, 80)

                    cv2.setMouseCallback(window_name, on_mouse, {'image': color_image})
                    cv2.imshow(window_name, color_image)
                    cv2.imshow('edges', edges)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    # cv2.imwrite('Rope.png',color_image)
                    break

    finally:
        zed.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()