import pyzed.sl as sl
import cv2

def main():
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

    key = ''
    while key != 113:  # 按Q键退出
        # 捕获图像
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # 将图像从ZED相机转移到图像矩阵
            zed.retrieve_image(image, sl.VIEW.RIGHT)
            # 将图像矩阵转换为OpenCV格式
            frame = image.get_data()
            # 显示图像
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(10)

    cv2.imwrite('zed_image.png', frame)
    # 关闭相机
    zed.close()

if __name__ == "__main__":
    main()
