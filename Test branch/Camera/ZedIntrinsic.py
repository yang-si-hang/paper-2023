import pyzed.sl as sl

def main():
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

    # 准备要写入文件的内容
    left_camera_matrix = (f"[[{left_cam_params.fx}, 0, {left_cam_params.cx}],\n"
                          f"[0, {left_cam_params.fy}, {left_cam_params.cy}],\n"
                          f"[0, 0, 1]]\n")
    right_camera_matrix = (f"[[{right_cam_params.fx}, 0, {right_cam_params.cx}],\n"
                           f"[0, {right_cam_params.fy}, {right_cam_params.cy}],\n"
                           f"[0, 0, 1]]")

    # 写入文件
    with open('camera_parameters.txt', 'w') as file:
        file.write("Left Camera Matrix:\n")
        file.write(left_camera_matrix)
        file.write("Right Camera Matrix:\n")
        file.write(right_camera_matrix)

    # 关闭相机
    zed.close()

if __name__ == "__main__":
    main()
