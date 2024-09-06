"""
Zed相机的通用函数
"""

import os
import cv2
import pyzed.sl as sl



def init_camera(resolution:int, fps:int):
    # 创建一个相机对象
    zed = sl.Camera()

    # 设置相机配置
    init_params = sl.InitParameters()
    if resolution == 2000:
        resolution_set = sl.RESOLUTION.HD2K
    elif resolution == 1080:
        resolution_set = sl.RESOLUTION.HD1080
    elif resolution == 720:
        resolution_set = sl.RESOLUTION.HD720
    else:
        raise NameError('Resolution not supported')
    init_params.camera_resolution = resolution_set
    # init_params.camera_resolution = sl.RESOLUTION.HD720  # 设置相机分辨率为HD720
    init_params.camera_fps = fps  # 设置相机的帧率为30 fps

    # 打开相机
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # 创建一个图像矩阵对象
    image = sl.Mat()
    return zed, image


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


def image_save(image, i, frame_name_list, output_folder:str):
    # 保存当前帧为图像文件
    frame_filename = os.path.join(output_folder, f'frame_{i:04d}.png')
    cv2.imwrite(frame_filename, image)
    frame_name_list.append(frame_filename)
    return  frame_name_list


def image_to_video(frame_name_list, video_filename:str='output_video.mp4'):
    # 合成视频
    frame = cv2.imread(frame_name_list[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame_file in frame_name_list:
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()



if __name__ == '__main__':
    zed, image = init_camera(1080, 30)
    cv2.imwrite('zed_image.png', get_image(zed, image))