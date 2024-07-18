"""
包含存图像和转换为视频等opencv库的操作

"""

import os
import cv2

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