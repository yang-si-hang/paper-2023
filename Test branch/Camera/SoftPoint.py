"""

"""

import os, sys
import cv2
import numpy as np
import open3d as o3d
from pyorbbecsdk import Config
from pyorbbecsdk import OBError, OBAlignMode
from pyorbbecsdk import OBSensorType, OBFormat
from pyorbbecsdk import Pipeline, FrameSet
from pyorbbecsdk import VideoStreamProfile

# 设置工作目录为当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.orbbec_utils import frame_to_bgr_image


def main():
    # 初始化管道
    pipeline = Pipeline()
    config = Config()
    
    # 配置流
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)

    config.set_align_mode(OBAlignMode.SW_MODE)          # mega只能设置为软解析度对齐
    pipeline.enable_frame_sync()

    pipeline.start(config)

    while True:
        frames: FrameSet = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        
        # 获取颜色和深度帧
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        depth_frame = frames.get_depth_frame()
        if depth_frame is None:
            continue
        color_image = frame_to_bgr_image(color_frame)
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(depth_frame.get_height(), depth_frame.get_width())
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        
        # HSV掩码
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        # 提取点云
        indices = np.where(red_mask > 0)
        z_values = depth_data[indices]
        fx, fy = 600, 600
        cx, cy = depth_frame.get_width() / 2, depth_frame.get_height() / 2
        point_cloud = [( (x - cx) * z / fx, (y - cy) * z / fy, z ) for y, x, z in zip(indices[0], indices[1], z_values)]
        
        # 保存点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud("red_soft.ply", pcd)
        
        # 显示结果
        cv2.imshow("Red Masked Object", cv2.bitwise_and(color_image, color_image, mask=red_mask))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()

if __name__ == "__main__":
    main()
