"""Orbbec相机使用函数
"""
import os, sys
import cv2
from pyorbbecsdk import Config, OBError, OBSensorType, OBFormat, Pipeline, FrameSet, VideoStreamProfile
from pyorbbecsdk import Context, OBLogLevel
# 添加根目录到 sys.path（跨目录导入模块）
script_dir = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
from Utilize.orbbec_utils import frame_to_bgr_image


ESC_KEY = 27

def initialize_orbbec_camera(width: int, height: int, fps: int) -> Pipeline:
    """
    初始化Orbbec相机并设置指定参数
    :param width: 分辨率宽度
    :param height: 分辨率高度
    :param fps: 帧率
    :return: 初始化成功的Pipeline对象, 失败返回None
    """
    Context.set_logger_to_console(OBLogLevel.NONE)
    config = Config()
    pipeline = Pipeline()
    
    try:
        profile_list = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        
        try:
            # 尝试获取指定参数的配置
            color_profile = profile_list.get_video_stream_profile(width, height, OBFormat.RGB, fps)
            print(f"成功设置分辨率: {width}x{height}@{fps}fps")
        except OBError:
            # 获取失败时使用默认配置
            print("使用默认配置")
            color_profile = profile_list.get_default_video_stream_profile()
        
        config.enable_stream(color_profile)
        pipeline.start(config)
        return pipeline
    
    except Exception as e:
        print(f"初始化失败: {str(e)}")
        return None

def get_color_frame(pipeline: Pipeline) -> [cv2.typing.MatLike, None]:
    """
    从已初始化的相机获取彩色帧
    :param pipeline: 初始化后的Pipeline对象
    :return: OpenCV格式的BGR图像, 失败返回None
    """
    try:
        frames: FrameSet = pipeline.wait_for_frames(100)
        if frames is None:
            return None
            
        color_frame = frames.get_color_frame()
        if color_frame is None:
            return None
            
        return frame_to_bgr_image(color_frame)
    
    except Exception as e:
        print(f"获取帧失败: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    # 初始化相机（示例参数：1280x720@30fps）
    camera_pipeline = initialize_orbbec_camera(1280, 720, 30)
    
    if not camera_pipeline:
        print("相机初始化失败，请检查连接和参数设置")
        exit(1)
    
    try:
        while True:
            frame = get_color_frame(camera_pipeline)
            if frame is not None:
                cv2.imshow("Orbbec Camera", frame)
            
            if cv2.waitKey(1) in (ord('q'), ESC_KEY):
                break
    finally:
        camera_pipeline.stop()
        cv2.destroyAllWindows()