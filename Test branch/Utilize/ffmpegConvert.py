"""使用ffmmpeg将图片序列转换为视频"""

import os, sys
import subprocess
# 添加根目录到 sys.path（跨目录导入模块）
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)  # 修改当前工作目录
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)
print(os.getcwd())

# Path to your image sequence. This example assumes images are named in a sequential format like img001.png, img002.png, etc.
file_path = "RobotCon/captured_frames/frame_%04d.png"        # Adjust the pattern as needed
image_pattern = os.path.join(root_path, file_path)
print(image_pattern)

# Define output video file and frame rate
output_video = "output.mp4"
fps = "20"

# Build the ffmpeg command
command = [
    "ffmpeg",
    "-y",                       # Overwrite output file if it exists
    "-framerate", fps,          # Set the frame rate
    "-i", image_pattern,        # Input file pattern
    # "-vf", "split=2[bg][fg];[bg]drawbox=color=white@1:t=fill:replace=1[bg];[bg][fg]overlay",    "-c:v", "libx264",          # Use the H.264 codec
    "-profile:v", "high",       # Set the profile
    "-crf", "18",               # Set quality (lower values mean higher quality)
    "-pix_fmt", "yuv420p",      # Set pixel format for better compatibility
    output_video
]

# Execute the command
subprocess.run(command)
# print(f"Video saved successfully as {output_video}")
