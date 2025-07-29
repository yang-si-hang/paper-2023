"""使用ffmmpeg将图片序列转换为视频"""
from pathlib import Path
import subprocess

dir_path = Path(__file__).parent
root_path = dir_path.parent

# Path to your image sequence. This example assumes images are named in a sequential format like img001.png, img002.png, etc.
# file_path = "RobotCon/captured_frames/frame_%04d.png"        # Adjust the pattern as needed
file_path = "SOFACode/DataAnalyse/20250727-surface edge/surface-edge.%04d.png"  # Adjust the pattern as needed
image_pattern = root_path / file_path
print(image_pattern)

# Define output video file and frame rate
output_video = dir_path / "surface-control-multipoint.mp4"
fps = "100"

# Build the ffmpeg command
command = [
    "ffmpeg",
    "-y",                       # Overwrite output file if it exists
    "-framerate", fps,          # Set the frame rate
    "-i", image_pattern,        # Input file pattern
    "-vf", "split=2[bg][fg];[bg]drawbox=color=white@1:t=fill:replace=1[bg];[bg][fg]overlay",    "-c:v", "libx264",          # Use the H.264 codec
    "-profile:v", "high",       # Set the profile
    "-crf", "10", #"18",               # Set quality (lower values mean higher quality)
    "-c:v", "libx264",          # Use the H.264 codec
    "-pix_fmt", "yuv420p",      # Set pixel format for better compatibility
    output_video
]

# Execute the command
subprocess.run(command)
# print(f"Video saved successfully as {output_video}")
