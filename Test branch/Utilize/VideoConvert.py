import os
import glob
import cv2

frame_size = (1080, 720)

# 裁剪参数
x, y, w, h = 204, 184, 383, 354  # 示例裁剪区域

out = cv2.VideoWriter('Video/control_top.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                      (w, h))

files = os.listdir('FigureDemo')
num_png = len(files)

image_files = sorted(glob.glob(os.path.join('FigureDemo', '*.png')))

for i in range(200):
    img = cv2.imread(f'FigureWrite/{i}.png')
    # 裁剪图片
    cropped_img = img[y:y+h, x:x+w]
    out.write(cropped_img)

# for image_file in image_files:
#     img = cv2.imread(image_file)
#     out.write(img)

out.release()
exit(0)