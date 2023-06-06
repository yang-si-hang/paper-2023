import os
import glob
import cv2

frame_size = (512, 512)
out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50,
                      frame_size)

files = os.listdir('FigureDemo')
num_png = len(files)

image_files = sorted(glob.glob(os.path.join('FigureDemo', '*.png')))

# for i in range(num_png):
#     img = cv2.imread(f'FigureDemo/{10*(i+1)}.png')
#     out.write(img)

for image_file in image_files:
    img = cv2.imread(image_file)
    out.write(img)

out.release()
exit(0)