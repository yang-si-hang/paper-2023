"""
This code for detecting red dots.
"""


import cv2
import numpy as np
import pyk4a
from pyk4a import PyK4A, Config, K4AException

# Initialize Azure Kinect camera
k4a = PyK4A(Config(color_resolution=pyk4a.ColorResolution.RES_720P))

# Define the lower and upper bounds for the red color in HSV
lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 43, 46])
upper_red_2 = np.array([180, 255, 255])

try:
    k4a.start()
    while True:
        # Capture a frame from the Azure Kinect camera
        capture = k4a.get_capture()
        if capture is not None:
            color_image = capture.color
            if color_image is not None:
                # Convert color image to OpenCV format
                color_image = color_image[:, :, 0:3]
                color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                color_image_bgr = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

                # Convert the image to the HSV color space (for red detection)
                hsv = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2HSV)

                # Create a mask to extract the red tissue
                red_mask_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
                red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

                red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
                # cv2.imwrite('red_mask.png', red_mask)

                # 定义一个小的闭运算核
                kernel = np.ones((9, 9), np.uint8)  # 7x7大小的正方形核。你可以根据空洞的大小调整这个。

                # 使用闭运算，可能会破坏掉边界
                closed_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

                # Apply the mask to the original image
                masked_image = cv2.bitwise_and(color_image_bgr, color_image_bgr, mask=closed_mask)
                # cv2.imwrite('masked_image.png', masked_image)

                # Convert the image to grayscale
                gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
                # cv2.imwrite('gray.png', gray)

                # Apply a threshold to create a binary image
                _, binary_image = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
                # cv2.imwrite('binary_image.png', binary_image)

                binary_dots = cv2.bitwise_and(binary_image, closed_mask)

                kernel = np.ones((3, 3), np.uint8)
                binary_dots = cv2.morphologyEx(binary_dots, cv2.MORPH_CLOSE, kernel)

                # Find contours in the mask
                contours, _ = cv2.findContours(binary_dots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Define a list to store the coordinates of detected black dots
                dot_coordinates = []

                # Define a minimum area for a contour to be considered a dot
                min_dot_area = 3

                # Loop through the contours and filter out small ones (potential dots)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > min_dot_area:
                        # Get the centroid of the contour
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                            dot_coordinates.append((cX, cY))

                # Draw circles around the detected dots on the original image
                for (cX, cY) in dot_coordinates:
                    cv2.circle(color_image_bgr, (cX, cY), 5, (0, 0, 0), -1)  # Black circles

                # Display the result (with circles around detected dots on red tissue)
                cv2.imshow("Detected Dots on Red Tissue", color_image_bgr)

                # Press 'q' to exit the application
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

except K4AException as e:
    print(f"Error: {e}")

finally:
    k4a.stop()
    cv2.destroyAllWindows()

    # Print the coordinates of detected dots
    print("Detected Dot Coordinates on Red Tissue:")
    for i, (cX, cY) in enumerate(dot_coordinates, 1):
        print(f"Dot {i}: ({cX}, {cY})")