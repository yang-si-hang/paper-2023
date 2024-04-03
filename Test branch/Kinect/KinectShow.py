"""
Read kinect image and show
"""

import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A

def main():
    cv2.namedWindow('Azure Kinect Capture', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Azure Kinect Capture', 1080, 720)

    # Initialize Azure Kinect camera
    config = Config()
    config.color_resolution = pyk4a.ColorResolution.RES_1080P
    k4a = PyK4A(config)

    # Start camera capture
    k4a.start()

    while True:
        # Capture a frame
        capture = k4a.get_capture()
        if capture is not None:
            color_image = capture.color
            if color_image is not None:
                # Convert color image to OpenCV format
                color_image = color_image[:, :, :3]
                # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

                # Display the image (you can also save it)
                cv2.imshow("Azure Kinect Capture", color_image)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close OpenCV window
    k4a.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()