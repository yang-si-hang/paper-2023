"""
使用Adaptive Controller的方法控制软体上的图案变形
created at 2024-08-02 by hsy
"""

import os
import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from RobAction import URROb
from DotsPatternDetect import *
from CVVideo import *
from CoordinateTransform import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

POINTS_NUM = 4

# Adjust these values as needed
lower_red_1 = np.array([0, 65, 100])
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 65, 100])
upper_red_2 = np.array([180, 255, 255])
red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

image_width, image_height = int(1280), int(720)

kalman = kalman_filter_init()
kalmans = [kalman_filter_init() for _ in range(POINTS_NUM)]

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def read_desired_pos(dot_pos_init):
    data = np.load('data/desired_pos.npz')
    pos_desired = data['desired_pos']
    pixel_desired = data['desired_pixel']
    tree = KDTree(dot_pos_init)
    _, indices = tree.query(pos_desired)
    ordered_pos_desired = pos_desired[indices]
    ordered_pixel_desired = pixel_desired[indices]
    print("The desired position of the dot in soft object: ", ordered_pos_desired)
    print('The desired pixel position of the dot in soft object: ', ordered_pixel_desired)

    return ordered_pos_desired, ordered_pixel_desired


def action_compress(vec:npt.NDArray, max_length:float=3.e-4):
    """
    Compress action vector in a safe range
    """
    length = np.linalg.norm(vec)

    if length > max_length:
        factor = max_length / length
        return vec * factor
    else:
        return vec


def pattern_detect_in_actionloop(zed_id, image_init, old_gray, detected_dots_old_np:npt.NDArray):
    """
    Detect the dot in robot action loop
    """
    color_image_bgra = get_image(zed_id, image_init)
    image_gray = bgr2gray(color_image_bgra, [0.2,0.2,0.6])
    dots_now = [None] * POINTS_NUM

    dots_pred = kalman_predict(kalmans)

    if not np.isnan(detected_dots_old_np).any():
        dots_new, st, err = cv2.calcOpticalFlowPyrLK(old_gray, image_gray, detected_dots_old_np, None, **lk_params)
        st = st.flatten()

        dots_new_matched = dots_new[st == 1]
        # print(f'New dots matched: {dots_new_matched}')
        dots_now = match_kalman(dots_new_matched, dots_pred, dots_now)
        # print(f'Now dots: {dots_now.tolist()}')

    # optical flow & kalman filter
    processed_dots = kalman_process(kalmans, dots_now)
    old_gray = image_gray.copy()

    return processed_dots, color_image_bgra, old_gray


def cal_loss(dot_pos_soft, dot_pos_desired, Ja):
    error = dot_pos_soft - dot_pos_desired
    L = np.linalg.norm(error) ** 2
    dL = 2*error.flatten()
    dL_dx = dL @ Ja
    # for idx, ele_idx in enumerate(self.marker_element):
    #     self.dL[ele_idx * dim] = 2 * error[0] * barycentric[idx]
    #     self.dL[ele_idx * dim + 1] = 2 * error[1] * barycentric[idx]

    return error, L, dL_dx


def update_jacobian(delta_action, delta_pos, Ja:npt.NDArray):
    factor = 1.e0

    a = Ja.flatten()
    W = np.zeros((2*POINTS_NUM, 4*POINTS_NUM))
    for idx in range(2*POINTS_NUM):
        W[idx, 2*idx] = delta_action[0]
        W[idx, 2*idx+1] = delta_action[1]

    e = W @ a - delta_pos.flatten()
    a -= factor * W.transpose() @ e

    return a.reshape((2*POINTS_NUM, 2))


def main():
    learning_rate = 1.e1
    dt = 1./100
    ja = np.tile(np.eye(2, 2), (POINTS_NUM, 1))

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    camera_id, image_init, window_name = init_camera()
    dots_init = init_region_range(camera_id, image_init, red_range)
    dot_pos_init = dot_in_soft(dots_init, trans_soft, intrinsic)
    print("The initial position of the dot in soft object: ", dot_pos_init)
    color_image_bgr = get_image(camera_id, image_init)

    dots_pos_desired, dots_pixel_desired = read_desired_pos(dot_pos_init)

    for idx, dot in enumerate(dots_init):
        kalmans[idx].statePre = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)
        kalmans[idx].statePost = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)


    MyRob = URROb(500)
    MyRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    MyRob.start_record_data('experiment_data.csv')

    old_gray = bgr2gray(color_image_bgr, [0.2, 0.2, 0.6])
    dot_desired = dot_in_pixel(dots_pos_desired, trans_soft, intrinsic)
    detected_marker_old = dots_init
    marker_pos = dot_pos_init
    end_movement_np = np.zeros(2)

    dot_pixel_list = []
    loss_list = []
    rob_movement_list = []
    frame_name_list = []

    try:
        for step in range(200):
            print(f'Step: {step} ------------------------------------')
            processed_markers, color_image, old_gray = pattern_detect_in_actionloop(camera_id, image_init, old_gray, detected_marker_old)

            edges = image_process(color_image, red_range)
            detected_dots, area, ellipse = ellipse_fitting(edges)
            detected_marker_old = np.array(detected_dots, dtype=np.float32).copy()
            # print(f'Detected dots: {detected_dots}')
            print(f'Processed markers: {processed_markers.tolist()}')

            # for marker in processed_markers:
            for idx in range(POINTS_NUM):
                cv2.circle(color_image, (int(processed_markers[idx][0]), int(processed_markers[idx][1])), 6, (0, 255, 0), 1)
                cv2.circle(color_image, (int(dots_pixel_desired[idx][0]), int(dots_pixel_desired[idx][1])), 4, (255, 0, 0), -1)

            cv2.imshow(window_name, color_image)

            frame_name_list = image_save(color_image, step, frame_name_list, output_folder)

            marker_pos_new = dot_in_soft(processed_markers, trans_soft, intrinsic)
            print(f'The dot coordinates: {marker_pos_new.tolist()}')
            desired_marker_pos = dot_in_soft(dots_pixel_desired, trans_soft, intrinsic)
            print(f'The desired dot coordinates: {desired_marker_pos.tolist()}')

            delta_dot_pos = marker_pos_new - marker_pos
            marker_pos = marker_pos_new

            error, loss_tmp, dL_daction = cal_loss(marker_pos, dots_pos_desired, ja)
            ja_new = update_jacobian(end_movement_np, delta_dot_pos, ja)
            ja = ja_new
            print(f'Estimated Jacobian: {ja}')

            # print('The gradient of the action:', soft_obj.grad_y[soft_obj.grasp_particle_list[0]].to_numpy())
            end_speed_np = -learning_rate * dL_daction
            # end_speed = end_speed_np.tolist()
            end_movement_np = end_speed_np * dt
            end_movement = action_compress(end_movement_np).tolist()

            print(f'Loss: {loss_tmp:.6e}; Error: {np.array_str(error, precision=6)}')
            print(f'The tool movement: {np.array_str(end_movement, precision=6)}')

            # 机器人控制
            MyRob.move_add_movel([-end_movement[0], end_movement[1], 0, 0, 0, 0], a=0.1, v=0.1)            # 设置机械臂末端速度

            # 写入数据
            dot_pixel_list.append(processed_markers.flatten())
            loss_list.append(loss_tmp)
            rob_movement_list.append(end_movement)

            # Press 'q' to exit the application
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        MyRob.stop_movel()
        MyRob.exit_script()
        MyRob.stop_record_data()

        camera_id.close()
        cv2.destroyAllWindows()

        np.savetxt('dot_pixel_list.csv', np.array(dot_pixel_list), fmt='%.10f', delimiter=',')
        np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
        np.savetxt('rob_movement_list.csv', np.array(rob_movement_list), fmt='%.10f', delimiter=',')

        # 将保存的图像转换为视频
        # image_to_video(frame_name_list, 'DataAnalyse/output_video.mp4')


if __name__ == '__main__':
    main()