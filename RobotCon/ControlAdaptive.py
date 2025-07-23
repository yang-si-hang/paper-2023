"""
使用Adaptive Controller的方法控制软体上的图案变形
created at 2024-08-02 by hsy
"""

import os
import cv2
import numpy as np
import numpy.typing as npt
from RobAction import URROb
from DotsDetectCanny import *
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


def dot_detect_in_actionloop(zed_id, masked_region, image_init):
    """
    Detect the dot in robot action loop
    """
    color_image_bgra = get_image(zed_id, image_init)
    edges = image_process(color_image_bgra, red_range, masked_region)
    dot, area, ellipse = ellipse_fitting(edges)

    filtered_dot_cell = kalman_filter_process(kalman, dot, 3)

    return filtered_dot_cell, color_image_bgra, ellipse


def cal_loss(dot_pos_soft, dot_pos_desired, Ja):
    error = dot_pos_soft - dot_pos_desired
    L = np.linalg.norm(error) ** 2
    dL = 2*error
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

    e = W @ a - delta_pos
    a -= factor * W @ e

    return a.reshape((2*POINTS_NUM, 2))


def main():
    learning_rate = 1.e1
    dt = 1./100
    ja = np.eye(2*POINTS_NUM, 2)

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    camera_id, image_init, window_name = init_camera()
    dot_init = init_region_range(camera_id, image_init, red_range)
    dot_pos_init = dot_in_soft(dot_init, trans_soft, intrinsic)
    dot_pos_desired = dot_pos_init + np.array([0.005, -0.003])
    print("The initial position of the dot in soft object: ", dot_pos_init)
    masked_region = get_mask_region(dot_init, np.zeros((image_height, image_width), dtype=np.uint8), 50)
    kalman.statePre = np.array([[dot_init[0]], [dot_init[1]], [0], [0]], np.float32)
    kalman.statePost = np.array([[dot_init[0]], [dot_init[1]], [0], [0]], np.float32)

    MyRob = URROb(500)
    MyRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    MyRob.start_record_data('experiment_data.csv')

    dot_desired = dot_in_pixel(dot_pos_desired, trans_soft, intrinsic)
    dot_pos = dot_pos_init
    end_movement_np = np.zeros(2)

    dot_pixel_list = []
    loss_list = []
    rob_movement_list = []
    frame_name_list = []

    try:
        for i in range(200):
            dot, color_image, ellipse = dot_detect_in_actionloop(camera_id, masked_region, image_init)
            masked_region = get_mask_region(dot, masked_region, 50)
            print(f'Step {i}: The dot coordinates: {dot}')

            # 显示图像
            cv2.circle(color_image, (int(dot_desired[0]), int(dot_desired[1])), 2, (255, 0, 0), -1)
            if ellipse is None:
                cv2.addText(color_image, f'Not Correct Detection', (10, 20), 'Times New Roman', 1, (0, 0, 0))
            else:
                cv2.ellipse(color_image, ellipse, (0, 255, 0), 1)
            cv2.imshow(window_name, color_image)

            # 保存图像
            frame_name_list = image_save(color_image, i, frame_name_list, output_folder)

            dot_pos_new = dot_in_soft(dot, trans_soft, intrinsic)
            delta_dot_pos = dot_pos_new - dot_pos
            dot_pos = dot_pos_new

            _, loss_tmp, dL_daction = cal_loss(dot_pos, dot_pos_desired, ja)
            ja_new = update_jacobian(end_movement_np, delta_dot_pos, ja)
            ja = ja_new

            # print('The gradient of the action:', soft_obj.grad_y[soft_obj.grasp_particle_list[0]].to_numpy())
            end_speed_np = -learning_rate * dL_daction
            # end_speed = end_speed_np.tolist()
            end_movement_np = end_speed_np * dt
            end_movement = action_compress(end_movement_np).tolist()

            print('Loss:', loss_tmp)
            print(f'The tool movement: {end_movement}')

            # 机器人控制
            MyRob.move_add_movel([-end_movement[0], end_movement[1], 0, 0, 0, 0], a=0.1, v=0.1)            # 设置机械臂末端速度

            # 写入数据
            dot_pixel_list.append(dot)
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