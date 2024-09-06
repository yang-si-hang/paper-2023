"""
实物实验，使用视觉伺服的方法控制目标点，选择固定的雅可比矩阵
created at 2024-7-10 by hsy
"""

import time
import os
import cv2
import numpy as np
from RobAction import URROb
from DotsDetectCanny import *
from CVVideo import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the lower and upper bounds for the red color in HSV
lower_red_1 = np.array([0, 65, 100])  # Adjust these values as needed
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 65, 100])
upper_red_2 = np.array([180, 255, 255])
red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

image_width, image_height = 1280, 720

kalman = kalman_filter_init()

Jm = np.eye(2, 2)


def pixel_to_camera_coordinates(pixel, K_inv):
    """
    将图像坐标转换为相机坐标系中的射线
    """
    pixel_homogeneous = np.append(pixel, 1.0)
    camera_ray = K_inv @ pixel_homogeneous
    return camera_ray


def line_plane_intersection(line_dir, plane_normal, plane_point):
    """
    计算直线与平面的交点
    """
    d = np.dot(plane_normal, plane_point)
    t = d / np.dot(plane_normal, line_dir)          # 这里不需要对line_dir进行归一化
    intersection_point = t * line_dir
    return intersection_point


def dot_in_soft(dot_pixel, trans_soft, intrinsic):
    """
    将标记点的像素坐标转换到软体坐标系
    :return:
    """
    intrisic_inv = np.linalg.inv(intrinsic)
    dot_camera = pixel_to_camera_coordinates(dot_pixel, intrisic_inv)

    plane_normal = trans_soft[:3, 2]            # Z轴方向
    plane_point = trans_soft[:3, 3]             # 变换矩阵的平移部分

    dot_soft = line_plane_intersection(dot_camera, plane_normal, plane_point)       # 在相机坐标系下的三维位置
    dot_soft = np.linalg.inv(trans_soft) @ np.append(dot_soft, 1.)

    return dot_soft[0:2]


def dot_in_pixel(dot_soft, trans_soft, intrinsic):
    """
    将软体坐标系中的点转换到像素坐标
    """
    dot_soft = np.append(dot_soft, 0.)
    dot_camera = trans_soft @ np.append(dot_soft, 1.)
    dot_pixel = intrinsic @ dot_camera[:3]
    return dot_pixel[:2] / dot_pixel[2]


def feature_barycentric_coordinates(p, mesh_nodes):
    """
    Compute the barycentric coordinates of a point p with respect to the triangle p0, p1, p2
    """
    p0, p1, p2 = mesh_nodes
    v0 = p1 - p0
    v1 = p2 - p0
    v2 = p - p0
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return np.array([u, v, w])


def find_element(tri, dot_pos):
    """
    Find the element which contains the dot
    :param tri:
    :param dot_pos:
    :return: element index
    """
    # 查找包含点的三角形
    simplex = tri.find_simplex(dot_pos)

    if simplex != -1:
        # 返回包含点的三角形的顶点索引
        return tri.simplices[simplex]
    else:
        return None


def action_compress(vec:np.ndarray, max_length:float=3.e-4):
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


def cal_loss(dot_pos_soft, dot_pos_desired):
    error = dot_pos_soft - dot_pos_desired
    L = np.linalg.norm(error) ** 2
    dL = 2*error
    dL_dx = dL @ Jm
    # for idx, ele_idx in enumerate(self.marker_element):
    #     self.dL[ele_idx * dim] = 2 * error[0] * barycentric[idx]
    #     self.dL[ele_idx * dim + 1] = 2 * error[1] * barycentric[idx]

    return error, L, dL_dx


def main():
    learning_rate = 1.e1
    dt = 1./100

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    camera_id, image_init, window_name = init_camera()
    # 运行一次，获得标记点的初始位置
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

    dot_pixel_list = []
    loss_list = []
    rob_movement_list = []
    frame_name_list = []

    try:
        for i in range(200):
            dot, color_image, ellipse = dot_detect_in_actionloop(camera_id, masked_region, image_init)
            masked_region = get_mask_region(dot, masked_region, 50)

            # 显示图像
            cv2.circle(color_image, (int(dot_desired[0]), int(dot_desired[1])), 2, (255, 0, 0), -1)
            if ellipse is None:
                cv2.addText(color_image, f'Not Correct Detection', (10, 20), 'Times New Roman', 1, (0, 0, 0))
            else:
                cv2.ellipse(color_image, ellipse, (0, 255, 0), 1)
            cv2.imshow(window_name, color_image)

            # 保存图像
            frame_name_list = image_save(color_image, i, frame_name_list, output_folder)

            dot_pos = dot_in_soft(dot, trans_soft, intrinsic)
            print(f'Step {i}: The dot coordinates: {dot}')

            _, loss_tmp, dL_daction = cal_loss(dot_pos, dot_pos_desired)

            # np.savetxt('dL.txt', soft_obj.dL.to_numpy())
            # np.savetxt('grad_y.txt', soft_obj.grad_y.to_numpy())

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
        image_to_video(frame_name_list, 'DataAnalyse/output_video.mp4')


if __name__ == '__main__':
    main()