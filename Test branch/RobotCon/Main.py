"""
Obtain the position of the black on red soft object.
"""

import time
import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
from ControlSimulation import *
from RobAction import URROb
from DotsDetectCanny import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the lower and upper bounds for the red color in HSV
lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 43, 46])
upper_red_2 = np.array([180, 255, 255])
red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

image_width, image_height = 1280, 720


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


def image_save(image, i, frame_name_list):
    # 保存当前帧为图像文件
    frame_filename = os.path.join(output_folder, f'frame_{i:04d}.png')
    cv2.imwrite(frame_filename, image)
    frame_name_list.append(frame_filename)
    return  frame_name_list


def image_to_video(frame_name_list, video_filename:str='output_video.mp4'):
    # 合成视频
    frame = cv2.imread(frame_name_list[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for frame_file in frame_name_list:
        frame = cv2.imread(frame_file)
        video.write(frame)

    video.release()


def dot_detect_in_actionloop(zed_id, masked_region, image_init):
    """
    Detect the dot in robot action loop
    """
    color_image_bgra = get_image(zed_id, image_init)
    edges = image_process(color_image_bgra, red_range, masked_region)
    dot, area, ellipse = ellipse_fitting(edges)
    # if show_flag:
    #     image_show(color_image_bgra, ellipse)

    return dot, color_image_bgra, ellipse


def main():
    obj_shape = [0.1, 0.1]
    obj_seed_size = 0.01
    learning_rate = 1.e1

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    camera_id, image_init, window_name = init_camera()
    # 运行一次，获得标记点的初始位置
    dot_init = init_region_range(camera_id, image_init, red_range)
    dot_pos_init = dot_in_soft(dot_init, trans_soft, intrinsic)
    print("The initial position of the dot in soft object: ", dot_pos_init)
    masked_region = get_mask_region(dot_init, np.zeros((image_height, image_width), dtype=np.uint8), 50)

    class MyObject(SoftObject):
        def __init__(self, shape, seed_size, contact_idx):
            super().__init__(shape, seed_size, contact_idx)
            self.dt = 1./100
            self.marker_element = None
            self.barycentric = None
            self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=1)
            self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=1)
            self.dot_pos_desired = ti.Vector.field(2, dtype=ti.f64, shape=1)
            self.dot_pos[0] = dot_pos_init
            self.dot_pos_init[0] = dot_pos_init
            self.dot_pos_desired[0] = self.dot_pos_init[0] + ti.Vector([0.002, 0.])

            self.marker_element_get()


        def marker_element_get(self):
            mesh_nodes = self.tri.points
            element_np = find_element(self.tri, self.dot_pos_init[0])
            if element_np is not None:
                self.marker_element = list(element_np)
                barycentric_init = feature_barycentric_coordinates(self.dot_pos_init[0], mesh_nodes[element_np])
                self.barycentric = barycentric_init
                print("The dot is in element: ", element_np)
                print("The barycentric coordinates are: ", barycentric_init)
            else:
                print("The dot is not in the mesh object.")


        def construct_L_mrker(self):
            """
            Construct the L with marker that doesn't position on the node.
            """
            dim = self.dim
            barycentric = self.barycentric
            desired_pos = self.dot_pos_desired[0]
            current_pos = self.dot_pos[0]
            error = current_pos - desired_pos
            L = error.norm() ** 2
            for idx, ele_idx in enumerate(self.marker_element):
                self.dL[ele_idx * dim] = 2 * (current_pos[0] - desired_pos[0]) * barycentric[idx]
                self.dL[ele_idx * dim + 1] = 2 * (current_pos[1] - desired_pos[1]) * barycentric[idx]

            return error, L


        def construct_L_camera(self, dot_pos_soft):
            """
            Construct the L with camera image detect.
            """
            dim = self.dim
            barycentric = self.barycentric
            error = dot_pos_soft - self.dot_pos_desired[0]
            L = error.norm() ** 2
            for idx, ele_idx in enumerate(self.marker_element):
                self.dL[ele_idx * dim] = 2 * error[0] * barycentric[idx]
                self.dL[ele_idx * dim + 1] = 2 * error[1] * barycentric[idx]

            return error, L


        def diff_pd(self, itr_num):
            # compute Jacobian matrix by DiffPD
            self.partial_p()
            dA = self.rhs_dA.to_numpy()
            par_L = self.dL.to_numpy()
            z_np = self.z.to_numpy()
            for itr in ti.static(range(itr_num)):
                rhs_diff_np = dA @ z_np + par_L
                z_new_np = self.pre_fact_lhs_solve(rhs_diff_np)
                z_np = z_new_np
            self.z.from_numpy(z_np)


        @ti.kernel
        def compute_grad_y(self):
            for i in range(self.PARTICLE_NUM):
                idx0, idx1 = i*self.dim, i*self.dim+1
                self.grad_y[i].x = self.z[idx0]*self.node_mass[i]/self.dt**2
                self.grad_y[i].y = self.z[idx1]*self.node_mass[i]/self.dt**2


        def substep(self, step_num):
            # PD forward simulation
            self.construct_sn()
            self.warm_up()
            for itr in ti.static(range(self.solve_iteration)):
                self.local_solve()
                self.construct_rhs()
                rhs_np = self.rhs.to_numpy()
                node_pos_new_np = self.pre_fact_lhs_solve(rhs_np)
                self.update_pos_new(node_pos_new_np)

            self.update_vel_pos()


        def actuate_action(self, contact_speed):
            self.GRASP_VEL[0] = contact_speed


        def compute_gradient(self, dot_soft):
            error, loss_tmp = self.construct_L_camera(dot_soft)
            self.loss = loss_tmp
            self.diff_pd(10)
            self.compute_grad_y()

            return loss_tmp

    MyRob = URROb(500)
    MyRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    MyRob.start_record_data('experiment_data2.csv')

    soft_obj = MyObject(obj_shape, obj_seed_size, [10])
    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    dot_desired = dot_in_pixel(soft_obj.dot_pos_desired[0], trans_soft, intrinsic)

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
            cv2.ellipse(color_image, ellipse, (0, 255, 0), 1)
            cv2.imshow(window_name, color_image)

            # 保存图像
            frame_name_list = image_save(color_image, i, frame_name_list)

            dot_pos = dot_in_soft(dot, trans_soft, intrinsic)
            print(f'Step {i}: The dot coordinates: {dot}')

            soft_obj.substep(1)
            loss_tmp = soft_obj.compute_gradient(dot_pos)

            # np.savetxt('dL.txt', soft_obj.dL.to_numpy())
            # np.savetxt('grad_y.txt', soft_obj.grad_y.to_numpy())

            # print('The gradient of the action:', soft_obj.grad_y[soft_obj.grasp_particle_list[0]].to_numpy())
            end_speed_np = -learning_rate * soft_obj.grad_y[soft_obj.grasp_particle_list[0]].to_numpy()
            # end_speed = end_speed_np.tolist()
            end_movement_np = end_speed_np * soft_obj.dt
            end_movement = action_compress(end_movement_np).tolist()

            soft_obj.actuate_action(end_speed_np)
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
        image_to_video(frame_name_list, 'output_video.mp4')


if __name__ == '__main__':
    main()