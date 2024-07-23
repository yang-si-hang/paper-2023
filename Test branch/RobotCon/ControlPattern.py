"""
实物实验,使用Strain Constraint & Volume Constraint的DiffPD进行控制
created at 2024-07-17 by hsy
"""


import time
import os
import cv2
import numpy as np
from scipy.spatial import KDTree
from ControlSimulation import *
from RobAction import URROb
from DotsPatternDetect import *
from CVVideo import *
from CoordinateTransform import *

output_folder = 'captured_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# pattern中有几个点，重新赋值
POINTS_NUM:int = 4

# Define the lower and upper bounds for the red color in HSV
lower_red_1 = np.array([0, 43, 46])  # Adjust these values as needed
upper_red_1 = np.array([10, 255, 255])

lower_red_2 = np.array([156, 43, 46])
upper_red_2 = np.array([180, 255, 255])
red_range = [lower_red_1, upper_red_1, lower_red_2, upper_red_2]

image_width, image_height = int(1280), int(720)

kalman = kalman_filter_init()
kalmans = [kalman_filter_init() for _ in range(POINTS_NUM)]

lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

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


class MyObject(SoftObject):
    def __init__(self, shape, seed_size, marker_pos_init, contact_idx: list):
        super().__init__(shape, seed_size, contact_idx)
        self.dt = 1. / 100
        self.loss = np.zeros(POINTS_NUM)
        self.marker_elements = ti.Vector.field(3, dtype=ti.i32, shape=POINTS_NUM)
        self.barycentrics = ti.Vector.field(3, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_init = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos_desired = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pixel_desired = ti.Vector.field(2, dtype=ti.f64, shape=POINTS_NUM)
        self.dot_pos.from_numpy(marker_pos_init)
        self.dot_pos_init.from_numpy(marker_pos_init)

        self.get_marker_element()
        self.read_desired_pos()

    def get_marker_element(self):
        mesh_nodes = self.tri.points
        for i in range(POINTS_NUM):
            element_np = find_element(self.tri, self.dot_pos_init[i])
            if element_np is not None:
                self.marker_elements[i] = element_np
                self.barycentrics[i] = feature_barycentric_coordinates(self.dot_pos_init[i], mesh_nodes[element_np])
                print("The dot is in element: ", element_np)
                print("The barycentric coordinates are: ", self.barycentrics[i])
            else:
                print("The dot is not in the mesh object.")


    def read_desired_pos(self):
        data = np.load('data/desired_pos.npz')
        pos_desired = data['desired_pos']
        pixel_desired = data['desired_pixel']
        tree = KDTree(self.dot_pos_init.to_numpy())
        _, indices = tree.query(pos_desired)
        ordered_pos_desired = pos_desired[indices]
        ordered_pixel_desired = pixel_desired[indices]
        self.dot_pos_desired.from_numpy(ordered_pos_desired)
        self.dot_pixel_desired.from_numpy(ordered_pixel_desired)
        print("The desired position of the dot in soft object: ", self.dot_pos_desired)
        print('The desired pixel position of the dot in soft object: ', self.dot_pixel_desired)


    def constrcut_L_model(self):
        """
        使用Model得到的marker_pos计算loss
        :return:
        """
        dim:int = 2
        error = np.zeros((POINTS_NUM, 2))
        for marker_i in range(POINTS_NUM):
            barycentric = self.barycentrics[marker_i]
            desired_pos = self.dot_pos_desired[marker_i]
            current_pos = self.dot_pos[marker_i]
            error[marker_i] = (current_pos - desired_pos).to_numpy()
            for idx, ele_idx in enumerate(self.marker_elements[marker_i]):
                self.dL[ele_idx * dim] += 2 * (current_pos[0] - desired_pos[0]) * barycentric[idx]
                self.dL[ele_idx * dim + 1] += 2 * (current_pos[1] - desired_pos[1]) * barycentric[idx]

        return error, np.linalg.norm(error, axis=1) ** 2


    def construct_L_soft(self, marker_pos_soft:npt.NDArray):
        """
        在Soft的二维坐标系上计算loss
        :param marker_pos_soft: shape: (POINTS_NUM, 2)
        """
        dim = self.dim
        error = np.zeros((POINTS_NUM, 2), dtype=np.float64)
        self.dL.fill(0.)
        for marker_i in range(POINTS_NUM):
            barycentric = self.barycentrics[marker_i]
            desired_pos = self.dot_pos_desired[marker_i]
            current_pos = marker_pos_soft[marker_i]
            error[marker_i] = (current_pos - desired_pos).to_numpy()
            print(f'Soft Coordinate Error： {error[marker_i]}')
            for idx, ele_idx in enumerate(self.marker_elements[marker_i]):
                self.dL[ele_idx * dim] += 2 * (current_pos[0] - desired_pos[0]) * barycentric[idx]
                self.dL[ele_idx * dim + 1] += 2 * (current_pos[1] - desired_pos[1]) * barycentric[idx]

        return error, np.linalg.norm(error, axis=1) ** 2


    def diff_pd(self, itr_num:int):
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
            idx0, idx1 = i * self.dim, i * self.dim + 1
            self.grad_y[i].x = self.z[idx0] * self.node_mass[i] / self.dt ** 2
            self.grad_y[i].y = self.z[idx1] * self.node_mass[i] / self.dt ** 2


    def substep(self, step_num:int):
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
        error, loss_tmp = self.construct_L_soft(dot_soft)
        self.loss = loss_tmp
        self.diff_pd(10)
        self.compute_grad_y()

        return loss_tmp


def main():
    obj_shape = [0.14, 0.14]
    obj_seed_size = 0.01
    learning_rate = 1.e1

    camera_data = np.load('data/camera_param.npz')
    trans_soft = camera_data['matrix1']
    intrinsic = camera_data['matrix2']

    camera_id, image_init, window_name = init_camera()
    dots_init = init_region_range(camera_id, image_init, red_range)
    dot_pos_init = dot_in_soft(dots_init, trans_soft, intrinsic)
    print("The initial position of the dot in soft object: ", dot_pos_init)
    color_image_bgr = get_image(camera_id, image_init)

    for idx, dot in enumerate(dots_init):
        kalmans[idx].statePre = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)
        kalmans[idx].statePost = np.array([[dot[0]], [dot[1]], [0], [0]], dtype=np.float32)

    MyRob = URROb(500)
    MyRob.record_variable = ['timestamp', 'actual_TCP_pose', 'actual_TCP_speed']
    MyRob.start_record_data('experiment_data.csv')

    soft_obj = MyObject(obj_shape, obj_seed_size, dot_pos_init, [15])
    soft_obj.precomputation()
    lhs_np = soft_obj.lhs.to_numpy()
    s_lhs_np = sparse.csc_matrix(lhs_np)
    soft_obj.pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs_np)

    marker_pixel_list = []
    loss_list = []
    rob_movement_list = []
    frame_name_list = []

    old_gray = bgr2gray(color_image_bgr, [0.2, 0.2, 0.6])
    detected_marker_old = dots_init
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
                cv2.circle(color_image, (int(soft_obj.dot_pixel_desired[idx][0]), int(soft_obj.dot_pixel_desired[idx][1])), 4, (255, 0, 0), -1)

            cv2.imshow(window_name, color_image)

            frame_name_list = image_save(color_image, step, frame_name_list, output_folder)

            marker_pos = dot_in_soft(processed_markers, trans_soft, intrinsic)
            print(f'The dot coordinates: {marker_pos.tolist()}')
            desired_marker_pos = dot_in_soft(soft_obj.dot_pixel_desired.to_numpy(), trans_soft
                                             , intrinsic)
            print(f'The desired dot coordinates: {desired_marker_pos.tolist()}')

            soft_obj.substep(1)
            loss_tmp = soft_obj.compute_gradient(marker_pos)

            end_speed_np = -learning_rate * soft_obj.grad_y[soft_obj.grasp_particle_list[0]].to_numpy()
            # end_speed = end_speed_np.tolist()
            end_movement_np = end_speed_np * soft_obj.dt
            end_movement = action_compress(end_movement_np).tolist()

            soft_obj.actuate_action(end_speed_np)
            print('Loss:', np.sum(loss_tmp))
            print(f'The tool movement: {end_movement}')

            # 机器人控制
            MyRob.move_add_movel([-end_movement[0], end_movement[1], 0, 0, 0, 0], a=0.1, v=0.1)            # 设置机械臂末端速度

            # 写入数据
            marker_pixel_list.append(processed_markers.flatten())
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

        np.savetxt('dot_pixel_list.csv', np.array(marker_pixel_list), fmt='%.10f', delimiter=',')
        np.savetxt('loss_list.csv', np.array(loss_list), fmt='%.10f', delimiter=',')
        np.savetxt('rob_movement_list.csv', np.array(rob_movement_list), fmt='%.10f', delimiter=',')

        # 将保存的图像转换为视频
        image_to_video(frame_name_list, 'DataAnalyse/output_video.mp4')

if __name__ == '__main__':
    main()