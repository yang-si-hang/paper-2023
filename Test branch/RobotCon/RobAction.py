"""
Give action to the robot
"""

import time
import numpy as np
import rtde_control
import rtde_receive
import copy


class URROb:
    def __init__(self, control_frequency, ur_ip='192.168.253.101'):
        self.UR_IP = ur_ip
        self.rtde_frequency = control_frequency
        self.record_variable = []

        self.rtde_c = rtde_control.RTDEControlInterface(self.UR_IP, self.rtde_frequency)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.UR_IP, self.rtde_frequency)


    def move_speedj(self, v, a=5., dt=1./500):
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.speedJ(v, a, dt)
        self.rtde_c.waitPeriod(t_start)
        return
    

    def move_speedl(self, v, a=1., dt=1./500):
        """
        机器人末端速度不能稳定在给定速度
        """
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.speedL(v, a, dt)
        self.rtde_c.waitPeriod(t_start)
        return


    def move_add_servoj(self, joint_add):
        dt = 1./self.rtde_frequency
        pose_start = self.rtde_r.getTargetTCPPose()
        t_start = self.rtde_c.initPeriod()
        pose_end = copy.deepcopy(pose_start)
        pose_end = [pose_end[i] + joint_add[i] for i in range(6)]
        # velocity & accleration 没有效果，time与最终位置几乎无关，lookahead_time相当于前瞻时间，gain越大越平滑
        # pose_end可以与pose_init有较大的差距
        self.rtde_c.servoL(pose_end, 0.5, 0.5, dt, 0.1, 300)
        self.rtde_c.waitPeriod(t_start)
        self.rtde_c.servoStop()


    def move_add_servol(self, pose_add):
        dt = 1./self.rtde_frequency
        pose_start = self.rtde_r.getTargetTCPPose()
        t_start = self.rtde_c.initPeriod()
        pose_end = copy.deepcopy(pose_start)
        pose_end = [pose_end[i] + pose_add[i] for i in range(6)]
        self.rtde_c.servoL(pose_end, 0.5, 0.5, 0.1, 0.05, 2000)
        self.rtde_c.waitPeriod(t_start)


    def move_add_movej(self, joint_add):
        joint_start = self.rtde_r.getActualQ()
        joint_end = copy.deepcopy(joint_start)
        joint_end = [joint_start[i] + joint_add[i] for i in range(6)]
        self.rtde_c.moveJ(joint_end, 0.5, 0.5, False)


    def move_add_movel(self, pose_add, a=0.5, v=0.5):
        pose_start = self.rtde_r.getActualTCPPose()
        pose_end = copy.deepcopy(pose_start)
        pose_end = [pose_end[i] + pose_add[i] for i in range(6)]
        # True是非阻塞，False是阻塞
        self.rtde_c.moveL(pose_end, a, v, False)


    def move_add_movej_ik(self, pose_add):
        """
        Linear in joint space, move to the target TCP pose
        """
        pose_start = self.rtde_r.getActualTCPPose()
        pose_end = copy.deepcopy(pose_start)
        pose_end = [pose_end[i] + pose_add[i] for i in range(6)]
        self.rtde_c.moveJ_IK(pose_end, 0.5, 0.5, False)


    def move_add_movel_fk(self, joint_add):
        """
        Linear in tool space, move to the target joint position
        """
        joint_start = self.rtde_r.getActualQ()
        joint_end = copy.deepcopy(joint_start)
        joint_end = [joint_start[i] + joint_add[i] for i in range(6)]
        self.rtde_c.moveL_FK(joint_end, 0.5, 0.5, False)


    def get_joint(self, print_flag: bool = True):
        actual_q = self.rtde_r.getActualQ()
        if print_flag:
            time_stamp = self.rtde_r.getTimestamp()
            print(f'Time stamp: {time_stamp}. The joint position is: {actual_q}')
        return np.array(actual_q)


    def get_pose(self, print_flag: bool = True):
        actual_pose = self.rtde_r.getTargetTCPPose()
        if print_flag:
            time_stamp = self.rtde_r.getTimestamp()
            print(f'Time stamp: {time_stamp}. The TCP Pose is: {actual_pose}')
        return np.array(actual_pose)


    def start_record_data(self, output_file='robot_data.csv'):
        variable = self.record_variable
        self.rtde_r.startFileRecording(output_file, variable)


    def stop_record_data(self):
        self.rtde_r.stopFileRecording()


    def stop_speed(self):
        self.rtde_c.speedStop()


    def stop_movel(self):
        self.rtde_c.stopL()


    def stop_movej(self):
        self.rtde_c.stopJ()

    def exit_script(self):
        self.rtde_c.stopScript()


if __name__ == '__main__':
    """
    freq = 500
    MyUR = URROb(freq)
    MyUR.record_variable = ['timestamp', 'target_q', 'actual_q', 'target_qd', 'actual_qd', 'target_qdd']
    # MyUR.record_variable = ['timestamp', 'actual_TCP_pose', 'target_TCP_speed', 'actual_TCP_speed']
    MyUR.start_record_data()
    pose_init = MyUR.get_joint()
    # for i in range(500):
    MyUR.move_speedj([0, 0, 0., 0, 0., 0.0], 2.)
    time.sleep(0.5)
    pose_middle = MyUR.get_joint()
    print('The pose difference:', pose_middle - pose_init)

    MyUR.move_speedj([0, 0, 0., 0, 0., -0.01], 2., 0.9)
    time.sleep(.5)

    pose_end = MyUR.get_joint()
    print('The pose difference:', pose_end - pose_middle)
    MyUR.rtde_c.speedStop(5.)
    MyUR.rtde_r.stopFileRecording()

    MyUR.exit_script()
    """

    freq = 500
    MyUR = URROb(freq)
    MyUR.record_variable = ['timestamp', 'target_TCP_pose', 'actual_TCP_pose', 'target_TCP_speed', 'actual_TCP_speed']

    MyUR.start_record_data()
    for i in range(50):
        pose_init = MyUR.get_pose()
        MyUR.move_add_movel([0, 0, 0., 0, 0., 0.])
        pose_middle = MyUR.get_pose()
        print('The pose difference:', pose_middle - pose_init)

        time.sleep(1.)

    # MyUR.move_add_movej([0, 0, 0., 0, 0., 0.01])
    # pose_end = MyUR.get_joint()
    # print('The pose difference:', pose_end - pose_middle)

    MyUR.rtde_c.stopL()
    MyUR.rtde_r.stopFileRecording()

    MyUR.exit_script()
