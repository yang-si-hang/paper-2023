"""
Give action to the robot
"""

import numpy as np
import rtde_control
import rtde_receive
import copy


class URROb:
    def __init__(self, control_frequecy):
        self.UR_IP = '192.168.253.10'
        self.rtde_frequecy = control_frequecy

        self.rtde_c = rtde_control.RTDEControlInterface(self.UR_IP, self.rtde_frequecy)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.UR_IP, self.rtde_frequecy)


    def move_speedj(self, v, a=0.5, dt=1./500):
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.speedL(v, a, dt)
        self.rtde_c.waitPeriod(t_start)
        return
    

    def move_speedl(self, v, a=0.5, dt=1./500):
        t_start = self.rtde_c.initPeriod()
        self.rtde_c.speedL(v, a, dt)
        self.rtde_c.waitPeriod(t_start)
        return


    def move_add_servoj(self, joint_add):
        dt = 1./self.rtde_frequecy
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
        dt = 1./self.rtde_frequecy
        pose_start = self.rtde_r.getTargetTCPPose()
        t_start = self.rtde_c.initPeriod()
        pose_end = copy.deepcopy(pose_start)
        pose_end = [pose_end[i] + pose_add[i] for i in range(6)]
        self.rtde_c.servoL(pose_end, 0.5, 0.5, 0.1, 0.05, 2000)
        self.rtde_c.waitPeriod(t_start)


    def move_add_movel(self, pose_add):
        pose_start = self.rtde_r.getTargetTCPPose()
        pose_end = copy.deepcopy(pose_start)
        pose_end = [pose_end[i] + pose_add[i] for i in range(6)]
        # True是非阻塞，False是阻塞
        self.rtde_c.moveL(pose_end, 0.5, 0.5, False)


    def get_joint(self):
        actual_q = self.rtde_r.getActualQ()
        print('The joint position is:', actual_q)


    def get_pose(self):
        actual_pose = self.rtde_r.getTargetTCPPose()
        print('The TCP Pose is:', actual_pose)
        return np.array(actual_pose)


    def exit_script(self):
        self.rtde_c.stopScript()


freq = 500
MyUR = URROb(freq)
pose_init = MyUR.get_pose()