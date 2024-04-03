"""
UR10 control test
"""
import numpy as np
import rtde_control
import rtde_receive
import rtde_io
import time
import copy

UR_IP = '192.168.253.10'
rtde_frequency = 500.0

# read positions
"""
rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
actual_q = rtde_r.getActualQ()
print('actual_q: ', actual_q)
"""

# Move Asynchronous
"""
rtde_c = rtde_control.RTDEControlInterface(UR_IP)
rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP)
init_q = rtde_r.getActualQ()

# Target in the robot base
new_q = init_q[:]
new_q[4] += 0.50

# Move asynchronously in joint space to new_q, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopJ function due to the blocking behaviour.
rtde_c.moveJ(new_q, 1.05, 1.4, True)
time.sleep(0.4)
# Stop the movement before it reaches new_q
rtde_c.stopJ(0.5)

# Target in the Z-Axis of the TCP
target = rtde_r.getActualTCPPose()
target[2] += 0.1

# Move asynchronously in cartesian space to target, we specify asynchronous behavior by setting the async parameter to
# 'True'. Try to set the async parameter to 'False' to observe a default synchronous movement, which cannot be stopped
# by the stopL function due to the blocking behaviour.
rtde_c.moveL(target, 0.25, 0.5, True)
time.sleep(0.2)
# Stop the movement before it reaches target
rtde_c.stopL(0.5)

# Move back to initial joint configuration
rtde_c.moveJ(init_q)

# Stop the RTDE control script
rtde_c.stopScript()
"""

# Force control
"""
rtde_c = rtde_control.RTDEControlInterface(UR_IP, rtde_frequency)

task_frame = [0, 0, 0, 0, 0, 0]
selection_vector = [0, 0, 1, 0, 0, 0]
wrench_down = [0, 0, -10, 0, 0, 0]
wrench_up = [0, 0, 10, 0, 0, 0]
force_type = 2
limits = [2, 2, 1.5, 1, 1, 1]
dt = 1.0/500  # 2ms

time0 = time.time()
# Execute 500Hz control loop for 4 seconds, each cycle is 2ms
for i in range(2000):
    t_start = rtde_c.initPeriod()
    # First move the robot down for 2 seconds, then up for 2 seconds
    if i > 1000:
        rtde_c.forceMode(task_frame, selection_vector, wrench_up, force_type, limits)
    else:
        rtde_c.forceMode(task_frame, selection_vector, wrench_down, force_type, limits)
    rtde_c.waitPeriod(t_start)

print(time.time()-time0)

rtde_c.forceModeStop()
rtde_c.stopScript()
"""

# servoj control
"""
rtde_c = rtde_control.RTDEControlInterface(UR_IP, rtde_frequency)
rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP, rtde_frequency)

# Parameters
velocity = 0.5
acceleration = 0.5
dt = 1.0/500  # 2ms
lookahead_time = 0.1
gain = 300
joint_q_init = rtde_r.getActualQ()
joint_q = copy.deepcopy(joint_q_init)

# Execute 500Hz control loop for 2 seconds, each cycle is 2ms
for i in range(1000):
    t_start = rtde_c.initPeriod()
    rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)
    joint_q[4] += 0.001
    joint_q[5] += 0.001
    rtde_c.waitPeriod(t_start)

rtde_c.servoStop()          # need stop servo mode

rtde_c.moveJ(joint_q_init)
rtde_c.stopScript()
"""

# speedj
"""
rtde_c = rtde_control.RTDEControlInterface(UR_IP, rtde_frequency)
rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP, rtde_frequency)

# Parameters
acceleration = 0.5
dt = 1.0/500  # 2ms
joint_speed = np.zeros((6))

joint_q_init = rtde_r.getActualQ()

# Execute 500Hz control loop for 2 seconds, each cycle is 2ms
for i in range(1000):
    t_start = rtde_c.initPeriod()
    rtde_c.speedJ(joint_speed, acceleration, dt)
    joint_speed[4] = 0.5 + 0.1*np.random.random()
    joint_speed[5] = 0.5
    print(joint_speed)
    rtde_c.waitPeriod(t_start)

rtde_c.speedStop()

rtde_c.moveJ(joint_q_init)
rtde_c.stopScript()
"""

# moveL 是直线运动，blend是平滑
"""
rtde_c = rtde_control.RTDEControlInterface(UR_IP, rtde_frequency)
rtde_r = rtde_receive.RTDEReceiveInterface(UR_IP, rtde_frequency)

joint_q_init = np.array(rtde_r.getActualTCPPose())
# getActualTCPPose
joint_q_1 = joint_q_init + np.array([0, 0, 0, 0, 0.1, 0.08])
joint_q_2 = joint_q_1 + np.array([0, 0, 0, 0, 0.15, 0.1])

velocity = 0.5
acceleration = 0.5
blend_1 = 0.0
blend_2 = 0.02
blend_3 = 0.0
path_pose1 = list(joint_q_init) + [velocity, acceleration, blend_1]
path_pose2 = list(joint_q_1) + [velocity, acceleration, blend_2]
path_pose3 = list(joint_q_2) + [velocity, acceleration, blend_3]
path = [path_pose1, path_pose2, path_pose3]

print(path)
# exit(0)

# Send a linear path with blending in between - (currently uses separate script)
rtde_c.moveL(path)
rtde_c.stopScript()
"""


rtde_io_ = rtde_io.RTDEIOInterface(UR_IP, rtde_frequency)
rtde_receive_ = rtde_receive.RTDEReceiveInterface(UR_IP, rtde_frequency)

# How-to set and get standard and tool digital outputs. Notice that we need the
# RTDEIOInterface for setting an output and RTDEReceiveInterface for getting the state
# of an output.

if rtde_receive_.getDigitalOutState(7):
    print("Standard digital out (7) is HIGH")
else:
    print("Standard digital out (7) is LOW")

if rtde_receive_.getDigitalOutState(16):
    print("Tool digital out (16) is HIGH")
else:
    print("Tool digital out (16) is LOW")

rtde_io_.setStandardDigitalOut(7, True)
rtde_io_.setToolDigitalOut(0, True)
time.sleep(0.01)

if rtde_receive_.getDigitalOutState(7):
    print("Standard digital out (7) is HIGH")
else:
    print("Standard digital out (7) is LOW")

if rtde_receive_.getDigitalOutState(16):
    print("Tool digital out (16) is HIGH")
else:
    print("Tool digital out (16) is LOW")

# How to set a analog output with a specified current ratio
rtde_io_.setAnalogOutputCurrent(1, 0.25)