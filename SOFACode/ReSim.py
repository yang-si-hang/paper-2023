import time

import taichi as ti
import numpy as np
import os
ti.init(arch=ti.gpu)  # 使用GPU进行加速，如果你没有GPU可以改为ti.cpu


script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径
folder_path = script_dir + '/data'  # 数据文件夹路径

files_in_directory = os.listdir(folder_path)
csv_files = [file for file in files_in_directory if file.endswith('.csv')]
num_steps = len(csv_files)

edge_np = np.loadtxt(f'{script_dir}/edge.csv', delimiter=',')

# 参数设定
num_particles = 121  # 粒子数量
num_edges = np.size(edge_np, 0)  # 边的数量

# 创建一个2D矢量场来存储粒子位置
particles = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
edge = ti.Vector.field(2, dtype=ti.i32, shape=num_edges)
marker = ti.Vector.field(3, dtype=ti.f32, shape=1)
marker_desired = ti.Vector.field(3, dtype=ti.f32, shape=1)
marker_desired[0] = ti.Vector([0.092, -0.02, 0.])
edge.from_numpy(edge_np)

# 创建GGUI窗口
window = ti.ui.Window("Particle Simulation", res=(1080, 720), vsync=True)
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()

# initialize camera position
camera.position(0.1, 0., 0.2)
camera.lookat(0.1, 0., 0.)
camera.projection_mode(ti.ui.ProjectionMode.Perspective)
# 设置相机的向上轴的方向，在相机模型中是-Y轴
camera.up(0., 1., 0.)
camera.z_near(0.01)
camera.fov(60)

# set the camera, you can move around by pressing 'wasdeq'
camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
scene.set_camera(camera)

# set the light
scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
# scene.point_light(pos=(0.01, 0, 3), color=(1., 1., 1.))
scene.ambient_light((1., 1., 1.))


def update_particles(i):
    particles_np = np.loadtxt(f'{folder_path}/pos_{i+1}.csv', delimiter=' ')
    particles.from_numpy(particles_np)


def main():
    for step in range(num_steps):
        update_particles(step)
        if window.running:
            time.sleep(1./60)
            marker[0] = particles[42]

            canvas.set_background_color((1, 1, 1))
            scene.lines(particles, width=1., indices=edge, color=(0., 0., 0.),
                        vertex_count=0)
            scene.particles(particles, radius=0.001, color=(0., 0., 0.))
            scene.particles(marker, radius=0.0012, color=(1., 0., 0.))
            scene.particles(marker_desired, radius=0.0012, color=(0., 0., 1.))
            canvas.scene(scene)
            window.show()


if __name__ == "__main__":
    main()
