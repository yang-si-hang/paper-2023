"""
用于Taichi的GUI设置
create at 2024-07-28 by hsy
"""

import taichi as ti

def gui_set(pos, target, FOV=60):
    # init the window, canvas, scene and camerea
    window = ti.ui.Window("Projective Dynamics", (1080, 720), vsync=True)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initialize camera position
    camera.position(pos[0], pos[1], pos[2])
    camera.lookat(target[0], target[1], target[2])
    camera.projection_mode(ti.ui.ProjectionMode.Perspective)
    # 设置相机的向上轴的方向，在相机模型中是-Y轴
    camera.up(0., 0., -1.)
    camera.z_near(0.01)
    camera.fov(FOV)

    # set the camera, you can move around by pressing 'wasdeq'
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # set the light
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    scene.ambient_light((1., 1., 1.))
    return window, camera, scene