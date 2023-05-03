"""
This file simulate the object deformation with Mass Spring Method
"""

import taichi as ti
ti.init(arch=ti.cpu, debug=True, default_fp=ti.f64)
import numpy as np

from MeshLoad import *


"""-----------Initialization-----------"""



@ti.kernel
def compute_force():
    for i in range(PARTICLE_NUM):
        force[i] = ti.Vector([0., 0., 0.])

    # No gravity in this case!

    for i in range(EDGE_NUM):
        idx1, idx2 = edge[i][0], edge[i][1]
        



def substep():
    compute_force()
    compute_Jacobian_x()
    compute_Jacobian_v()





def gui_set(pos, target, FOV=60):
    # init the window, canvas, scene and camerea
    window = ti.ui.Window("MSS", (1080, 720), vsync=True)
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()

    # initialize camera position
    camera.position(pos[0], pos[1], pos[2])
    camera.lookat(target[0], target[1], target[2])
    camera.fov(FOV)

    # set the camera, you can move around by pressing 'wasdeq'
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # set the light
    scene.point_light(pos=(0, 1, 3), color=(1., 1., 1.))
    scene.point_light(pos=(0, 0, 3), color=(1., 1., 1.))
    scene.ambient_light((0.7, 0.7, 0.7))
    return window, camera, scene


def gui_show(window, canvas, scene, SHOW_FLAG=True):
    """
    Show the GUI
    """
    if SHOW_FLAG is False:
        return
    # the conversion of object particles, etc. the ggui of the taichi only support float32
    particle_show.from_numpy(particle.to_numpy(dtype=np.float32))

    scene.mesh(particle_show, indices=surf_show, color=(1, 1, 0))
    scene.particles(particle_show, radius=0.002, color=(0, 1, 1))
    canvas.scene(scene)
    # window.save_image(f'png/{n}.png')
    window.show()



def main():
    window, camera, scene = gui_set([0., 0.4, 0.], [0.1, 0., 0.])
    canvas = window.get_canvas()

    while window.running:
        # show the GUI
        gui_show(window, canvas, scene)



        # # update the GUI
        # window.GUI.begin("MSS")
        # window.GUI.slider_float("FOV", 0, 180, 60)
        # window.GUI.end()
        #
        # # update the window
        # window.show()


if __name__ == '__main__':
    main()