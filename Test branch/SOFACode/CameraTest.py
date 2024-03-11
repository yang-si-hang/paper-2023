"""
Test recorded camera
"""
import time

import Sofa
import Sofa.Gui
import Sofa.SofaGL
import Sofa.Simulation
import numpy as np
import os
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image

display_size = (800, 600)
script_dir = os.path.dirname(os.path.abspath(__file__))         # 获取脚本文件所在的绝对路径


def init_display(rootNode):
    pygame.display.init()
    pygame.display.set_mode(display_size, pygame.DOUBLEBUF | pygame.OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    Sofa.SofaGL.glewInit()
    Sofa.Simulation.initVisual(rootNode)
    Sofa.Simulation.initTextures(rootNode)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 50.0)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def simple_render(rootNode):
    # Reset in each simulation loop
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display_size[0] / display_size[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    cameraMVM = rootNode.camera.getOpenGLModelViewMatrix()
    glMultMatrixd(cameraMVM)
    Sofa.SofaGL.draw(rootNode)

    pygame.display.get_surface().fill((0,0,0))
    pygame.display.flip()

    # _, _, width, height = glGetIntegerv(GL_VIEWPORT)
    # buff = glReadPixels(0, 0, width, height, GL_DEPTH_COMPONENT, GL_FLOAT)
    # image = np.frombuffer(buff, dtype=np.float32)
    # image = image.reshape(height, width)
    # image = np.flipud(image)  #<-- image is now a numpy array you can use
    #
    #
    # z_far = rootNode.camera.findData('zFar').value
    # z_near = rootNode.camera.findData('zNear').value
    # depth_image = -z_far*z_near/(z_far + image*(z_near-z_far))
    # depth_image = (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    # depth_image = depth_image * 255
    # print(depth_image.min())
    # print(depth_image.max())
    # print(depth_image.size)
    #
    # img2 = Image.fromarray(depth_image.astype(np.uint8), 'L')
    # img2.save(f"{script_dir}/test.bmp")


def createScene(rootNode):
    rootNode.addObject('RequiredPlugin', pluginName=['Sofa.Component',
                                                     'Sofa.Component.Visual',
                                                     'Sofa.Component.IO.Mesh',
                                                     'Sofa.GL.Component.Rendering3D',
                                                     'Sofa.GL.Component.Shader'])

    rootNode.dt = 0.01
    rootNode.bbox = [[-0.1, -0.1, 0.], [0.2, 0.2, 0.1]]
    rootNode.gravity = [0., 0., 0.]
    rootNode.addObject('VisualStyle', displayFlags='showAll')
    # cam1 = rootNode.addObject('InteractiveCamera', name='cam', position=[0., 0.3, 0.], orientation=[1., 0., 0., 0.], zoomSpeed=0.1)

    rootNode.addObject('FreeMotionAnimationLoop')
    rootNode.addObject('GenericConstraintSolver', tolerance=1e-6, maxIterations=100)

    obj_visual = rootNode.addChild('VisualObject')
    obj_visual.addObject('MeshObjLoader', name='loader', filename='mesh/liver.obj', scale='1', flipNormals='0')
    # obj_visual.addObject('MeshVTKLoader', name='loader', filename='mesh/liver.vtk', scale='1', flipNormals='0')
    obj_visual.addObject('OglModel', name='visual', src='@loader', color=[1, 0, 0])

    rootNode.addObject("LightManager")
    rootNode.addObject("DirectionalLight", direction=[0,1,0])
    # add camera
    rootNode.addObject("InteractiveCamera", name="camera", position=[0, 15, 0], lookAt=[0,0,0], distance=37,
                       fieldOfView=45, zNear=0.01, zFar=55.69)


def main():
    root = Sofa.Core.Node("root")
    createScene(root)
    Sofa.Simulation.init(root)
    init_display(root)
    try:
        while True:
            Sofa.Simulation.animate(root, root.dt.value)
            Sofa.Simulation.updateVisual(root)
            simple_render(root)
            time.sleep(root.dt.value)
    except KeyboardInterrupt:
        pass


    # Sofa.Simulation.init(root)
    # Sofa.Gui.GUIManager.Init("myscene", "qt")
    # Sofa.Gui.GUIManager.createGUI(root, __file__)
    # Sofa.Gui.GUIManager.SetDimension(1080, 900)
    #
    # Sofa.Gui.GUIManager.MainLoop(root)
    # Sofa.Gui.GUIManager.closeGUI()



if __name__ == '__main__':
    main()