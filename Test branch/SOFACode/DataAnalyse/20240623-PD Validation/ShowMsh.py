"""
为加载的msh文件中的物体,得到最小包围盒
Taichi绘制Mesh网格的效果不好
"""

import numpy as np
import time
import taichi as ti
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sklearn.decomposition import PCA
ti.init(arch=ti.cpu)


def read_msh_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    nodes = []
    cells = []
    is_node_section = False
    is_cell_section = False

    for line in lines:
        if line.strip() == "$NOD":
            is_node_section = True
            continue
        if line.strip() == "$ENDNOD":
            is_node_section = False
            continue
        if line.strip() == "$ELM":
            is_cell_section = True
            continue
        if line.strip() == "$ENDELM":
            is_cell_section = False
            continue

        if is_node_section:
            parts = line.strip().split()
            if len(parts) == 4:
                index, x, y, z = parts
                nodes.append([float(x), float(y), float(z)])

        if is_cell_section:
            parts = line.strip().split()
            if len(parts) > 4:
                index = parts[0]
                cell_nodes = parts[5:]
                cells.append([int(node) for node in cell_nodes])

    nodes_array = np.array(nodes)
    cells_array = np.array(cells)

    return nodes_array, cells_array


def get_tetrahedron_edges(tet_indices):
    # 四面体的6条边
    edge_combinations = [(0, 1), (0, 2), (0, 3),
                         (1, 2), (1, 3),
                         (2, 3)]
    
    # 使用集合来存储唯一的边
    unique_edges = set()
    
    for tet in tet_indices:
        for i, j in edge_combinations:
            # 确保边的顶点索引是有序的
            edge = tuple(sorted([tet[i], tet[j]]))
            unique_edges.add(edge)
    
    # 将集合转换为numpy数组
    return np.array(list(unique_edges))


def get_OBB(node_np):
    # 主成分分析 (PCA)
    pca = PCA(n_components=3)
    pca.fit(node_np)
    transformed_nodes = pca.transform(node_np)

    # 计算投影数据的AABB
    min_coords = transformed_nodes.min(axis=0)
    max_coords = transformed_nodes.max(axis=0)

    # AABB的八个顶点
    vertices = np.array([[min_coords[0], min_coords[1], min_coords[2]],
                         [min_coords[0], min_coords[1], max_coords[2]],
                         [min_coords[0], max_coords[1], min_coords[2]],
                         [min_coords[0], max_coords[1], max_coords[2]],
                         [max_coords[0], min_coords[1], min_coords[2]],
                         [max_coords[0], min_coords[1], max_coords[2]],
                         [max_coords[0], max_coords[1], min_coords[2]],
                         [max_coords[0], max_coords[1], max_coords[2]],])

    # 将AABB顶点变换回原坐标系
    vertices = pca.inverse_transform(vertices)

    # OBB的12条边
    edges = [[vertices[0], vertices[1]], [vertices[0], vertices[2]], [vertices[0], vertices[4]],
             [vertices[1], vertices[3]], [vertices[1], vertices[5]],
             [vertices[2], vertices[3]], [vertices[2], vertices[6]],
             [vertices[3], vertices[7]],
             [vertices[4], vertices[5]], [vertices[4], vertices[6]],
             [vertices[5], vertices[7]],
             [vertices[6], vertices[7]]]

    return vertices, edges


node_np, element_np = read_msh_file("liver.msh")
node_np = node_np * 0.05
edge_np = get_tetrahedron_edges(element_np)

PARTICLE_NUM = np.shape(node_np)[0]
ELEMENT_NUM = np.shape(element_np)[0]
EDGE_NUM = np.shape(edge_np)[0]

fix_particle_list = [3, 39, 64]
contact_particles_list = [85]
exclude_set = set(fix_particle_list + contact_particles_list)
# surface_moveable_particles_list = [i for i in surfaces_node_np if i not in exclude_set]

# 获取OBB
vertices, edges = get_OBB(node_np)

# Define the data for GGUI
node_show = ti.Vector.field(3, dtype=ti.f32, shape=PARTICLE_NUM)
element_show = ti.field(ti.i32, shape=4*ELEMENT_NUM)
# surface_moveable_node_show = ti.Vector.field(3, dtype=ti.f32, shape=len(surface_moveable_particles_list))
# fix_node_show = ti.Vector.field(3, dtype=ti.f32, shape=len(fix_particle_list))
# contact_node_show = ti.Vector.field(3, dtype=ti.f32, shape=len(contact_particles_list))
# edge_show = ti.Vector.field(2, dtype=ti.i32, shape=EDGE_NUM)
# edge_show.from_numpy(edge_np)
# surfaces_edge_show = ti.Vector.field(2, dtype=ti.i32, shape=surfaces_edge_num)
# surfaces_edge_show.from_numpy(surfaces_edge_np)


def preset_gui(camera_pos:list, camera_target:list):
    """
    Define the camera position & target
    """
    # window, camera, scene = gui_set(pos=[0.1, 0.2, 0.], target=[0.1, 0., 0.])
    window, camera, scene = gui_set(pos=camera_pos, target=camera_target)
    canvas = window.get_canvas()
    return window, scene, canvas


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



def gui_show(window, canvas, scene, SHOW_FLAG=True, WRITE_FLAG=False, itr_num=0):
    """
    Show the GGUI
    """
    if SHOW_FLAG is False:
        return
    scene.point_light(pos=(0.01, 1, 3), color=(1., 1., 1.))
    scene.ambient_light((0.8, 0.8, 0.8))
    # the conversion of object particles, etc. the ggui of the taichi only support float32
    # surface_moveable_node_show.from_numpy(node_pos.to_numpy(dtype=np.float32)[surface_moveable_particles_list])
    # fix_node_show.from_numpy(node_pos.to_numpy(dtype=np.float32)[fix_particle_list])
    # contact_node_show.from_numpy(node_pos.to_numpy(dtype=np.float32)[contact_particles_list])
    node_show.from_numpy(node_np)
    element_show.from_numpy(element_np.flatten())

    # scene.particles(.node_show, radius=0.002, color=(0., 0., 0.))
    # scene.particles(surface_moveable_node_show, radius=0.002, color=(0., 0., 0.))
    # scene.particles(fix_node_show, radius=0.004, color=(1., 0., 0.))
    # scene.particles(contact_node_show, radius=0.0035, color=(0., 1., 0.))
    # scene.lines(node_show, width=1., indices=surfaces_edge_show, color=(0., 0., 0.),
    #             vertex_count=0)
    scene.mesh(node_show, element_show, color=(0., 0., 0.), two_sided=False, show_wireframe=True)
    canvas.scene(scene)
    canvas.set_background_color((1.0, 1.0, 1.0))
    # if WRITE_FLAG is True and itr_num % 10 == 0:
    if WRITE_FLAG is True:
        window.save_image(f'FigureWrite/{itr_num}.png')
    window.show()


if __name__ == "__main__":


    # 计算最小包围盒的长宽高
    length = np.linalg.norm(vertices[0] - vertices[4])
    width = np.linalg.norm(vertices[0] - vertices[2])
    height = np.linalg.norm(vertices[0] - vertices[1])

    print(f'Length: {length}. Width: {width}. Height: {height}.')

    window, scene, canvas = preset_gui([-0.1, 0.5, 0.3], [-0.05, 0., -0.1])
    for i in range(100):
        time.sleep(0.1)
        gui_show(window, canvas, scene)
    
    # # 绘制网格物体和OBB
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    # # 绘制节点
    # ax.scatter(node_np[:, 0], node_np[:, 1], node_np[:, 2], s=55, c='r', marker='.')

    # # 绘制OBB
    # for edge in edges:
    #     ax.plot3D(*zip(*edge), color='k')

    # # 设置标签和图例
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # plt.show()