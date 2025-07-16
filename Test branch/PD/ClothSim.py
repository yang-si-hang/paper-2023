"""
用于面试ppt的布料仿真demo
created by hsy at 2025-06-23
"""
import os, sys
import numpy as np
import taichi as ti
ti.init(arch=ti.gpu, debug=True, default_fp=ti.f64)

script_dir = os.path.dirname(os.path.abspath(__file__))     # 设置工作目录为当前脚本所在目录
os.chdir(script_dir)

# 添加根目录到 sys.path（跨目录导入模块）
root_path = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(root_path)


@ti.data_oriented
class Cloth:
    def __init__(self, n=128, quad_size=1.0, dt=4e-2, substeps=15):
        self.n = n                      # 边的节点离散数量
        self.quad_size = quad_size
        self.dt = dt                # 时间步长
        self.substeps = substeps

        self.gravity = ti.Vector([0, 0, -9.8])
        self.spring_Y = 3e4             # 弹簧系数
        self.dashpot_damping = 1e4      # 阻尼系数
        self.drag_damping = 1.0         # 拖动阻尼系数
        
        self.ball_radius = 0.3
        self.ball_center = ti.Vector.field(3, dtype=ti.f64, shape=(1, ))
        self.ball_center[0] = [0, 0, 0]

        self.floor_z = 0.05
        self.floor_show = ti.Vector.field(3, dtype=ti.f64, shape=4)
        self.floor_show[0] = [-1, -1, self.floor_z - 0.02]
        self.floor_show[1] = [1, -1, self.floor_z - 0.02]
        self.floor_show[2] = [1, 1, self.floor_z - 0.02]
        self.floor_show[3] = [-1, 1, self.floor_z - 0.02]
        self.floor_indices = ti.field(ti.i32, shape=6)  # 地板的索引
        self.floor_indices.from_numpy(np.array([0, 1, 2, 0, 2, 3], dtype=np.int32))

        self.node_pos = ti.Vector.field(3, dtype=ti.f64, shape=(n, n))
        self.node_vel = ti.Vector.field(3, dtype=ti.f64, shape=(n, n))

        self.num_triangles = (n - 1) * (n - 1) * 2
        self.indices = ti.field(ti.i32, shape=self.num_triangles * 3)      # 三角形网格的节点索引
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=n * n)
        self.colors = ti.Vector.field(3, dtype=ti.f32, shape=n * n)

        self.spring_offsets = self.define_spring_offsets()
        self.initialize_mass_points()
        self.initialize_mesh_indices()

        self.fixed = ti.field(bool, shape=1)
        self.fixed[0] = False
        self.contact = [[n-1, 0], [n-1, n-1]]
        self.contact_vel = ti.Vector.field(3, dtype=ti.f64, shape=1)  # 接触点的速度
        self.contact_vel.fill(0.)
        self.contact_show = ti.Vector.field(3, dtype=ti.f32, shape=2)


    @ti.kernel
    def initialize_mass_points(self):
        random_offset = ti.Vector([ti.random() - 0.5, ti.random() - 0.5]) * 0.1 - ti.Vector([0.5, 0.5])
        for i, j in self.node_pos:
            self.node_pos[i, j] = [i * self.quad_size + random_offset[0], j * self.quad_size + random_offset[1], 0.6]

            self.node_vel[i, j] = [0, 0, 0]


    @ti.kernel
    def initialize_mesh_indices(self):
        for i, j in ti.ndrange(self.n - 1, self.n - 1):
            quad_id = (i * (self.n - 1)) + j
            # 1st triangle of the square
            self.indices[quad_id * 6 + 0] = i * self.n + j
            self.indices[quad_id * 6 + 1] = (i + 1) * self.n + j
            self.indices[quad_id * 6 + 2] = i * self.n + (j + 1)
            # 2nd triangle of the square
            self.indices[quad_id * 6 + 3] = (i + 1) * self.n + j + 1
            self.indices[quad_id * 6 + 4] = i * self.n + (j + 1)
            self.indices[quad_id * 6 + 5] = (i + 1) * self.n + j

        # 设定颜色
        for i, j in ti.ndrange(self.n, self.n):
            if (i // 4 + j // 4) % 2 == 0:
                self.colors[i * self.n + j] = (0.22, 0.72, 0.52)
            else:
                self.colors[i * self.n + j] = (1, 0.334, 0.52)

    
    def define_spring_offsets(self):
        spring_offsets = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if (i, j) != (0, 0):
                    spring_offsets.append(ti.Vector([i, j]))
        return spring_offsets
    

    @ti.kernel
    def substep(self):
        for i in ti.grouped(self.node_vel):
            self.node_vel[i] += self.gravity * self.dt

        if self.fixed[0]:
            for i in ti.static(self.contact):
                self.node_vel[i] = self.contact_vel[0]

        for i in ti.grouped(self.node_pos):
            force = ti.Vector([0.0, 0.0, 0.0])
            for spring_offset in ti.static(self.spring_offsets):
                j = i + spring_offset           # 节点i通过弹簧连接的节点j
                if 0 <= j[0] < self.n and 0 <= j[1] < self.n:
                    x_ij = self.node_pos[i] - self.node_pos[j]
                    v_ij = self.node_vel[i] - self.node_vel[j]
                    d = x_ij.normalized()
                    current_dist = x_ij.norm()
                    original_dist = self.quad_size * float(i - j).norm()
                    # Spring force
                    force += -self.spring_Y * d * (current_dist / original_dist - 1)
                    # Dashpot damping
                    force += -v_ij.dot(d) * d * self.dashpot_damping * self.quad_size

            self.node_vel[i] += force * self.dt

        for i in ti.grouped(self.node_pos):
            self.node_vel[i] *= ti.exp(-self.drag_damping * self.dt)

        if self.fixed[0]:
            for i in ti.static(self.contact):
                self.node_vel[i] = self.contact_vel[0]

        for i in ti.grouped(self.node_pos):
            offset_to_center = self.node_pos[i] - self.ball_center[0]
            offset_to_floor = self.node_pos[i][2] - self.floor_z
            if offset_to_center.norm() <= self.ball_radius:
                # Velocity projection
                normal = offset_to_center.normalized()
                self.node_vel[i] -= min(self.node_vel[i].dot(normal), 0) * normal
                self.node_vel[i] *= ti.exp(-10*self.drag_damping * self.dt)
            elif offset_to_floor < 0:
                # Floor collision response
                normal = ti.Vector([0, 0, 1])
                self.node_vel[i] -= min(self.node_vel[i].dot(normal), 0) * normal
                self.node_vel[i] *= ti.exp(-10*self.drag_damping * self.dt)

            self.node_pos[i] += self.dt * self.node_vel[i]


    @ti.kernel
    def update_vertices_show(self):
        for i, j in ti.ndrange(self.n, self.n):
            self.vertices[i * self.n + j] = self.node_pos[i, j]


    def initialize_visual(self):
        self.window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color((1, 1, 1))
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.make_camera()


    def run_simulation(self, step_num=0):
        for i in range(self.substeps):
            self.substep()

        itr_num = step_num

        self.update_vertices_show()

        self.camera.position(0.0, -1.0, 2.0)
        self.camera.lookat(0.0, 0.0, 0)
        self.camera.up(0.0, 0.0, 1.0)
        self.camera.fov(60)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        
        self.scene.set_camera(self.camera)
        self.scene.point_light(pos=(0, 1, 2), color=(1, 1, 1))
        self.scene.ambient_light((0.5, 0.5, 0.5))
        self.scene.mesh(self.vertices,
                        indices=self.indices,
                        per_vertex_color=self.colors,
                        two_sided=True)

        # Draw a smaller ball to avoid visual penetration
        self.scene.particles(self.ball_center, radius=self.ball_radius * 0.95, color=(0.5, 0.42, 0.8))
        self.scene.mesh(self.floor_show, indices=self.floor_indices, color=(0.5, 0.42, 0.8), two_sided=True)
        self.canvas.scene(self.scene)

        # self.window.save_image(f"Figure/{itr_num:05d}.png")
        self.window.show()


if __name__ == "__main__":
    n = 300
    quad_size = 1.0 / n
    dt = 4e-2 / n
    substeps = int(1 / 60 // dt)
    cloth = Cloth(n, quad_size, dt, substeps)
    cloth.initialize_visual()

    while cloth.window.running:
        cloth.run_simulation()

    step:int = 0

    cloth.fixed[0] = False
    for i in range(100):
        cloth.run_simulation(step)
        step += 1


    cloth.fixed[0] = True
    cloth.contact_vel[0] = [0, 0, 0.5]  # 设置接触点的速度
    while cloth.window.running:
        cloth.run_simulation(step)
        step += 1
        cloth.contact_show[0] = cloth.node_pos[cloth.contact[0][0], cloth.contact[0][1]]
        cloth.contact_show[1] = cloth.node_pos[cloth.contact[1][0], cloth.contact[1][1]]
        cloth.scene.particles(cloth.contact_show, radius=0.02, color=(0.9, 0, 0))
        tmp = cloth.node_pos[cloth.n-1, 0].z
        if tmp > 1.1:
            break

    cloth.contact_vel[0] = [-0.5, 0, -0.3]
    while cloth.window.running:
        cloth.run_simulation(step)
        step += 1
        cloth.contact_show[0] = cloth.node_pos[cloth.contact[0][0], cloth.contact[0][1]]
        cloth.contact_show[1] = cloth.node_pos[cloth.contact[1][0], cloth.contact[1][1]]
        cloth.scene.particles(cloth.contact_show, radius=0.02, color=(0.9, 0, 0))
        tmp = cloth.node_pos[cloth.n-1, 0].x
        if tmp < -0.2:
            break

    cloth.contact_vel[0] = [-0.1, 0, -1.0]
    while cloth.window.running:
        cloth.run_simulation(step)
        step += 1
        cloth.contact_show[0] = cloth.node_pos[cloth.contact[0][0], cloth.contact[0][1]]
        cloth.contact_show[1] = cloth.node_pos[cloth.contact[1][0], cloth.contact[1][1]]
        cloth.scene.particles(cloth.contact_show, radius=0.02, color=(0.9, 0, 0))
        tmp = cloth.node_pos[cloth.n-1, 0].z
        if tmp < 0.2:
            break

    cloth.fixed[0] = False
    for i in range(50):
        cloth.run_simulation(step)
        step += 1

    ti.sync()  # 确保所有计算完成
    print("Simulation finished.")