"""
Test gradient of triangle area loss by using DiffTaichi
"""


import taichi as ti
ti.init(arch=ti.cpu, debug=True)

q = ti.Vector.field(2, dtype=ti.f32, shape=3, needs_grad=True)
L = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


@ti.kernel
def compute_L(desired_area:ti.f32):
    qba = q[1] - q[0]
    qca = q[2] - q[0]
    area_current = ti.abs(qba.cross(qca))/2.
    L[None] = (area_current - desired_area)**2


def substep():
    with ti.ad.Tape(loss=L):
        compute_L(0.01)
    # print(L.grad[0])


@ti.kernel
def init():
    for i in q:
        q[i] = ti.Vector([ti.random(), ti.random()])

    # q[0] = ti.Vector([0., 0.])
    # q[1] = ti.Vector([0.1, 0.])
    # q[2] = ti.Vector([0., 0.1])


@ti.kernel
def update(learning_rate: ti.f32):
    for i in q:
        q[i] -= learning_rate*q.grad[i]


init()
gui = ti.GUI('AD')
while gui.running:
    for i in range(5):
        substep()
        # print(L[None])
    gui.circles(q.to_numpy(), radius=5)
    gui.show()
    update(0.05)
    print('Loss:', L[None])
    print('q grad:', q.grad.to_numpy())
    print('q:', q.to_numpy())
    print('Gravity pos: ', q.to_numpy().sum(axis=0))