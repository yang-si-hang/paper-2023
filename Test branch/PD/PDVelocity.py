"""
This file to test **Velocity Constraint**
"""


from ast import main
import numpy as np


dt = 1./100
node_pos_init = np.array([[0, 0],
                          [0, 1]])

node_pos = node_pos_init.copy()
node_vel = np.zeros_like(node_pos)

mass = np.ones(node_pos.shape[0])

A_spring = np.array([[1, 0, -1, 0],
                     [0, 1, 0, -1]])
weight_s = 1.e

A_positional = np.array([[1., 0.],
                         [0., 1.]])
weight_p = 1.e5

A_velocity = np.array([[1., 0.],
                       [0., 1.]])
weight_v = 1.e5

lhs = np.zeros((4, 4))
lhs += np.diag(mass) / dt**2
lhs += weight_s * A_spring.T @ A_spring

# No. 1 is the fix node
lhs[0:2, 0:2] += weight_p * A_positional.T @ A_positional 

# No. 2 is the velocity constraint node
lhs[2:4, 2:4] += weight_v * A_velocity.T @ A_velocity

rhs = np.zeros(4)
sn = np.ones(4)

# construct sn
sn = node_pos.flatten() + dt * node_vel.flatten()

# warm start
node_pos_new = sn.copy()

# local solver
