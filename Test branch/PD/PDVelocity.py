"""
This file to test **Velocity Constraint** with 2 nodes.
"""


import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


# Inital setting
dt = 1./100
node_pos_init = np.array([[0, 0],
                          [0, 1]])

node_pos = node_pos_init.copy()
node_vel = np.zeros_like(node_pos)

mass = np.ones(node_pos.shape[0])
vel = np.array([0., 0.01])

k0 = np.linalg.norm(node_pos_init[0] - node_pos_init[1])

# Precomputations
A_spring = np.array([[1, 0, -1, 0],
                     [0, 1, 0, -1]])
weight_s = 1.e0

A_positional = np.array([[1., 0.],
                         [0., 1.]])
weight_p = 1.e5

A_velocity = np.array([[1., 0.],
                       [0., 1.]])
weight_v = 1.e10

lhs = np.zeros((4, 4))
lhs += np.diag(np.repeat(mass, 2)) / dt**2
lhs += weight_s * A_spring.T @ A_spring

# No. 1 is the fix node
lhs[0:2, 0:2] += weight_p * A_positional.T @ A_positional 

# No. 2 is the velocity constraint node
lhs[2:4, 2:4] += weight_v * A_velocity.T @ A_velocity


# global solver
s_lhs = sparse.csr_matrix(lhs)
pre_fact_lhs_solve = sparse.linalg.factorized(s_lhs)


def subsetp():
    global node_pos, node_vel
    # construct sn
    # sn = np.ones(4)
    sn = node_pos.flatten() + dt * node_vel.flatten()

    # warm start
    node_pos_new = sn.copy()

    for itr in range(10):
        rhs = np.zeros(4)
        # local solver & construct rhs
        rhs += sn * np.repeat(mass, 2) / dt ** 2  # element-wise multiplication

        # spring constraint
        direction = node_pos_new[0:2] - node_pos_new[2:4]
        k_dir = k0 * direction / np.linalg.norm(direction)
        rhs += weight_s * A_spring.T @ k_dir

        rhs[0:2] += weight_p * A_positional.T @ node_pos_init[0].T

        rhs[2:4] += weight_v * A_velocity.T @ (node_pos[1].T + dt * vel)

        node_pos_new = pre_fact_lhs_solve(rhs)

    # update node pos and vel
    node_pos_new = node_pos_new.reshape(-1, 2)
    node_vel = (node_pos_new - node_pos) / dt
    node_pos = node_pos_new.copy()


for i in range(100):
    subsetp()
    print(node_pos)