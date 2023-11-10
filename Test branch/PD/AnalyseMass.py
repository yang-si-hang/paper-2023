"""
This file for analyse the influence of the mass matrix of a triangle element to forward
simulation and DiffPD method.
"""


import numpy as np
np.set_printoptions(linewidth=300)


lhs = np.loadtxt('lhs.csv', delimiter=',')
dA = np.loadtxt('dA.csv', delimiter=',')

Aq = np.loadtxt('Aq.csv', delimiter=',')

M = lhs - Aq

print('M = \n', M)
print('Aq = \n', Aq)
print('dA = \n', dA)

print('dx/dy=\n', np.linalg.solve(M+Aq+dA, M))
np.savetxt('dx_dy.csv', np.linalg.solve(M+Aq+dA, M), delimiter=',')