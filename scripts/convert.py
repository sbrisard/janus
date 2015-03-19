import numpy as np

"""
n0 = 200
n1 = 300
n2 = 6

depth = 8

basename = './truncated_green_operator_200x300_unit_tau_xy_10x10+95+145'

a = np.fromfile(basename + '.dat', dtype = '>f{0}'.format(depth))
a.shape = (n0, n1, n2)
a.strides = (depth * n1, depth, depth * n0 * n1)

np.save(basename + '.npy', np.asarray(a, dtype = '<f{0}'.format(depth)))
"""

n0 = 40
n1 = 50
n2 = 60
n3 = 12

depth = 8

a = np.fromfile('/media/sf_brisard/Documents/tmp/ref3D.dat', dtype = '>f{0}'.format(depth))
a.shape = (n0, n1, n2, n3)
a.strides = (depth, depth * n0, depth * n0 * n1, depth * n0 * n1 * n2)

np.save('truncated_green_operator_40x50x60_unit_tau_xy_10x10x10+15+20+25.npy', np.asarray(a, dtype = '<f{0}'.format(depth)))
