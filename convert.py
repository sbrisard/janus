import numpy as np

n0 = 200
n1 = 300
n2 = 6

depth = 8

basename = './truncated_green_operator_200x300_unit_tau_xy_10x10+95+145'

a = np.fromfile(basename + '.dat', dtype = '>f{0}'.format(depth))
a.shape = (n0, n1, n2)
a.strides = (depth * n1, depth, depth * n0 * n1)

np.save(basename + '.npy', np.asarray(a, dtype = '<f{0}'.format(depth)))
