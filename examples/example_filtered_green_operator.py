import sys

sys.path.append('..')

import numpy as np
import skimage.io

skimage.io.use_plugin('freeimage', 'imread')
skimage.io.use_plugin('freeimage', 'imsave')

import janus.discretegreenop as discretegreenop
import janus.fft.serial as fft
import janus.greenop as greenop
import janus.material.elastic.linear.isotropic as material

mat = material.create(1.0, 0.3, 2)
n0 = 256
n1 = 256
n = (n0, n1)

greenc = greenop.create(mat)
greend = discretegreenop.FilteredGreenOperator2D(greenc, n, 1.)

g = np.empty((n0, n1), dtype=np.float64)
b = np.empty((2,), dtype=np.intc)

tau = np.array([0., 0., 1.])
eta = np.empty_like(tau)

for b0 in range(n0):
    b[0] = b0
    for b1 in range(n1):
        b[1] = b1
        greend.set_frequency(b)
        greend.apply_by_freq(tau, eta)
        g[b0, b1] = eta[0]

skimage.io.imsave('bla.tif', g.astype(np.float32))
