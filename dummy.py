import sys
sys.path.append('./src')

import numpy as np

import discretegreenop
import fft.serial
import greenop

from matprop import IsotropicLinearElasticMaterial as Material

mat = Material(0.75, 0.3, 2)
n = (200, 300)
green = discretegreenop.create(greenop.create(mat), n, 1.)
tau = np.zeros((3,))
eta = np.zeros((3,))
b = np.zeros((2,), dtype=np.intp)

#green.apply(b, tau, eta)
green.as_array(b)
