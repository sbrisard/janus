import numpy as np
from numpy import cos
from numpy import sin

from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import nottest

import mandelvoigt
from matprop import IsotropicLinearElasticMaterial as Material
from greenop import GreenOperator2d
from greenop import GreenOperator3d

def delta(i, j):
    if i == j:
        return 1
    else:
        return 0

def green_coefficient(i, j, k, l, n, mat):
    return ((delta(i, k) * n[j] * n[l]
             + delta(i, l) * n[j] * n[k]
             + delta(j, k) * n[i] * n[l]
             + delta(j, l) * n[i] * n[k]) / (4. * mat.g)
            - n[i] * n[j] * n[k] * n[l] / (2. * mat.g * (1. - mat.nu)))

def green_matrix(n, mat):
    return mandelvoigt.get_instance(mat.dim).create_array(green_coefficient, n, mat)

@nottest
def do_test_apply(k, mat):
    n = k / np.sqrt(sum(k**2))
    expected = green_matrix(n,  mat)
    sym = (mat.dim * (mat.dim + 1)) // 2
    if mat.dim == 2:
        green = GreenOperator2d(mat)
    elif mat.dim == 3:
        green = GreenOperator3d(mat)
    else:
        raise ValueError()
    actual = np.empty((sym,), np.float64)

    for j in range(sym):
        tau = np.zeros((sym,), np.float64)
        tau[j] = 1.
        green.apply(k, tau, actual)
        for i in range(sym):
            msg = 'coefficient [{0}, {1}]'.format(i, j)
            assert_almost_equal(expected[i, j], actual[i],
                                msg = msg, delta = 1.E-15)

def test_apply_2d():
    mat = Material(0.7, 0.3, 2)
    k_norms = [2.5, 3.5]
    num_angles = 20
    k_angles = [(2. * np.pi * i) / num_angles for i in range(num_angles)]

    vec = lambda r, theta: np.array([r * cos(theta), r * sin(theta)],
                                    dtype=np.float64)
    
    for k in k_norms:
        for theta in k_angles:
            yield do_test_apply, vec(k, theta), mat

def test_apply_3d():
    mat = Material(0.7, 0.3, 3)
    norms = [2.5, 3.5]
    num_thetas = 10
    num_phis = 20
    thetas = [(np.pi * i) / num_thetas for i in range(num_thetas)]
    phis = [(2. * np.pi * i) / num_phis for i in range(num_phis)]

    vec = lambda r, theta, phi: np.array([r * cos(theta) * cos(phi),
                                          r * cos(theta) * sin(phi),
                                          r * sin(theta)],
                                         dtype=np.float64)
    
    for k in norms:
        for theta in thetas:
            for phi in phis:
                yield do_test_apply, vec(k, theta, phi), mat

@nottest
def do_test_apply_null_wave_vector(dim):
    mat = Material(0.7, 0.3, dim)
    if dim == 2:
        green = GreenOperator2d(mat)
    elif dim == 3:
        green = GreenOperator3d(mat)
    k = np.zeros((dim,), dtype=np.float)
    sym = (dim * (dim + 1)) // 2
    for i in range(sym):
        tau = np.zeros((sym,), dtype=np.float)
        tau[i] = 1.
        eps = green.apply(k, tau)
        for j in range(sym):
            msg = 'coefficient [{0}, {1}]'.format(i, j)
            assert_equal(eps[j], 0., msg=msg)

def test_apply_null_wave_vector():
    dims = [2, 3]
    for dim in dims:
        yield do_test_apply_null_wave_vector, dim
