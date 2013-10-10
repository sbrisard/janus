import numpy as np
from numpy import cos
from numpy import sin

from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import nottest
from nose.tools import raises

import mandelvoigt as mv
from matprop import IsotropicLinearElasticMaterial as Material
from greenop import GreenOperator2d
from greenop import GreenOperator3d

DIMS = [2, 3]
MU = 0.7
NU = 0.3

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

def green_matrix(k, mat):
    k2 = sum(k**2)
    if k2 == 0.:
        sym = (mat.dim * (mat.dim + 1)) // 2
        return np.zeros((sym, sym), dtype=np.float64)
    else:
        n = k / np.sqrt(k2)
        return mv.get_instance(mat.dim).create_array(green_coefficient, n, mat)

def create_green_operator(mat):
    if mat.dim == 2:
        return GreenOperator2d(mat)
    elif mat.dim == 3:
        return GreenOperator3d(mat)

def wave_vectors(dim):
    norms = [2.5, 3.5]
    if dim == 2:
        num_angles = 20
        thetas = [(2. * np.pi * i) / num_angles for i in range(num_angles)]
        def vec(r, theta):
            return np.array([r * cos(theta), r * sin(theta)], dtype=np.float64)
        ret = [vec(k, theta) for theta in thetas for k in norms]
    elif dim == 3:
        num_thetas = 10
        num_phis = 20
        thetas = [(np.pi * i) / num_thetas for i in range(num_thetas)]
        phis = [(2. * np.pi * i) / num_phis for i in range(num_phis)]
        def vec(r, theta, phi):
            return np.array([r * cos(theta) * cos(phi),
                             r * cos(theta) * sin(phi),
                             r * sin(theta)],
                            dtype=np.float64)
        ret = [vec(k, theta, phi) for phi in phis for theta in thetas
               for k in norms]
    else:
        raise ValueError()
    ret.append(np.zeros((dim,), dtype=np.float64))
    return ret

@nottest
def do_test_apply(k, mat, in_place):
    expected = green_matrix(k,  mat)
    sym = (mat.dim * (mat.dim + 1)) // 2
    green = create_green_operator(mat)
    if in_place:
        eps = np.empty((sym,), np.float64)
    else:
        eps = None

    for j in range(sym):
        tau = np.zeros((sym,), np.float64)
        tau[j] = 1.
        actual = green.apply(k, tau, eps)
        if in_place:
            msg = 'actual.base and eps should be the same object'
            assert actual.base is eps, msg
        for i in range(sym):
            msg = 'coefficient [{0}, {1}]'.format(i, j)
            assert_almost_equal(expected[i, j], actual[i],
                                msg=msg, delta=1.E-15)

def test_apply():
    for in_place in [True, False]:
        for dim in DIMS:
            mat = Material(MU, NU, dim)
            for k in wave_vectors(dim):
                yield do_test_apply, k, mat, in_place

@nottest
def do_test_asarray(k, mat):
    expected = green_matrix(k,  mat)
    sym = (mat.dim * (mat.dim + 1)) // 2
    green = create_green_operator(mat)
    actual = np.empty((sym, sym), np.float64)
    green.asarray(k, actual)
    for j in range(sym):
        for i in range(sym):
            msg = 'coefficient [{0}, {1}]'.format(i, j)
            assert_almost_equal(expected[i, j], actual[i, j],
                                msg=msg, delta=1.E-15)

def test_asarray():
    for dim in DIMS:
        mat = Material(MU, NU, dim)
        for k in wave_vectors(dim):
            yield do_test_asarray, k, mat

@raises(ValueError)
def test_init_2d_invalid_dimension():
    GreenOperator2d(Material(MU, NU, 3))

@raises(ValueError)
def test_init_3d_invalid_dimension():
    GreenOperator3d(Material(MU, NU, 2))

@nottest
@raises(IndexError)
def do_test_apply_invalid_params(green, k, tau, eps):
    green.apply(k, tau, eps)
    
def test_apply_invalid_params():
    g2 = create_green_operator(Material(MU, NU, 2))
    @raises(IndexError)
    def apply2(k, tau, eps):
        return g2.apply(k, tau, eps)

    g3 = create_green_operator(Material(MU, NU, 3))
    @raises(IndexError)
    def apply3(k, tau, eps):
        return g3.apply(k, tau, eps)
    
    k2 = np.empty((2,), dtype=np.float64)
    k3 = np.empty((3,), dtype=np.float64)
    tau3 = np.empty((3,), dtype=np.float64)
    tau6 = np.empty((6,), dtype=np.float64)
    eps3 = np.empty((3,), dtype=np.float64)
    eps6 = np.empty((6,), dtype=np.float64)
    all_apply = [apply2, apply2, apply2, apply3, apply3, apply3]
    all_k = [k3, k2, k2, k2, k3, k3]
    all_tau = [tau3, tau6, tau3, tau6, tau3, tau6]
    all_eps = [eps3, eps3, eps6, eps6, eps6, eps3]

    for f, k, tau, eps in zip(all_apply, all_k, all_tau, all_eps):
        yield f, k, tau, eps

def test_asarray_invalid_params():
    g2 = create_green_operator(Material(MU, NU, 2))
    @raises(IndexError)
    def asarray2(k, arr):
        return g2.asarray(k, arr)

    g3 = create_green_operator(Material(MU, NU, 3))
    @raises(IndexError)
    def asarray3(k, arr):
        return g3.asarray(k, arr)
    
    k2 = np.empty((2,))
    k3 = np.empty((3,))
    arr3x3 = np.empty((3, 3), dtype=np.float64)
    arr6x6 = np.empty((6, 6), dtype=np.float64)
    arr3x6 = np.empty((3, 6), dtype=np.float64)
    arr6x3 = np.empty((6, 3), dtype=np.float64)
    all_asarray = [asarray2, asarray2, asarray2, asarray3, asarray3, asarray3]
    all_k = [k3, k2, k2, k2, k3, k3]
    all_arr = [arr3x3, arr6x3, arr3x6, arr6x6, arr6x3, arr3x6]

    for asarray, k, arr in zip(all_asarray, all_k, all_arr):
        yield asarray, k, arr
