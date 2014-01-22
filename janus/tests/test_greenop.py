import numpy as np

import janus.greenop as greenop
import janus.mandelvoigt as mv

from nose.tools import nottest
from nose.tools import raises
from numpy import cos
from numpy import sin
from numpy.testing import assert_array_almost_equal_nulp

from janus.matprop import IsotropicLinearElasticMaterial as Material

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
             + delta(j, l) * n[i] * n[k])
            - 2 * n[i] * n[j] * n[k] * n[l] / (1. - mat.nu)) / (4 * mat.g)

def green_matrix(k, mat):
    k2 = sum(k**2)
    if k2 == 0.:
        sym = (mat.dim * (mat.dim + 1)) // 2
        return np.zeros((sym, sym), dtype=np.float64)
    else:
        n = k / np.sqrt(k2)
        return mv.get_instance(mat.dim).create_array(green_coefficient, n, mat)

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
def do_test_apply(k, mat, flag):
    """flag allows the specification of various calling sequences:
      - flag = 0: apply_single_freq(b, tau)
      - flag = 1: apply_single_freq(b, tau, tau)
      - flag = 2: apply_single_freq(b, tau, eta)
    """
    expected = green_matrix(k,  mat)
    green = greenop.create(mat)
    tau = np.zeros((green.ncols,), np.float64)
    if flag == 0:
        base = None
    elif flag == 1:
        base = tau
    elif flag == 2:
        base = np.empty((green.nrows,), np.float64)
    else:
        raise ValueError()

    for j in range(green.ncols):
        tau[:] = 0.
        tau[j] = 1.
        actual = green.apply(k, tau, base)
        if flag != 0:
            assert actual.base is base
        assert_array_almost_equal_nulp(expected[:, j], actual, 325)

def test_apply():
    for dim in DIMS:
        mat = Material(MU, NU, dim)
        for k in wave_vectors(dim):
            for flag in range(3):
                yield do_test_apply, k, mat, flag

@nottest
def do_test_as_array(k, mat, inplace):
    expected = green_matrix(k,  mat)
    green = greenop.create(mat)
    if inplace:
        base = np.empty((green.nrows, green.ncols), np.float64)
        actual = green.as_array(k, base)
        assert actual.base is base
    else:
        actual = green.as_array(k)
    assert_array_almost_equal_nulp(expected, actual, 325)

def test_as_array():
    for dim in DIMS:
        mat = Material(MU, NU, dim)
        for k in wave_vectors(dim):
            for inplace in [True, False]:
                yield do_test_as_array, k, mat, inplace

@raises(ValueError)
def test_init_2D_invalid_dimension():
    greenop.GreenOperator2D(Material(MU, NU, 3))

@raises(ValueError)
def test_init_3D_invalid_dimension():
    greenop.GreenOperator3D(Material(MU, NU, 2))

@nottest
@raises(ValueError)
def do_test_apply_invalid_params(green, k, tau, eps):
    green.apply(k, tau, eps)

def test_apply_invalid_params():
    g2 = greenop.create(Material(MU, NU, 2))
    @raises(ValueError)
    def apply2(k, tau, eps):
        return g2.apply(k, tau, eps)

    g3 = greenop.create(Material(MU, NU, 3))
    @raises(ValueError)
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

def test_as_array_invalid_params():
    g2 = greenop.create(Material(MU, NU, 2))
    @raises(ValueError)
    def as_array2(k, arr):
        return g2.as_array(k, arr)

    g3 = greenop.create(Material(MU, NU, 3))
    @raises(ValueError)
    def as_array3(k, arr):
        return g3.as_array(k, arr)

    k2 = np.empty((2,))
    k3 = np.empty((3,))
    arr3x3 = np.empty((3, 3), dtype=np.float64)
    arr6x6 = np.empty((6, 6), dtype=np.float64)
    arr3x6 = np.empty((3, 6), dtype=np.float64)
    arr6x3 = np.empty((6, 3), dtype=np.float64)
    all_as_array = [as_array2, as_array2, as_array2,
                    as_array3, as_array3, as_array3]
    all_k = [k3, k2, k2, k2, k3, k3]
    all_arr = [arr3x3, arr6x3, arr3x6, arr6x6, arr6x3, arr3x6]

    for as_array, k, arr in zip(all_as_array, all_k, all_arr):
        yield as_array, k, arr
