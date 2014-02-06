import itertools
import os.path

import numpy as np
import numpy.random as rnd

import janus.discretegreenop
import janus.fft.serial
import janus.greenop

from nose.tools import nottest
from nose.tools import raises
from numpy.testing import assert_array_almost_equal_nulp

from janus.matprop import IsotropicLinearElasticMaterial as Material

# All tests are performed with discrete Green operators based on these grids
GRID_SIZES = ([(8, 8), (8, 16), (16, 8), (4, 4, 4)]
              + list(itertools.permutations((4, 8, 16))))

def discrete_green_operator(n, h, transform=None):
    """Create a discrete Green operator with default material constants
    g = 0.75 and nu = 0.3. The gridsize is specified by the tuple n, the
    length of which gives the dimension of the physical space (2 or 3). h is
    the voxel size (can be fixed to 1., as it plays no role in linear
    elasticity.
    """

    mat = Material(0.75, 0.3, len(n))
    if mat.dim == 2:
        return janus.discretegreenop.FilteredGreenOperator2D(janus.greenop.create(mat), n, h, transform)
    else:
        return janus.discretegreenop.create(janus.greenop.create(mat), n, h, transform)

def get_base(a):
    # TODO This is a hack to avoid writing uggly things like a.base.base.base.
    if a.base is None:
        return a
    else:
        return get_base(a.base)

def multi_indices(n):
    """Returns a list of all multi-indices [b[0], ..., b[dim - 1]] such that
    0 <= b[i] < n[i] for all i.

    """
    iterables = [range(ni) for ni in n]
    return [np.asarray(b, dtype=np.intc)
            for b in itertools.product(*iterables)]


def invalid_multi_indices(n):
    """Return a list of multi-indices which are incompatible with the specified
    grid-size (too many indices, negative and too large indices).

    """
    def f(i, x):
        b = np.zeros(len(n), dtype=np.intc)
        b[i] = x
        return b

    indices = [f(i, x) for i in range(len(n)) for x in [n[i], -1]]
    indices.append(np.zeros((len(n) + 1,), dtype=np.intc))
    return indices

def increment_element(a, i):
    """Return a copy of a (as a list), where the i-th element is incremented."""
    return [(a[j] + (1 if j==i else 0)) for j in range(len(a))]

def invalid_shapes(shape):
    """Return a list of invalid shapes, based on the specified (valid) shape.
    Each invalid shape is derived from the valid shape by incrementing one
    element at a time.

    """
    return [increment_element(shape, i) for i in range(len(shape))]

def invalid_tau_eta(valid_tau_shape, valid_eta_shape):
    """Returns a list of invalid (tau, eta) parameters."""
    valid_tau = np.zeros(valid_tau_shape)
    valid_eta = np.zeros(valid_eta_shape)

    invalid_tau_shapes = invalid_shapes(valid_tau_shape)
    invalid_eta_shapes = invalid_shapes(valid_eta_shape)

    params1 = [(np.zeros(shape), None) for shape in invalid_tau_shapes]
    params2 = [(np.zeros(shape), valid_eta) for shape in invalid_tau_shapes]
    params3 = [(valid_tau, np.zeros(shape)) for shape in invalid_eta_shapes]

    return params1 + params2 + params3

#
# 0 Test of the constructor with invalid parameters
#   ===============================================
#

def test_cinit_invalid_params():
    @raises(ValueError)
    def test(green, n, h, transform=None):
        janus.discretegreenop.create(green, n, h, transform)

    green_2D = janus.greenop.create(Material(0.75, 0.3, 2))
    green_3D = janus.greenop.create(Material(0.75, 0.3, 3))

    params = [(green_3D, (9, 9), 1., None),
              (green_2D, (-1, 9), 1., None),
              (green_2D, (9, -1), 1., None),
              (green_2D, (9, 9), -1., None),
              (green_2D, (9, 9), 1., janus.fft.serial.create_real((8, 9))),
              (green_2D, (9, 9), 1., janus.fft.serial.create_real((9, 8))),
              (green_2D, (9, 9, 9), 1., None),
              (green_3D, (-1, 9, 9), 1., None),
              (green_3D, (9, -1, 9), 1., None),
              (green_3D, (9, 9, -1), 1., None),
              (green_3D, (9, 9, 9), -1., None)]

    for green, n, h, transform in params:
        yield test, green, n, h, transform

#
# 1 Test of the method to_memoryview
#   ================================
#
# 1.1 Valid parameters
#     ----------------
#

@nottest
def do_test_to_memoryview(n, inplace):

    greend = discrete_green_operator(n, 1.0)
    greenc = greend.green
    k = np.empty((len(n),), dtype=np.float64)

    for b in multi_indices(n):
        i = 2 * b > n
        k[i] = 2. * np.pi * (b[i] - n[i]) / n[i]
        k[~i] = 2. * np.pi * b[~i] / n[~i]

        greend.set_frequency(b)
        greenc.set_frequency(k)
        expected = greenc.to_memoryview()

        if inplace:
            base = np.empty_like(expected)
            actual = greend.to_memoryview(base)
            assert get_base(actual) is base
        else:
            actual = greend.to_memoryview()
        print(np.asarray(expected), np.asarray(actual))
        assert_array_almost_equal_nulp(np.asarray(expected),
                                       np.asarray(actual), 1)

def test_to_memoryview():
    for n in GRID_SIZES:
        for inplace in [True, False]:
            yield do_test_to_memoryview, np.asarray(n), inplace

#
# 1.2 Invalid parameters
#     ------------------
#

def test_to_memoryview_invalid_parameters():
    for dim in [2, 3]:
        n = tuple(2**(i + 3) for i in range(dim))
        green = discrete_green_operator(n, 1.)
        out1 = np.zeros((green.oshape[-1] + 1, green.ishape[-1]),
                        dtype=np.float64)
        out2 = np.zeros((green.oshape[-1], green.ishape[-1] + 1),
                        dtype=np.float64)

        @raises(ValueError)
        def test(out):
            green.to_memoryview(out)

        for out in [out1, out2]:
            yield test, out

#
# 2 Test of the method apply_by_freq
#   ================================
#
# 2.1 Valid parameters
#     ----------------
#

@nottest
def do_test_apply(n, tau, flag):
    """flag allows the specification of various calling sequences:
      - flag = 0: apply_single_freq(b, tau)
      - flag = 1: apply_single_freq(b, tau, tau)
      - flag = 2: apply_single_freq(b, tau, eta)
    """
    dim = len(n)
    green = discrete_green_operator(n, 1.)

    for b in multi_indices(n):
        green.set_frequency(b)
        g = np.asarray(green.to_memoryview())
        # expected is by default a matrix, so that it has two dimensions.
        # First convert to ndarray so as to reshape is to a 1D array.
        expected = np.dot(g, tau)
        if flag == 0:
            base = None
        elif flag == 1:
            base = tau
        elif flag == 2:
            base = np.empty_like(expected)
        else:
            raise(ValueError)
        actual = green.apply_by_freq(tau, base)
        if flag != 0:
            assert get_base(actual) is base
        assert_array_almost_equal_nulp(expected, np.asarray(actual), 1)

def test_apply():
    shapes = GRID_SIZES
    tau = [np.array([0.3, -0.4, 0.5]),
           np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])]

    for n in GRID_SIZES:
        for flag in range(3):
            yield do_test_apply, n, tau[len(n) - 2], flag

#
# 2.2 Invalid parameters
#     ------------------
#

def test_apply_invalid_params():
    for dim in [2, 3]:
        n = tuple(2**(i + 3) for i in range(dim))
        green = discrete_green_operator(n, 1.)

        tau_valid = np.zeros((green.ishape[-1],), dtype=np.float64)
        tau_invalid = np.zeros((green.ishape[-1] + 1,), dtype=np.float64)
        eta_invalid = np.zeros((green.oshape[-1] + 1,), dtype=np.float64)

        @raises(ValueError)
        def test(tau, eta):
            green.apply_by_freq(tau, eta)

        for tau, eta in [(tau_invalid, None), (tau_valid, eta_invalid)]:
            yield test, tau, eta

# #
# # 3 Test of the method apply_all_freqs
# #   ==================================
# #
# # 3.1 Valid parameters
# #     ----------------
# #

# @nottest
# def do_test_apply_all_freqs(n, flag):
#     """flag allows the specification of various calling sequences:
#       - flag = 0: apply_all_freqs(tau)
#       - flag = 1: apply_all_freqs(tau, tau)
#       - flag = 2: apply_all_freqs(tau, eta)
#     """
#     green = discrete_green_operator(n, 1.)
#     tau = rnd.rand(*(n + (green.isize,)))
#     expected = np.empty(n + (green.osize,), dtype=np.float64)
#     for b in multi_indices(n):
#         index = tuple(b)
#         green.apply_single_freq(b, tau[index], expected[index])
#     if flag == 0:
#         base = None
#     elif flag == 1:
#         base = tau
#     elif flag == 2:
#         base = np.empty_like(expected)
#     else:
#         raise ValueError()
#     actual = green.apply_all_freqs(tau, base)
#     if flag != 0:
#         assert get_base(actual) is base
#     assert_array_almost_equal_nulp(expected, np.asarray(actual), 0)

# def test_apply_all_freqs():
#     for n in GRID_SIZES:
#         for flag in range(3):
#             yield do_test_apply_all_freqs, n, flag

# #
# # 3.2 Invalid parameters
# #     ------------------
# #

# def test_apply_all_freqs_invalid_params():
#     for dim in [2, 3]:
#         n = tuple(2**(i + 3) for i in range(dim))
#         green = discrete_green_operator(n, 1.)
#         params = invalid_tau_eta(n + (green.isize,),
#                                  n + (green.osize,))

#         @raises(ValueError)
#         def test(tau, eta):
#             green.apply_all_freqs(tau, eta)

#         for tau, eta in params:
#             yield test, tau, eta

#
# 4 Test of the method apply
#   ========================
#
# 4.1 Valid parameters
#     ----------------
#
# These tests are based on convolutions computed with an independent
# (Java-based) code. References values are stored as an array in a *.npy file.
# The array ref is organized as follows
#   - ref[:, :, 0:3] = tau
#   - ref[:, :, 3:6] = eta = - Gamma * tau

@nottest
def do_test_apply(path_to_ref, rel_err):
    dummy = np.load(path_to_ref)
    n = dummy.shape[:-1]
    transform = janus.fft.serial.create_real(n)
    green = discrete_green_operator(n, 1., transform)

    # TODO This is uggly
    if green.dim == 2:
        tau = dummy[:, :, 0:green.ishape[-1]]
        expected = dummy[:, :, green.ishape[-1]:]
    else:
        tau = dummy[:, :, :, 0:green.ishape[-1]]
        expected = dummy[:, :, :, green.ishape[-1]:]

    actual = np.zeros(transform.rshape + (green.oshape[-1],), np.float64)
    green.apply(tau, actual)

    error = actual - expected
    ulp = np.finfo(np.float64).eps
    nulp = rel_err / ulp
    assert_array_almost_equal_nulp(expected, np.asarray(actual), nulp)

def test_apply():

    directory = os.path.dirname(os.path.realpath(__file__))

    params = [('filtered_green_operator_200x300_unit_tau_xx_10x10+95+145.npy',
               7.1E-11),
              ('filtered_green_operator_200x300_unit_tau_yy_10x10+95+145.npy',
               1.6E-10),
              ('filtered_green_operator_200x300_unit_tau_xy_10x10+95+145.npy',
               6.7E-11),
              ('truncated_green_operator_40x50x60_unit_tau_xx_10x10x10+15+20+25.npy',
               1.7E-10),
              ('truncated_green_operator_40x50x60_unit_tau_yy_10x10x10+15+20+25.npy',
               7.6E-10),
              ('truncated_green_operator_40x50x60_unit_tau_zz_10x10x10+15+20+25.npy',
               1.6E-9),
              ('truncated_green_operator_40x50x60_unit_tau_yz_10x10x10+15+20+25.npy',
               1.6E-10),
              ('truncated_green_operator_40x50x60_unit_tau_zx_10x10x10+15+20+25.npy',
               6.4E-10),
              ('truncated_green_operator_40x50x60_unit_tau_xy_10x10x10+15+20+25.npy',
               1.6E-9),
               ]
    for filename, rel_err in params:
        path = os.path.join(directory, 'data', filename)
        yield do_test_apply, path, rel_err

#
# 4.2 Invalid parameters
#     ------------------
#

def test_apply_invalid_params():
    for dim in [2, 3]:
        n = tuple(2**(i + 3) for i in range(dim))
        transform = janus.fft.serial.create_real(n)
        green = discrete_green_operator(n, 1., transform)
        params = invalid_tau_eta(n + (green.ishape[-1],),
                                 n + (green.oshape[-1],))

        @raises(ValueError)
        def test(tau, eta):
            green.apply(tau, eta)

        for tau, eta in params:
            yield test, tau, eta
