if __name__ == '__main__':
    import sys
    sys.path.append('./src/')

# TODO Implement test of apply_single_freq with invalid params.

import discretegreenop
import greenop
import itertools
import numpy as np
import numpy.random as rnd

from numpy.testing import assert_array_max_ulp
from matprop import IsotropicLinearElasticMaterial as Material
from nose.tools import nottest
from nose.tools import raises

# All tests are performed with discrete Green operators based on these
# grids
GRID_SIZES = [(8, 8), (8, 16), (16, 8), (8, 8, 8), (8, 16, 32)]
#GRID_SIZES = [(8, 8), (8, 16)]

def discrete_green_operator(n, h):
    """Create a discrete Green operator with default material constants
    g = 0.75 and nu = 0.3. The gridsize is specified by the tuple n, the
    length of which gives the dimension of the physical space (2 or 3). h is
    the voxel size (can be fixed to 1., as it plays no role in linear
    elasticity.
    """

    mat = Material(0.75, 0.3, len(n))
    greenc = greenop.create(mat)
    if mat.dim == 2:
        return discretegreenop.TruncatedGreenOperator2D(greenc, n, h)
    else:
        return discretegreenop.TruncatedGreenOperator(greenc, n, h)

def get_base(a):
    # TODO This is a hack to avoid writing uggly things like a.base.base.base.
    if a.base is None:
        return a
    else:
        return get_base(a.base)

def invalid_frequency_multi_indices(n):
    """Return a list of frequency multi-indices which are incompatible with the
    specified grid-size (too many indices, negative and too large indices).

    """
    def f(i, x):
        a = np.zeros(len(n), dtype=np.intp)
        a[i] = x
        return a

    ret = [f(i, x) for i in range(len(n)) for x in [n[i], -1]]
    ret.append(np.zeros((len(n) + 1,), dtype=np.intp))
    return ret

def increment_element(a, i):
    """Return a copy of a (as a list), where the i-th element is incremented."""
    return [(a[j] + (1 if j==i else 0)) for j in range(len(a))]

def invalid_shapes(shape):
    """Return a list of invalid shapes, based on the specified (valid) shape.
    Each invalid shape is derived from the valid shape by incrementing one
    element at a time.

    """
    return [increment_element(shape, i) for i in range(len(shape))]

#
# 1 Test of the method as_array
#   ===========================
#
# 1.1 Valid parameters
#     ----------------
#

@nottest
def do_test_as_array(n, inplace):
    mat = Material(0.75, 0.3, len(n))

    n_arr = np.asarray(n)

    # Continuous Green operator
    greenc = greenop.create(mat)

    # Discrete Green operator
    greend = discretegreenop.TruncatedGreenOperator(greenc, n, 1.0)

    # Wave-vector
    k = np.empty((len(n),), dtype=np.float64)

    iterables = [range(ni) for ni in n]
    for b in itertools.product(*iterables):
        b_arr = np.asarray(b)
        i = 2 * b_arr > n_arr
        k[i] = 2. * np.pi * (b_arr[i] - n_arr[i]) / n_arr[i]
        k[~i] = 2. * np.pi * b_arr[~i] / n_arr[~i]

        expected = greenc.as_array(k)

        if inplace:
            base = np.empty_like(expected)
            actual = greend.as_array(b_arr, base)
            assert get_base(actual) is base
        else:
            actual = greend.as_array(b_arr)
        assert_array_max_ulp(expected, actual, 1)

def test_as_array():
    for n in GRID_SIZES:
        for inplace in [True, False]:
            yield do_test_as_array, n, inplace

#
# 1.2 Invalid parameters
#     ------------------
#

def test_as_array_invalid_parameters():
    for dim in [2, 3]:
        n = tuple(2**(i + 3) for i in range(dim))
        green = discrete_green_operator(n, 1.)
        b0 = np.zeros((green.dim,), dtype=np.intp)
        out1 = np.zeros((green.nrows + 1, green.ncols), dtype=np.float64)
        out2 = np.zeros((green.nrows, green.ncols + 1), dtype=np.float64)
        params = ([(b, None) for b in invalid_frequency_multi_indices(n)]
                  + [(b0, out1), (b0, out2)])

        @raises(ValueError)
        def test(b, out):
            green.as_array(b, out)

        for b, out in params:
            yield test, b, out

#
# 2 Test of the method apply_single_freq
#   ====================================
#
# 2.1 Valid parameters
#     ----------------
#

@nottest
def do_test_apply_single_freq(n, tau, inplace):
    dim = len(n)
    mat = Material(0.75, 0.3, dim)
    greenc = greenop.create(mat)
    greend = discretegreenop.TruncatedGreenOperator(greenc, n, 1.0)

    n_arr = np.asarray(n)
    tau_vec = tau.reshape(greend.ncols, 1)

    iterables = [range(ni) for ni in n]
    for b_tuple in itertools.product(*iterables):
        b = np.asarray(b_tuple)
        g = np.asmatrix(greend.as_array(b))
        # expected is by default a matrix, so that it has two dimensions.
        # First convert to ndarray so as to reshape is to a 1D array.
        expected = np.asarray(g * tau_vec).reshape((greend.nrows,))

        if inplace:
            base = np.empty_like(expected)
            actual = greend.apply_single_freq(b, tau, base)
            assert get_base(actual) is base
        else:
            actual = greend.apply_single_freq(b, tau)

        actual = np.asarray(actual)
        assert_array_max_ulp(expected, actual, 1)

def test_apply_single_freq():
    shapes = GRID_SIZES
    tau = [np.array([0.3, -0.4, 0.5]),
           np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])]

    for n in GRID_SIZES:
        for inplace in [False, True]:
            yield do_test_apply_single_freq, n, tau[len(n) - 2], inplace

#
# 2.2 Invalid parameters
#     ------------------
#

def test_apply_single_freq_invalid_params():
    for dim in [2, 3]:
        n = tuple(2**(i + 3) for i in range(dim))
        green = discrete_green_operator(n, 1.)

        b_valid = np.zeros((dim,), dtype=np.intp)
        tau_valid = np.zeros((green.ncols,), dtype=np.float64)
        tau_invalid = np.zeros((green.ncols + 1,), dtype=np.float64)
        eta_invalid = np.zeros((green.nrows + 1,), dtype=np.float64)

        params = ([(b, tau_valid, None)
                   for b in invalid_frequency_multi_indices(n)]
                  + [(b_valid, tau_invalid, None),
                     (b_valid, tau_valid, eta_invalid)])

        @raises(ValueError)
        def test(b, tau, eta):
            green.apply_single_freq(b, tau, eta)

        for b, tau, eta in params:
            yield test, b, tau, eta

#
# 3 Test of the method apply_all_freqs
#   ==================================
#
# 3.1 Valid parameters
#     ----------------
#

@nottest
def do_test_apply_all_freqs(n, inplace):
    green = discrete_green_operator(n, 1.)
    tau = rnd.rand(*(n[::-1] + (green.ncols,)))
    expected = np.empty(n[::-1] + (green.nrows,), dtype=np.float64)
    b = np.empty((green.dim,), dtype=np.intp)
    for b[0] in range(n[0]):
        for b[1] in range(n[1]):
            green.apply_single_freq(b,
                                    tau[b[1], b[0], :],
                                    expected[b[1], b[0], :])
    if inplace:
        base = np.empty_like(expected)
        actual = green.apply_all_freqs(tau, base)
        assert get_base(actual) is base
    else:
        actual = green.apply_all_freqs(tau)

    assert_array_max_ulp(expected, actual, 0)

def test_apply_all_freqs():
    for n in GRID_SIZES:
        # TODO remove this statement
        if len(n) == 2:
            for inplace in [True, False]:
                yield do_test_apply_all_freqs, n, inplace

#
# 3.2 Invalid parameters
#     ------------------
#

def test_apply_all_freqs_invalid_params():
    for dim in [2, 3]:
        n = tuple(2**(i + 3) for i in range(dim))
        green = discrete_green_operator(n, 1.)

        tau = np.zeros(n[::-1] + (green.ncols,), dtype=np.float64)
        eta = np.zeros(n[::-1] + (green.nrows,), dtype=np.float64)

        tau_invalid_shapes = invalid_shapes(tau.shape)
        eta_invalid_shapes = invalid_shapes(eta.shape)

        params = ([(np.zeros(shape, dtype=np.float64), None)
                   for shape in tau_invalid_shapes]
                   + [(np.zeros(shape, dtype=np.float64), eta)
                      for shape in tau_invalid_shapes]
                      + [(tau, np.zeros(shape, dtype=np.float64))
                        for shape in eta_invalid_shapes])

        def test(tau, eta):
            green.apply_all_freqs(tau, eta)

        for tau, eta in params:
            yield test, tau, eta
