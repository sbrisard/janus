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
#GRID_SIZES = [(8, 8), (8, 16), (8, 8, 8), (8, 16, 32)]
GRID_SIZES = [(8, 8), (8, 16)]

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

@nottest
def do_test_asarray(n, inplace):
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

        expected = greenc.asarray(k)

        if inplace:
            base = np.empty_like(expected)
            actual = greend.asarray(b_arr, base)
            assert get_base(actual) is base
        else:
            actual = greend.asarray(b_arr)
        assert_array_max_ulp(expected, actual, 1)

def test_asarray():
    for n in GRID_SIZES:
        for inplace in [True, False]:
            yield do_test_asarray, n, inplace

@nottest
def do_test_apply_single_freq(n, tau, inplace):
    dim = len(n)
    sym = (dim * (dim + 1)) // 2
    mat = Material(0.75, 0.3, dim)

    n_arr = np.asarray(n)
    tau_vec = tau.reshape(sym, 1)

    greenc = greenop.create(mat)
    greend = discretegreenop.TruncatedGreenOperator(greenc, n, 1.0)

    iterables = [range(ni) for ni in n]
    for b in itertools.product(*iterables):
        b_arr = np.asarray(b)

        g = np.asmatrix(greend.asarray(b_arr))
        # expected is by default a matrix, so that it has two dimensions.
        # First convert to ndarray so as to reshape is to a 1D array.
        expected = np.asarray(g * tau_vec).reshape((sym,))

        if inplace:
            base = np.empty_like(expected)
            actual = greend.apply_single_freq(b_arr, tau, base)
            assert get_base(actual) is base
        else:
            actual = greend.apply_single_freq(b_arr, tau)

        actual = np.asarray(actual)
        assert_array_max_ulp(expected, actual, 1)

def test_apply_single_freq():
    shapes = GRID_SIZES

    tau = [np.array([0.3, -0.4, 0.5]),
           np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])]

    for n in shapes:
        for inplace in [False, True]:
            yield do_test_apply_single_freq, n, tau[len(n) - 2], inplace

@nottest
def do_test_apply_all_freqs(n, inplace):
    greend = discrete_green_operator(n, 1.)
    dim = len(n)
    sym = (dim * (dim + 1)) // 2
    tau = rnd.rand(n[1], n[0], sym)
    expected = np.empty_like(tau)
    b = np.empty((dim,), dtype=np.intp)
    for b[0] in range(n[0]):
        for b[1] in range(n[1]):
            greend.apply_single_freq(b,
                                     tau[b[1], b[0], :],
                                     expected[b[1], b[0], :])
    if inplace:
        base = np.empty_like(tau)
        actual = greend.apply_all_freqs(tau, base)
        assert get_base(actual) is base
    else:
        actual = greend.apply_all_freqs(tau)

    assert_array_max_ulp(expected, actual, 0)

def test_apply_all_freqs():
    shapes = ((8, 8), (8, 16))
    for n in GRID_SIZES:
        for inplace in [True, False]:
            yield do_test_apply_all_freqs, n, inplace

@raises(ValueError)
def test_apply_all_freqs_tau_wrong_num_dims():
    green = discrete_green_operator((8, 16), 1.)
    green.apply_all_freqs(np.empty((8, 8), dtype=np.float64))

@nottest
@raises(ValueError)
def do_test_apply_all_freqs_tau_invalid_shape(n, shape):
    green = discrete_green_operator(n, 1.)
    dim = len(n)
    sym = (dim * (dim + 1)) // 2
    tau = np.empty(shape + (sym,), dtype=np.float64)
    green.apply_all_freqs(tau)

def incr_elem(n, i):
    n_list = list(n)
    n_list[i] += 1
    return tuple(n_list)

def test_apply_all_freqs_tau_invalid_shape():
    for n in GRID_SIZES:
        n_mirror = list(n)[::-1]
        for shape in [incr_elem(n_mirror, i) for i in range(len(n))]:
            yield do_test_apply_all_freqs_tau_invalid_shape, n, shape

if __name__ == '__main__':
    do_test_apply_all_freqs_tau_invalid_shape((8, 16), (17, 8))
