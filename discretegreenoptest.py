# if __name__ == '__main__':
#     import sys
#     sys.path.append('./src/')

import discretegreenop
import greenop
import itertools
import numpy as np

from numpy.testing import assert_array_max_ulp
from matprop import IsotropicLinearElasticMaterial as Material
from nose.tools import nottest

def get_base(a):
    """This is a hack. To avoid writing uggly things like a.base.base.base."""
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
    shapes = ((8, 8), (8, 16), (8, 8, 8), (8, 16, 32))
    for n in shapes:
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
    shapes = ((8, 8), (8, 16), (8, 8, 8), (8, 16, 32))

    tau = [np.array([0.3, -0.4, 0.5]),
           np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6])]

    for n in shapes:
        for inplace in [False, True]:
            yield do_test_apply_single_freq, n, tau[len(n) - 2], inplace
