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
