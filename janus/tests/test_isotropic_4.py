import itertools

import numpy as np

from nose.tools import nottest
from nose.tools import raises
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal_nulp

from janus.operators import isotropic_4

# The value of one unit in the last place, for the float64 format.
ULP = np.finfo(np.float64).eps

#
# Test of the apply() method
# ==========================
#
# Valid parameters
# ----------------
#

@nottest
def do_test_apply(sph, dev, dim, flag):
    """flag allows the specification of various calling sequences:
      - flag = 0: apply(x)
      - flag = 1: apply(x, x)
      - flag = 2: apply(x, y)
    """
    sym = (dim * (dim + 1)) // 2
    t = isotropic_4(sph, dev, dim)

    x = np.empty((sym,), dtype=np.float64)
    expected = np.empty_like(x)

    if flag == 0:
        base = None
    elif flag == 1:
        base = x
    elif flag == 2:
        base = np.empty_like(x)

    for i in range(sym):
        x[:] = 0.
        x[i] = 1.

        if i < dim:
            expected[0:dim] = (sph - dev) / dim
            expected[i] = (sph + (dim - 1) * dev) / dim
            expected[dim:] = 0.
        else:
            expected[:] = 0.
            expected[i] = dev

        actual = t.apply(x, base)
        print(expected, np.asarray(actual))
        assert_array_almost_equal_nulp(expected, np.asarray(actual), 1)

def test_apply():
    coefs = [(1., 0.), (0., 1.), (2.5, -3.5)]
    for dim in [2, 3]:
        for flag in [0, 1, 2]:
            for sph, dev in coefs:
                yield do_test_apply, sph, dev, dim, flag

#
# Invalid parameters
# ------------------
#

def test_apply_invalid_params():
    for dim in [2, 3]:
        sym = (dim * (dim + 1)) // 2
        t = isotropic_4(2.5, -3.5, dim)
        valid = np.zeros((sym,), dtype=np.float64)
        invalid = np.zeros((sym + 1), dtype=np.float64)

        @raises(ValueError)
        def test(x, y):
            t.apply(x, y)

        params = [(invalid, valid), (valid, invalid)]
        for x, y in params:
            yield test, x, y

#
# Test of the to_memoryview() method
# ==================================
#
# Valid parameters
# ----------------
#

@nottest
def do_test_to_memoryview(sph, dev, dim):
    sym = (dim * (dim + 1)) // 2
    I = np.eye(sym)
    aux = np.array(list(itertools.chain(itertools.repeat(1., dim),
                                        itertools.repeat(0., sym - dim))))
    aux = aux.reshape(sym, 1)
    J = 1. / dim * aux * aux.T
    K = I - J
    expected = sph * J + dev * K
    actual = isotropic_4(sph, dev, dim).to_memoryview()
    assert_allclose(expected, actual, ULP, ULP)

def test_to_memoryview():
    for dim in [2, 3]:
        for sph, dev in [(1.0, 0.0), (0.0, 1.0), (2.5, -3.5)]:
            yield do_test_to_memoryview, sph, dev, dim

#
# Invalid parameters
# ------------------
#

def test_to_memoryview_invalid_params():
    @raises(ValueError)
    def test(dim, shape):
        out = np.empty(shape, dtype=np.float64)
        isotropic_4(0., 0., dim).to_memoryview(out)

    for dim in [2, 3]:
        sym = (dim * (dim + 1)) // 2
        shapes = [(sym - 1, sym), (sym + 1, sym),
                  (sym, sym - 1), (sym, sym + 1)]
        for shape in shapes:
            yield test, dim, shape

