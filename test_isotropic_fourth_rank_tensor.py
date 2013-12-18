import numpy as np

from nose.tools import nottest
from nose.tools import raises
from numpy.testing import assert_array_almost_equal_nulp

from tensors import create_fourth_rank_isotropic as tensor

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
    t = tensor(sph, dev, dim)

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
        t = tensor(2.5, -3.5, dim)
        valid = np.zeros((sym,), dtype=np.float64)
        invalid = np.zeros((sym + 1), dtype=np.float64)

        @raises(ValueError)
        def test(x, y):
            t.apply(x, y)

        params = [(invalid, valid), (valid, invalid)]
        for x, y in params:
            yield test, x, y
