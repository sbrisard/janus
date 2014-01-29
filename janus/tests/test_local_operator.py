import itertools

import numpy as np
import numpy.random as rnd

import janus.local_operator as local_operator
import janus.utils.tensors as tensors

from nose.tools import nottest
from numpy.testing import assert_array_almost_equal_nulp

def indices(shape):
    return itertools.product(*map(range, shape))

def create_local_operators(shape):
    dim = len(shape)
    ops = np.empty(shape, dtype=object)
    for index in indices(shape):
        ops[index] = tensors.isotropic_4(2. * rnd.rand() - 1,
                                         2. * rnd.rand() - 1, dim)
    return ops

@nottest
def do_test_apply(shape, nulp):
    dim = len(shape)
    nrows = (dim * (dim + 1)) // 2
    ncols = (dim * (dim + 1)) // 2
    x = rnd.rand(*(shape + (ncols,)))
    expected = np.empty(shape + (nrows,), dtype=np.float64)

    ops = create_local_operators(shape)

    for index in indices(shape):
        ops[index].apply(x[index], expected[index])

    actual = np.empty_like(expected)
    local_operator.BlockDiagonalOperator2D(ops).apply(x, actual)

    assert_array_almost_equal_nulp(expected, np.asarray(actual), nulp)

def test_apply():
    do_test_apply((32, 64), 1)
