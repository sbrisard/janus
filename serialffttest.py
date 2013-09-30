from nose.tools import assert_less
from nose.tools import nottest
from nose.tools import raises
import numpy as np
import numpy.random as nprnd
import numpy.fft as npfft
from serialfft import *

@nottest
def do_test_r2c_2d(shape, delta):

    fft = SerialFFT2D(shape)
    a = 2. * nprnd.rand(*shape) - 1.
    dummy = np.empty(fft.out_shape, dtype=np.float64)
    fft.r2c(a, dummy)
    actual = dummy[:, 0::2] + 1j * dummy[:, 1::2]
    expected = npfft.rfft2(a)

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert_less(error, delta)

def test_r2c_2d():
    params = [((256, 128), 2E-15),
              ((256, 129), 2E-14),
              ((257, 128), 2E-15),
              ((257, 129), 2E-14)]
    for shape, delta in params:
        yield do_test_r2c_2d, shape, delta

@nottest
@raises(ValueError)
def do_test_invalid_dimension(dim):
    if dim == 2:
        return SerialFFT2D((128, 128, 128))
    else:
        return SerialFFT3D((128, 128))

def test_invalid_dimension():
    dims = [2]
    for dim in dims:
        yield do_test_invalid_dimension, dim
