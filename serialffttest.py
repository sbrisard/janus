from nose.tools import assert_less
from nose.tools import nottest
import numpy as np
import numpy.random as nprnd
import numpy.fft as npfft
from serialfft import *

@nottest
def do_test_r2c_2d(shape, delta):

    fft = SerialFFT2D(shape)
    a = 2. * nprnd.rand(*shape) - 1.
    actual = np.empty(fft.out_shape(), dtype=np.complex128)
    fft.r2c(a, actual)
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

