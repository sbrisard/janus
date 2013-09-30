from nose.tools import assert_less
from nose.tools import nottest
from nose.tools import raises
import numpy as np
import numpy.random as nprnd
import numpy.fft as npfft
from serialfft import *

def create_fft(shape):
    if len(shape) == 2:
        return SerialFFT2D(shape)
    elif len(shape) == 3:
        return SerialFFT3D(shape)
    else:
        raise ValueError()

@nottest
def do_test_r2c_2d(shape, inplace, delta):

    fft = create_fft(shape)
    a = 2. * nprnd.rand(*shape) - 1.
    if inplace:
        dummy = fft.r2c(a, np.empty(fft.out_shape, dtype=np.float64))
    else:
        dummy = fft.r2c(a)

    dummy = np.asarray(dummy)
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
        for inplace in [True, False]:
            yield do_test_r2c_2d, shape, inplace, delta

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

@nottest
@raises(ValueError)
def do_test_r2c_invalid_parameters(shape, input_shape, output_shape):
    ain = np.empty(input_shape, dtype = np.float64)
    aout = np.empty(output_shape, dtype = np.float64)
    create_fft(shape).r2c(ain, aout)

def test_r2c_invalid_parameters():
    params = [((128, 256), (127, 256), (128, 258)),
              ((128, 256), (128, 255), (128, 258)),
              ((128, 256), (128, 256), (129, 258)),
              ((128, 256), (128, 256), (128, 259)),]
    for shape, input_shape, output_shape in params:
        yield do_test_r2c_invalid_parameters, shape, input_shape, output_shape
