from nose.tools import assert_less
from nose.tools import nottest
from nose.tools import raises
import numpy as np
import numpy.random as nprnd
import numpy.fft as npfft
from serialfft import *

def create_fft(shape):
    if len(shape) == 2:
        return SerialRealFFT2D(shape)
    elif len(shape) == 3:
        return SerialRealFFT3D(shape)
    else:
        raise ValueError()

@nottest
@raises(ValueError)
def do_test_create_invalid_dimension(dim):
    if dim == 2:
        return SerialRealFFT2D((128, 128, 128))
    else:
        return SerialRealFFT3D((128, 128))

def test_create_invalid_dimension():
    dims = [2]
    for dim in dims:
        yield do_test_create_invalid_dimension, dim

@nottest
def do_test_r2c_2d(shape, inplace, delta):
    fft = create_fft(shape)
    a = 2. * nprnd.rand(*fft.rshape) - 1.
    if inplace:
        dummy = fft.r2c(a, np.empty(fft.cshape, dtype=np.float64))
    else:
        dummy = fft.r2c(a)

    dummy = np.asarray(dummy)
    actual = dummy[:, 0::2] + 1j * dummy[:, 1::2]
    expected = npfft.rfft2(a)

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert_less(error, delta)

@nottest
def do_test_c2r_2d(shape, inplace, delta):

    fft = create_fft(shape)
    expected = 2. * nprnd.rand(*fft.rshape) - 1.
    a = np.asarray(fft.r2c(expected))
    if inplace:
        actual = fft.c2r(a, np.empty(fft.rshape, dtype=np.float64))
    else:
        actual = fft.c2r(a)
    actual = np.asarray(actual)
    expected *= expected.size

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert_less(error, delta)

def test_transform_2d():
    params = [((256, 128), 2E-15),
              ((256, 129), 2E-14),
              ((257, 128), 2E-15),
              ((257, 129), 2E-14)]
    for do_test in [do_test_r2c_2d, do_test_c2r_2d]:
        for shape, delta in params:
            for inplace in [True, False]:
                yield do_test_c2r_2d, shape, inplace, delta

@nottest
@raises(ValueError)
def do_test_r2c_invalid_params(shape, rshape, cshape):
    r = np.empty(rshape, dtype = np.float64)
    c = np.empty(cshape, dtype = np.float64)
    create_fft(shape).r2c(r, c)

@nottest
@raises(ValueError)
def do_test_c2r_invalid_params(shape, rshape, cshape):
    c = np.empty(cshape, dtype = np.float64)
    r = np.empty(rshape, dtype = np.float64)
    create_fft(shape).c2r(c, r)

def test_transform_invalid_params():
    params = [((128, 256), (127, 256), (128, 258)),
              ((128, 256), (128, 255), (128, 258)),
              ((128, 256), (128, 256), (129, 258)),
              ((128, 256), (128, 256), (128, 259)),]
    for do_test in [do_test_r2c_invalid_params,
                    do_test_c2r_invalid_params]:
        for shape, rshape, cshape in params:
            yield do_test, shape, rshape, cshape

if __name__ == '__main__':
    do_test_c2r_2d((256, 128), False, 2E-15)
