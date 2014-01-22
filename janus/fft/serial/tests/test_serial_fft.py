import numpy as np
import numpy.random as nprnd
import numpy.fft as npfft

import janus.fft.serial

from nose.tools import assert_less
from nose.tools import nottest
from nose.tools import raises

@nottest
def do_test_r2c(shape, inplace, delta):
    transform = janus.fft.serial.create_real(shape)
    a = 2. * nprnd.rand(*transform.rshape) - 1.
    if inplace:
        output = transform.r2c(a, np.empty(transform.cshape, dtype=np.float64))
    else:
        output = transform.r2c(a)

    output = np.asarray(output)
    output = np.rollaxis(output, output.ndim - 1, 0)
    actual = output[0::2] + 1j * output[1::2]
    actual = np.rollaxis(actual, 0, actual.ndim)
    expected = npfft.rfftn(a)

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert_less(error, delta)

@nottest
def do_test_c2r(shape, inplace, delta):
    transform = janus.fft.serial.create_real(shape)
    expected = 2. * nprnd.rand(*transform.rshape) - 1.
    a = np.asarray(transform.r2c(expected))
    if inplace:
        actual = transform.c2r(a, np.empty(transform.rshape, dtype=np.float64))
    else:
        actual = transform.c2r(a)
    actual = np.asarray(actual)

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert_less(error, delta)

def test_transform():
    params = [((256, 128), 2E-15),
              ((256, 129), 2E-14),
              ((257, 128), 2E-15),
              ((257, 129), 2E-14),
              ((128, 32, 64), 3E-15),
              ((127, 32, 64), 3E-15),
              ((128, 31, 64), 3.5E-15),
              ((128, 32, 63), 3.5E-15),]
    for do_test in [do_test_r2c, do_test_c2r]:
        for shape, delta in params:
            for inplace in [True, False]:
                yield do_test, shape, inplace, delta

@nottest
@raises(ValueError)
def do_test_r2c_invalid_params(shape, rshape, cshape):
    r = np.empty(rshape, dtype = np.float64)
    c = np.empty(cshape, dtype = np.float64)
    janus.fft.serial.create_real(shape).r2c(r, c)

@nottest
@raises(ValueError)
def do_test_c2r_invalid_params(shape, rshape, cshape):
    c = np.empty(cshape, dtype = np.float64)
    r = np.empty(rshape, dtype = np.float64)
    janus.fft.serial.create_real(shape).c2r(c, r)

def test_transform_invalid_params():
    params = [((128, 256), (127, 256), (128, 258)),
              ((128, 256), (128, 255), (128, 258)),
              ((128, 256), (128, 256), (129, 258)),
              ((128, 256), (128, 256), (128, 259)),
              ((128, 64, 32), (127, 64, 32), (128, 64, 34)),
              ((128, 64, 32), (128, 63, 32), (128, 64, 34)),
              ((128, 64, 32), (128, 64, 31), (128, 64, 34)),
              ((128, 64, 32), (128, 64, 32), (127, 64, 34)),
              ((128, 64, 32), (128, 64, 32), (128, 63, 34)),
              ((128, 64, 32), (128, 64, 32), (128, 64, 33)),]
    for do_test in [do_test_r2c_invalid_params,
                    do_test_c2r_invalid_params]:
        for shape, rshape, cshape in params:
            yield do_test, shape, rshape, cshape
