import numpy as np
import numpy.random
import numpy.fft
import pytest

import janus.fft.serial


def pytest_generate_tests(metafunc):
    if ((metafunc.function is test_r2c) or
        (metafunc.function is test_c2r)) :
        shape_delta = [((256, 128), 2E-15),
                       ((256, 129), 2E-14),
                       ((257, 128), 2E-15),
                       ((257, 129), 2E-14),
                       ((128, 32, 64), 3E-15),
                       ((127, 32, 64), 3E-15),
                       ((128, 31, 64), 3.5E-15),
                       ((128, 32, 63), 3.5E-15),]
        params = ([x + (True,) for x in shape_delta] +
                  [x + (False,) for x in shape_delta])
        metafunc.parametrize('shape, delta, inplace', params)
    if metafunc.function is test_invalid_params:
        shapes = [((128, 256), (127, 256), (128, 258)),
                  ((128, 256), (128, 255), (128, 258)),
                  ((128, 256), (128, 256), (129, 258)),
                  ((128, 256), (128, 256), (128, 259)),
                  ((128, 64, 32), (127, 64, 32), (128, 64, 34)),
                  ((128, 64, 32), (128, 63, 32), (128, 64, 34)),
                  ((128, 64, 32), (128, 64, 31), (128, 64, 34)),
                  ((128, 64, 32), (128, 64, 32), (127, 64, 34)),
                  ((128, 64, 32), (128, 64, 32), (128, 63, 34)),
                  ((128, 64, 32), (128, 64, 32), (128, 64, 33)),]
        params = ([x + (True,) for x in shapes] +
                  [x + (False,) for x in shapes])
        metafunc.parametrize('shape, rshape, cshape, direct', params)


def test_r2c(shape, inplace, delta):
    transform = janus.fft.serial.create_real(shape)
    a = 2. * numpy.random.rand(*transform.rshape) - 1.
    if inplace:
        output = np.empty(transform.cshape, dtype=np.float64)
    else:
        output = None

    output = np.asarray(transform.r2c(a, output))
    actual = output[..., 0::2] + 1j * output[..., 1::2]
    expected = numpy.fft.rfftn(a)

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert error <= delta


def test_c2r(shape, delta, inplace):
    transform = janus.fft.serial.create_real(shape)
    expected = 2. * numpy.random.rand(*transform.rshape) - 1.
    a = np.asarray(transform.r2c(expected))
    if inplace:
        actual = transform.c2r(a, np.empty(transform.rshape, dtype=np.float64))
    else:
        actual = transform.c2r(a)
    actual = np.asarray(actual)

    error = (np.sum(np.absolute(actual - expected))
             / np.sum(np.absolute(expected)))

    assert error <= delta


def test_invalid_params(shape, rshape, cshape, direct):
    r = np.empty(rshape, dtype = np.float64)
    c = np.empty(cshape, dtype = np.float64)
    with pytest.raises(ValueError):
        if direct:
            janus.fft.serial.create_real(shape).r2c(r, c)
        else:
            janus.fft.serial.create_real(shape).c2r(c, r)
