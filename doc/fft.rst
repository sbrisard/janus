.. -*- coding: utf-8-unix -*-

*************************************
Computing discrete Fourier transforms
*************************************

Discrete Fourier transforms are computed through the Fast Fourier Transform method (FFT) implemented in the `FFTW <http://www.fftw.org/>`_ library. Module :mod:`janus.fft` provides a Python wrapper to this C library. This module exposes both serial and parallel (MPI) implementations through a unified interface.

Before the main methods and functions of the :mod:`janus.fft` module are introduced, an important design issue should be mentioned. In the present implementation of the module, input data (to be transformed) is not passed directly to FFTW. Rather, a local copy is first made, and FFTW then operates on this local copy. This allows reusing the same plan to perform many transforms (which is advantageous in the context of iterative solvers). This certainly induces a performance hit, which is deemed negligible for transforms of large 2D or 3D arrays.

.. TODO Confirm above point on performance hit.

Although not essential, it might be useful to have a look to the FFTW `manual <http://www.fftw.org/fftw3_doc/>`_.

Allocation of FFT objects
=========================

For the time being, only two and three dimensional real-to-complex transforms are implemented.

Serial transforms
-----------------

The following piece of code creates an object ``transform`` which can perform real FFTs on ``32x64`` grids of real numbers.

>>> import janus.fft.serial
>>> transform = janus.fft.serial.create_real((32, 64))

The attributes of the returned object are

  - ``transform.shape`` contains the *global* shape of the input array,
  - ``transform.rshape`` contains the *local* shape of the input (real) array. For serial transforms, local and global shapes coincide,
  - ``transform.cshape`` contains the *local* shape of the output (complex) array.

>>> transform.shape
(32, 64)
>>> transform.rshape
(32, 64)
>>> transform.cshape
(32, 66)

It should be noted that complex-valued tables are implemented according to the FFTW library: even (resp. odd) values of the fast index correspond to the real (resp. imaginary) part of the complex number (see also the FFTW `manual <http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data>`_ ).

Direct (real-to-complex) transforms are computed through the method ``transform.r2c()``, which takes as input a ``MemoryView`` of shape ``transform.rshape``, and returns a ``MemoryView`` of shape ``transform.cshape``.

>>> import numpy as np
>>> np.random.seed(20150223)
>>> x = np.random.rand(*transform.rshape)
>>> y1 = transform.r2c(x)

It should be noted that ``y1`` is a ``MemoryView``, not a ``numpy`` array; it can, however, readily be converted into an array

>>> print(y1)
<MemoryView of 'array' object>
>>> y1 = np.asarray(y1)
>>> type(y1)
<class 'numpy.ndarray'>

The output can be converted to an array of complex numbers

>>> actual = y1[..., 0::2] + 1j * y1[..., 1::2]
>>> actual.shape
(32, 33)

and compared to the FFT of ``x`` computed by means of the ``numpy.fft`` module

>>> expected = np.fft.rfftn(x)
>>> expected.shape
(32, 33)
>>> abs_delta = np.absolute(expected - actual)
>>> abs_exp = np.absolute(expected)
>>> error = np.sqrt(np.sum(abs_delta**2) / np.sum(abs_exp**2))
>>> assert error < 1E-15

Inverse discrete Fourier transform is computed through the method ``transform.c2r()``

>>> x1 = transform.c2r(y1)
>>> error = np.sqrt(np.sum((x1 - x)**2) / np.sum(x**2))
>>> assert error < 1E-15

It should be noted that the output array can be passed as an argument to both ``transform.r2c()``

>>> y2 = np.empty(transform.cshape)
>>> out = transform.r2c(x, y2)
>>> assert out.base is y2
>>> assert np.sum((y2 - y1)**2) == 0.0

and ``transform.c2r()``

>>> x2 = np.empty(transform.rshape)
>>> out = transform.c2r(y1, x2)
>>> assert out.base is x2
>>> assert np.sum((x2 - x1)**2) == 0.0
