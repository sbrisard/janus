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
