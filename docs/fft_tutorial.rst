.. -*- coding: utf-8-unix -*-

.. _FFT:

*************************************
Computing discrete Fourier transforms
*************************************

Discrete Fourier transforms are computed through the Fast Fourier Transform method (FFT) implemented in the `FFTW <http://www.fftw.org/>`_ library. Module :mod:`janus.fft` provides a Python wrapper to this C library. This module exposes both serial and parallel (MPI) implementations through a unified interface.

Before the main methods and functions of the :mod:`janus.fft` module are introduced, an important design issue should be mentioned. In the present implementation of the module, input data (to be transformed) is not passed directly to FFTW. Rather, a local copy is first made, and FFTW then operates on this local copy. This allows reusing the same plan to perform many transforms (which is advantageous in the context of iterative solvers). This certainly induces a performance hit, which is deemed negligible for transforms of large 2D or 3D arrays.

.. TODO Confirm above point on performance hit.

Although not essential, it might be useful to have a look to the `FFTW manual <http://www.fftw.org/fftw3_doc/>`_. For the time being, only two and three dimensional real-to-complex transforms are implemented.

Serial computations
===================

The following piece of code creates an object ``transform`` which can perform real FFTs on ``32x64`` grids of real numbers.

>>> import janus.fft.serial
>>> transform = janus.fft.serial.create_real((32, 64))

The function :func:`janus.fft.serial.create_real` can be passed planner flags (see `Planner Flags <http://www.fftw.org/fftw3_doc/Planner-Flags.html#Planner-Flags>`_ in the FFTW manual). The attributes of the returned object are

  - ``transform.shape`` contains the *global* shape of the input array,
  - ``transform.rshape`` contains the *local* shape of the input (real) array. For serial transforms, local and global shapes coincide,
  - ``transform.cshape`` contains the *local* shape of the output (complex) array.

>>> transform.shape
(32, 64)
>>> transform.rshape
(32, 64)
>>> transform.cshape
(32, 66)

It should be noted that complex-valued tables are stored according to the FFTW library: even (resp. odd) values of the fast index correspond to the real (resp. imaginary) part of the complex number (see also `Multi-Dimensional DFTs of Real Data <http://www.fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html#Multi_002dDimensional-DFTs-of-Real-Data>`_ in the FFTW manual).

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

Parallel computations
=====================

The module :mod:`janus.fft.parallel` is a wrapper around the ``fftw3-mpi`` library (refer to `Distributed-memory FFTW with MPI <http://www.fftw.org/fftw3_doc/Distributed_002dmemory-FFTW-with-MPI.html#Distributed_002dmemory-FFTW-with-MPI>`_ in the FFTW manual for the inner workings of this library). This module must be used along with the `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_ module to handle MPI communications.

The Python API is very similar to the API for serial transforms. However, computing a parallel FFT is slightly more involved than computing a serial FFT, because the data must be distributed across the processes. The computation must go through the following steps

  1. create input data (root process),
  2. create a transform object (all processes),
  3. gather local shapes (root process),
  4. scatter the input data according to the previouly gathered local sizes (root process),
  5. compute the transform (all processes),
  6. gather the results (root process).

This is illustrated in the step-by-step tutorial below. This tutorial aims again at computing a ``32x64`` real Fourier transform. The full source can be :download:`downloaded here <./parallel_fft_tutorial.py>`, it must be run through the following command line::

    $ mpiexec -np 2 python3 parallel_fft_tutorial.py

where the number of processes can be adjusted (all output produced below was obtained with two parallel processes).

Before we proceed with the description of the program, it should be noted that communication will be carried out with the uppercase versions `MPI.Comm.Gather` and `MPI.Comm.Scatter`. The lowercase versions of `MPI.Comm.scatter` and `MPI.Comm.gather` are slightly easier to use, but communicate objects through pickling. This approach fails with very large objects (the size limit is much lower than the intrinsic MPI size limit). With `MPI.Comm.Gather` and `MPI.Comm.Scatter`, the intrinsic MPI size limit is restored. The FFT objects defined in the module :mod:`janus.fft.parallel` provide attributes to help call these methods.

A few modules must first be imported

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Imports
  :end-before: Init

Then, some useful variables are created

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Init
  :end-before: Create

Then, the transform objects (one for each process) are created (step 2), and their various shapes are printed out.

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Create
  :end-before: Prepare

This code snippet outputs the following messages

.. code-block:: none

    global_ishape  = (32, 64)
    global_oshape  = (32, 66)
    ishape = (16, 64)
    oshape = (16, 66)

The ``transform.shape`` attribute refers to the *global* (logical) shape of the transform. Since the data is distributed across all processes, the *local* size in memory of the input and output data differ from ``transform.shape``. Accordingly, the ``transform.rshape`` (resp. ``transform.cshape``) attribute refers to the local shape of the real, input (resp. complex, output) data, for the current process. As expected with FFTW, it is observed that the data is distributed with respect to the first dimension. Indeed, the global, first dimension is 64, and the above example is run with 2 processes; therefore, the local first dimension is ``64 / 2 = 32``.

In order to figure out how to scatter the input data, the root process then gathers all local sizes and displacements, and the parameters to be passed to ``mpi4py.MPI.Scatterv()`` and ``mpi4py.MPI.Gatherv()`` are prepared

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Prepare
  :end-before: Scatter

Then the input data ``x`` is scattered across all processes

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Scatter
  :end-before: Execute

Each process then executes its transform

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Execute
  :end-before: Gather

and the root process finally gathers the results

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Gather
  :end-before: Validate

To check that the computation is correct, the same transform is finally computed locally by the root process

.. literalinclude:: parallel_fft_tutorial.py
  :start-after: Validate


The complete program
--------------------

.. literalinclude:: parallel_fft_tutorial.py
