.. -*- coding: utf-8 -*-

************
Installation
************

For the moment, no precompiled binaries are available, and Janus must be compiled from sources, using ``setuptools``.

The sources can be retrieved from

.. todo:: Update when moved to Github

Prerequisites
=============

Janus requires Python 3k. The serial version depends on `FFTW`_ (version 3) only, while the parallel (MPI-based) version also requires `mpi4py`_.

Configuration
=============

Edit the file ``janus.cfg`` and add a new section (or modify an existing section) as follows::

  [myconfig]
  fftw3 = fftw3
  fftw3_mpi = fftw3_mpi
  fftw3-include = /opt/fftw-3.3.4/include
  fftw3-library = /opt/fftw-3.3.4/lib
  with-mpi = yes
  mpicc = /usr/bin/mpicc

.. data:: fftw3

   The name of the FFTW library (to be passed to the compiler as a ``-l`` flag).

.. data:: fftw3_mpi

   The name of the parallel (MPI) version of the FFTW library (to be passed to the compiler as a ``-l`` flag).

.. data:: fftw3-include

   The path to the FFTW headers.

.. data:: fftw3-library

   The path to the FFTW shared library.

.. data:: with-mpi

   Whether or not the parallel version of the code should be compiled (``yes/no``). Requires `mpi4py`_.

.. data:: mpicc

   The path to the MPI-enabled C compiler.


Compilation and installation under Unix-like systems (Linux, OSX)
=================================================================

Set the following values::

  [myconfig]
  fftw3 = fftw3
  fftw3_mpi = fftw3_mpi
  fftw3-include = /path/to/headers
  fftw3-library = /path/to/binaries
  with-mpi = yes
  mpicc = /path/to/MPI-enabled/compiler

Then, issue the standard commands in a console::

  python setup.py build_ext --config=myconfig
  python setup.py install --user --config=myconfig

Compilation and installation under Windows
==========================================

The parallel version of this code is not tested under Windows. You must first download and install the `precompiled binaries of FFTW for Windows`_.

Compilation with the Windows SDK 7.1
------------------------------------

Set the following values::

  [myconfig]
  fftw3 = libfftw3-3
  fftw3-include = C:\\PATH\\TO\\HEADERS
  fftw3-library = C:\\PATH\\TO\\BINARIES
  with-mpi = no

Then open the *Windows SDK 7.1 Command Prompt*, and issue the following command::

  set DISTUTILS_USE_SDK=1
  setenv /x64 /release

Change to the root directory of the Janus project, and issue the standard commands::

  python setup.py build_ext --config=myconfig
  python setup.py install --config=myconfig

Compilation with MinGW/MSYS
---------------------------

Set the following values::

  [myconfig]
  fftw3 = fftw3-3
  fftw3-include = C:\\PATH\\TO\\HEADERS
  fftw3-library = C:\\PATH\\TO\\BINARIES
  with-mpi = no

.. todo:: Complete installation procedure with MinGW.

Test your installation
======================

Testing the installation of Janus requires `pytest`_. To run all serial tests, issue the following command at the root of the project::

  python -m pytest tests

To run all parallel tests (assuming you compiled the MPI-enabled version of Janus), issue the following command at the root of the project::

  mpiexec -np 3 pytest tests/parallel

where the total number of processes can be adjusted (an odd number should preferably be used, as it is more likely to reveal bugs).

.. todo:: How to print only messages from root process with pytest?

.. _FFTW: http://www.fftw.org/
.. _mpi4py: https://bitbucket.org/mpi4py/mpi4py/
.. _precompiled binaries of FFTW for Windows: http://www.fftw.org/install/windows.html
.. _pytest: http://pytest.org/
