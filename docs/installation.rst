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

Edit the section ``[build_ext]`` of the file ``setup.cfg``. You must provide the following values::

  [build_ext]
  include_dirs = ...
  library_dirs = ...
  libraries = ...

.. data:: include_dirs

   The path to the FFTW headers.

.. data:: library_dirs

   The path to the FFTW shared library.

.. data:: libraries

   The name of the FFTW libraries (comma separated), including the MPI-enabled version if necessary.

Compilation and installation under Unix-like systems (Linux, OSX)
=================================================================

Set the following values::

  [myconfig]
  include_dirs = /path/to/headers
  library_dirs = /path/to/binaries
  libraries = fftw3, fftw3_mpi

Then, issue the standard commands in a console::

  python setup.py install --user

Compilation and installation under Windows
==========================================

The parallel version of this code is not tested under Windows. You must first download and install the `precompiled binaries of FFTW for Windows`_.

Compilation with the Windows SDK 7.1
------------------------------------

Set the following values::

  [build_ext]
  include_dirs = C:\PATH\TO\HEADERS
  library_dirs = C:\PATH\TO\BINARIES
  libraries = libfftw3-3

Then open the *Windows SDK 7.1 Command Prompt*, and issue the following command::

  set DISTUTILS_USE_SDK=1
  setenv /x64 /release

Change to the root directory of the Janus project, and issue the standard commands::

  python setup.py install --user

Compilation with MinGW/MSYS
---------------------------

Set the following values::

  [build_ext]
  include_dirs = C:\PATH\TO\HEADERS
  library_dirs = C:\PATH\TO\BINARIES
  libraries = fftw3-3

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
