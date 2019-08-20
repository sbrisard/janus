.. -*- coding: utf-8 -*-

************
Installation
************

For the moment, no precompiled binaries are available, and Janus must be compiled from sources, using ``setuptools``.

The sources can be retrieved from Github, https://github.com/sbrisard/janus.git.

Prerequisites
=============

Janus requires Python 3k. The serial version depends on `FFTW`_ (version 3) only, while the parallel (MPI-based) version also requires `mpi4py`_.

.. todo:: The present version of ``setup.py`` tries to install the parallel version of the code if it detects that ``mpi4py`` is installed. In other words, if ``mpi4py`` is installed, the MPI-enabled version of ``FFTW`` *must* be installed.

Configuration (all platforms)
=============================

Compilation and installation is configured through the ``setup.cfg`` file, which must be created in the root directory of the project if necessary (this file must reside in the same directory as ``setup.py``).

Two sections of this file must be filed: ``[fftw]`` and ``[fftw_mpi]`` (if you are compiling the MPI-enabled version of Janus)::

  [fftw]
  include_dirs = …
  library_dirs = …
  libraries = …
  [fftw_mpi]
  include_dirs = …
  library_dirs = …
  libraries = …

.. data:: include_dirs

   The path to the FFTW headers files (optional).

.. data:: library_dirs

   The path to the FFTW shared libraries (optional).

.. data:: libraries

   The name of the FFTW libraries.

All these entries can be (comma separated) *lists*. Examples are provided below for several platforms.

Compilation and installation under Unix-like systems (Linux, OSX)
=================================================================

Make sure that the MPI and FFTW packages are properly installed, including the ``dev`` packages (that include header files). On Ubuntu platforms, the following packages must be installed::

  sudo apt-get install libopenmpi-dev openmpi-bin libfftw3-bin libfftw3-dev libfftw3-mpi-dev libfftw3-mpi3 petsc-dev cython3 python3-numpy python3-h5py python3-mpi4py python3-petsc4py python3-pytest python3-scipy python3-sphinx

Usually, for linux platforms, it is not necessary to set the ``include_dirs`` and ``library_dirs`` values. Also, the library names must be stripped of the ``lib`` prefix (``libfftw3.so.3.5.7`` → ``fftw3``). On Ubuntu platforms, the ``setup.cfg`` file can be as simple as::

  [fftw]
  libraries = fftw3
  [fftw_mpi]
  libraries = fftw3_mpi

Then, issue the standard commands in a console::

  python3 setup.py install --user

Compilation and installation under MacOS
========================================

Compilation and installation under Windows
==========================================

The parallel version of this code is not tested under Windows. You must first download and install the `precompiled binaries of FFTW for Windows`_. Assuming you have uncompressed this archive in ``C:\opt\fftw-3.3.4-dll64, the ``setup.cfg`` file reads::

  [fftw]
  include_dirs = C:\opt\fftw-3.3.4-dll64
  library_dirs = C:\opt\fftw-3.3.4-dll64
  libraries = libfftw3-3

Note that the library name *must not* be stripped of the ``lib`` prefix (``libfftw3-3.dll`` → ``libfftw3``).

Compilation with Anaconda and Visual Studio
-------------------------------------------

If you use `Anaconda <https://www.anaconda.com/distribution/>`_ (which is highly recommended) then open the Anaconda console and issue the standard command::

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
