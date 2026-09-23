.. -*- coding: utf-8 -*-

************
Installation
************

For the moment, no precompiled binaries are available, and Janus must be compiled from sources, using ``setuptools``.

The sources can be retrieved from Github, https://github.com/sbrisard/janus.git.

Prerequisites
=============

Janus requires Python 3k, and depends on `FFTW`_ (version 3) only.

Configuration (all platforms)
=============================

Compilation and installation is configured through the optional ``setup.cfg`` file, which must be created in the root directory of the project if necessary (this file must reside in the same directory as ``setup.py``).

If ``setup.cfg`` does not exist, or provides no entry in its ``[fftw]`` section, defaults suitable for a conda environment are used: ``libraries = fftw3`` on all platforms and, on Windows, the ``Library\include`` and ``Library\lib`` subdirectories of the environment, where conda installs FFTW. Within the ``janus`` conda environment (see ``environment.yml``), ``setup.cfg`` is therefore not needed. It must be created in all other cases (e.g. FFTW installed by the system package manager in a non-standard location, or precompiled binaries downloaded from fftw.org).

The FFTW settings are given in the ``[fftw]`` section::

  [fftw]
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

Compilation and installation under Linux
========================================

Make sure that the FFTW packages are properly installed, including the ``dev`` packages (that include header files). On Ubuntu platforms, the following packages must be installed::

  sudo apt-get install libfftw3-bin libfftw3-dev cython3 python3-numpy python3-h5py python3-pytest python3-scipy python3-sphinx

Usually, for linux platforms, it is not necessary to set the ``include_dirs`` and ``library_dirs`` values. Also, the library names must be stripped of the ``lib`` prefix (``libfftw3.so.3.5.7`` → ``fftw3``). On Ubuntu platforms, the ``setup.cfg`` file can be as simple as::

  [fftw]
  libraries = fftw3

Then, issue the standard commands in a console::

  python3 setup.py install --user

Compilation and installation under MacOS
========================================

Compilation and installation under Windows
==========================================

Compilation with Miniconda and Visual Studio (recommended)
----------------------------------------------------------

This procedure was tested with Miniconda, Python 3.14 and Visual Studio Build Tools 2026.

1. Install the `Visual Studio Build Tools`_, with the *Desktop development with C++* workload. The compiler is located automatically by ``setuptools``: there is no need to use a *Developer Command Prompt*.

2. From the root of the project, create and activate the ``janus`` environment. It provides all the dependencies of Janus (including FFTW), as well as the packages that are required to run the tests and build the documentation::

     conda env create -f environment.yml
     conda activate janus

   To synchronize an existing environment with ``environment.yml``, use ``conda env update -n janus -f environment.yml`` instead.

3. There is no need to create ``setup.cfg``: FFTW is installed in the ``Library`` subdirectory of the environment, which ``setup.py`` uses by default. Should the paths need to be set explicitly, note that environment variables are not expanded in ``setup.cfg``: the path printed by ``echo %CONDA_PREFIX%`` must be written in full::

     [fftw]
     include_dirs = C:\path\to\miniconda3\envs\janus\Library\include
     library_dirs = C:\path\to\miniconda3\envs\janus\Library\lib
     libraries = fftw3

   Note that, unlike the precompiled binaries downloaded from fftw.org, the library provided by conda is called ``fftw3`` (without the ``lib`` prefix).

4. Install Janus in development (editable) mode::

     pip install --no-build-isolation -e .

   The ``--no-build-isolation`` flag ensures that the versions of Cython and setuptools installed in the environment are used for the build (otherwise, pip downloads the latest versions from PyPI).

   After modifying a ``*.pyx`` or ``*.pxd`` file, recompile the extension modules in place with ``python setup.py build_ext --inplace``.

.. _Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

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

Testing the installation of Janus requires `pytest`_. To run all tests, issue the following command at the root of the project::

  python -m pytest tests

Build the documentation
=======================

The documentation is written with `Sphinx`_, and its sources are located in the ``sphinx/`` directory. Since the API reference is extracted from the docstrings of the compiled modules, Janus must be compiled and installed first.

The HTML version of the documentation is published on `GitHub Pages <https://sbrisard.github.io/janus/>`_, from the ``docs/`` directory of the ``master`` branch. To update it, issue the following commands at the root of the project::

  python scripts/empty_docs.py
  python -m sphinx -b html -E -d sphinx/_build/doctrees sphinx docs

The first command empties the ``docs/`` directory, except the ``docs/.nojekyll`` file (which tells GitHub Pages not to process the site with Jekyll), so that the files that Sphinx no longer produces are removed. The second command builds the HTML documentation from scratch into ``docs/``. Check the result (open ``docs/index.html`` in a browser), then commit the ``docs/`` directory and push it to the ``master`` branch: GitHub Pages then redeploys the site automatically.

.. _FFTW: http://www.fftw.org/
.. _pytest: http://pytest.org/
.. _Sphinx: https://www.sphinx-doc.org/
