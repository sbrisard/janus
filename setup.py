# -*- coding: utf-8 -*-

import os
import sys
import setuptools

import numpy

from argparse import ArgumentParser
from configparser import ConfigParser
from configparser import NoOptionError

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_var
from distutils.util import get_platform

from Cython.Build import cythonize

include_dirs = [numpy.get_include()]
library_dirs = []

parser = ArgumentParser(add_help=False)
parser.add_argument('--config')
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
config = args.config

parser = ConfigParser()
parser.read('janus.cfg')
if config is not None:
    fftw3 = parser.get(config, 'fftw3')
    include_dirs.append(parser.get(config, 'fftw3-include'))
    library_dirs.append(parser.get(config, 'fftw3-library'))
    with_mpi = parser.getboolean(config, 'with-mpi')
    if with_mpi:
        fftw3_mpi = parser.get(config, 'fftw3_mpi')
        mpicc = parser.get(config, 'mpicc')
else:
    fftw3 = 'fftw3'
    fftw3_mpi = 'fftw3_mpi'
    with_mpi = True
    mpicc = '/usr/bin/mpicc'

extensions = []

extensions.append(Extension('janus.utils.checkarray',
                            sources=['janus/utils/checkarray.pyx']))

extensions.append(Extension('janus.operators',
                            sources=['janus/operators.pyx']))

extensions.append(Extension('janus.matprop',
                            sources=['janus/matprop.pyx']))

extensions.append(Extension('janus.greenop',
                            sources=['janus/greenop.pyx']))

extensions.append(Extension('janus.discretegreenop',
                            sources=['janus/discretegreenop.pyx'],
                            include_dirs=include_dirs,
                            library_dirs=library_dirs))

# TODO This module also depends on fftw.pxd
extensions.append(Extension('janus.fft.serial._serial_fft',
                            sources=['janus/fft/serial/_serial_fft.pyx'],
                            libraries=[fftw3],
                            library_dirs=library_dirs,
                            include_dirs=include_dirs))

if with_mpi:
    import subprocess

    import mpi4py
    include_dirs.append(mpi4py.get_include())

    showme_compile = subprocess.check_output([mpicc, '--showme:compile']).decode('ascii')
    showme_link = subprocess.check_output([mpicc, '--showme:link']).decode('ascii')

    # TODO This module also depends on fftw_mpi.pxd
    extensions.append(Extension('janus.fft.parallel._parallel_fft',
                                sources=['janus/fft/parallel/_parallel_fft.pyx'],
                                libraries=[fftw3, fftw3_mpi],
                                library_dirs=library_dirs,
                                include_dirs=include_dirs,
                                extra_compile_args=showme_compile.split(),
                                extra_link_args=showme_link.split()))

packages = ['janus',
            'janus.fft',
            'janus.fft.serial',
            'janus.utils']

if with_mpi:
    packages.append('janus.fft.parallel')

setup(name = 'Janus',
      version = '0.1',
      description = 'Galerkin approximation of the Lippmann--Schwinger equation',
      author = 'Sébastien Brisard',
      author_email = 'sebastien.brisard@ifsttar.fr',
      packages=packages,
      ext_modules = cythonize(extensions))
