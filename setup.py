# -*- coding: utf-8 -*-

import os
import sys

import numpy

from argparse import ArgumentParser
from configparser import ConfigParser
from configparser import NoOptionError

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_var
from distutils.util import get_platform

from Cython.Distutils import build_ext

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
else:
    fftw3 = 'fftw3'
    fftw3_mpi = 'fftw3_mpi'
    with_mpi = True

extensions = []
extensions.append(Extension('janus.utils.checkarray',
                            sources=['janus/utils/checkarray.pyx',
                                     'janus/utils/checkarray.pxd']))

extensions.append(Extension('janus.operators',
                            sources=['janus/operators.pyx',
                                     'janus/operators.pxd']))

extensions.append(Extension('janus.matprop',
                            sources=['janus/matprop.pyx',
                                     'janus/matprop.pxd']))

extensions.append(Extension('janus.greenop',
                            sources=['janus/greenop.pyx',
                                     'janus/greenop.pxd']))

extensions.append(Extension('janus.discretegreenop',
                            sources=['janus/discretegreenop.pyx'],
                            include_dirs=include_dirs,
                            library_dirs=library_dirs))

extensions.append(Extension('janus.fft.serial._serial_fft',
                            sources=['janus/fft/serial/_serial_fft.pyx',
                                     'janus/fft/serial/_serial_fft.pxd',
                                     'janus/fft/serial/fftw.pxd'],
                            libraries=[fftw3],
                            library_dirs=library_dirs,
                            include_dirs=include_dirs))

if with_mpi:
    import mpi4py
    include_dirs.append(mpi4py.get_include())

    # TODO improve this uggly hack
    gcc = 'gcc'
    mpicc = '/usr/bin/mpicc'
    os.environ['CC'] = get_config_var('CC').replace(gcc, mpicc)
    os.environ['LDSHARED'] = get_config_var('LDSHARED').replace(gcc, mpicc)

    extensions.append(Extension('janus.fft.parallel._parallel_fft',
                                sources=['janus/fft/parallel/_parallel_fft.pyx',
                                         'janus/fft/parallel/_parallel_fft.pxd',
                                         'janus/fft/parallel/fftw_mpi.pxd'],
                                libraries=[fftw3, fftw3_mpi],
                                library_dirs=library_dirs,
                                include_dirs = include_dirs))

setup(name = 'Janus',
      version = '0.1',
      description = 'Galerkin approximation of the Lippmann--Schwinger equation',
      author = 'Sébastien Brisard',
      author_email = 'sebastien.brisard@ifsttar.fr',
      packages=[''],
      package_dir = {'': ''},
      cmdclass = {'build_ext': build_ext},
      ext_modules = extensions)
