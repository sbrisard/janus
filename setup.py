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

FFTW3_DEFAULT = 'fftw3'
FFTW3_MPI_DEFAULT = 'fftw3-mpi'

include_dirs = [numpy.get_include()]
library_dirs = []

parser = ArgumentParser(add_help=False)
parser.add_argument('--config', help='', required=False)
args, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown
config = args.config

parser = ConfigParser()

if (config is not None) and (config in parser):
    fftw3 = parser.get(config, 'fftw3',
                       fallback=FFTW3_DEFAULT)
    fftw3_mpi = parser.get(config, 'fftw3-mpi',
                           fallback=FFTW3_MPI_DEFAULT)

    try:
        include_dirs.append(parser.get(config, 'fftw3_include'))
    except NoOptionError:
        pass

    try:
        library_dirs.append(parser.get(config, 'fftw3_library'))
    except NoOptionError:
        pass
else:
    fftw3 = FFTW3_DEFAULT
    fftw3_mpi = FFTW3_MPI_DEFAULT

# TODO Test for Mac platform and add this path '/opt/local/include'

try:
    import mpi4py
    include_dirs.append(mpi4py.get_include())
except ImportError:
    pass

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

if not(get_platform() in ('win32', 'win-amd64')):
    # TODO improve this uggly hack
    gcc = 'gcc'
    mpicc = '/usr/bin/mpicc'
    os.environ['CC'] = get_config_var('CC').replace(gcc, mpicc)
    os.environ['LDSHARED'] = get_config_var('LDSHARED').replace(gcc, mpicc)

    extensions.append(
               Extension('janus.fft.parallel._parallel_fft',
                         sources=['janus/fft/parallel/_parallel_fft.pyx',
                                  'janus/fft/parallel/_parallel_fft.pxd',
                                  'janus/fft/parallel/fftw_mpi.pxd'],
                         libraries=['fftw3', 'fftw3_mpi'],
                         include_dirs = [mpi4py.get_include(),
                                         '/opt/local/include'],
))

setup(name = 'Homogenization through FFT',
      packages=[''],
      package_dir = {'': ''},
      cmdclass = {'build_ext': build_ext},
      ext_modules = extensions)
