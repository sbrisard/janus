import numpy
import mpi4py
import os

from distutils.core import setup
from distutils.extension import Extension
from distutils.util import get_platform

include_dirs = [numpy.get_include(),
                mpi4py.get_include(),
                'C:\\opt\\Microsoft_HPC_Pack_2012\\Inc',]

library_dirs = ['C:\\opt\\Microsoft_HPC_Pack_2012\\Lib\\i386']

matprop = Extension('matprop',
                    sources=['src/matprop.c'])

greenop = Extension('greenop',
                    sources=['src/greenop.c'])

discretegreenop = Extension('discretegreenop',
                            sources=['src/discretegreenop.c'])

fft_serial = Extension('fft.serial._serial_fft',
                       sources=['src/fft/serial/_serial_fft.c'],
                       libraries=['fftw3'],)

ext_modules = [matprop, greenop, discretegreenop, fft_serial]

if not(get_platform() in ('win32', 'win-amd64')):
    ext_modules.append(
               Extension('fft.parallel._parallel_fft',
                         sources=['src/fft/parallel/_parallel_fft.c'],
                         libraries=['fftw3', 'fftw3_mpi']
))

setup(name = 'Homogenization through FFT',
      packages=[''],
      package_dir = {'': 'src'},
      ext_modules = ext_modules)
