import numpy
import mpi4py
import os

from distutils.core import setup
from distutils.extension import Extension
from distutils.util import get_platform
from Cython.Distutils import build_ext

include_dirs = [numpy.get_include(),
                mpi4py.get_include(),
                'C:\\opt\\Microsoft_HPC_Pack_2012\\Inc',]

library_dirs = ['C:\\opt\\Microsoft_HPC_Pack_2012\\Lib\\i386']

matprop = Extension('matprop',
                    sources=['matprop.c'])

greenop = Extension('greenop',
                    sources=['greenop.c'])

discretegreenop = Extension('discretegreenop',
                            sources=['discretegreenop.c'])

fft_serial = Extension('fft.serial._serial_fft',
                       sources=['fft/serial/_serial_fft.c'],
                       libraries=['fftw3'],)

ext_modules = [matprop, greenop, discretegreenop, fft_serial]

"""
if not(get_platform() in ('win32', 'win-amd64')):
    ext_modules.append(
               Extension('parallelfft',
                         sources=['fftw.pxd',
                                  'serialfft.pyx',
                                  'parallelfft.pyx'],
                         library_dirs=library_dirs,
                         include_dirs=include_dirs,
                         libraries=['fftw3', 'fftw3_mpi']
))
"""

setup(name = 'Homogenization through FFT',
      ext_modules = ext_modules)
