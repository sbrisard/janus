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

ext_modules = [Extension('matprop',
                         sources=['matprop.pxd', 'matprop.pyx'],
                         include_dirs=include_dirs),
               Extension('greenop',
                         sources=['greenop.pxd', 'greenop.pyx'],
                         include_dirs=include_dirs),
               Extension('discretegreenop',
                         sources=['discretegreenop.pyx'],
                         include_dirs=include_dirs),
               Extension('serialfft',
                         sources=['fftw.pxd', 'serialfft.pyx'],
                         library_dirs=library_dirs,
                         include_dirs=include_dirs,
                         libraries=['fftw3'],),
               ]

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

setup(name = 'Homogenization through FFT',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
