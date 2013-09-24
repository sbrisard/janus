from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

include_dirs = [numpy.get_include()]

ext_modules = [Extension('matprop',
                         sources=['matprop.pxd', 'matprop.pyx'],
                         include_dirs=include_dirs),
               Extension('greenop',
                         sources=['greenop.pxd', 'greenop.pyx'],
                         include_dirs=include_dirs),
               Extension('discretegreenop',
                         sources=['discretegreenop.pyx'],
                         include_dirs=include_dirs)]
    
setup(name = 'Homogenization through FFT',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)
