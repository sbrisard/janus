from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension('matprop', ['matprop.pyx']),
               Extension('greenop', ['greenop.pyx'])]

setup(
  name = 'Homogenization through FFT',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
