import numpy
import os

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_var
from distutils.util import get_platform

from Cython.Distutils import build_ext

include_dirs = [numpy.get_include(),
                'C:\\opt\\Microsoft_HPC_Pack_2012\\Inc',
                '/opt/local/include']

library_dirs = ['C:\\opt\\Microsoft_HPC_Pack_2012\\Lib\\i386']

try:
    import mpi4py
    include_dirs.append(mpi4py.get_include())
except ImportError:
    pass

extensions = []
extensions.append(Extension('utils.checkarray',
                            sources=['src/utils/checkarray.pyx',
                                     'src/utils/checkarray.pxd']))

extensions.append(Extension('interfaces',
                            sources=['src/interfaces.pyx',
                                     'src/interfaces.pxd']))

extensions.append(Extension('matprop',
                            sources=['src/matprop.pyx',
                                     'src/matprop.pxd']))

extensions.append(Extension('greenop',
                            sources=['src/greenop.pyx',
                                     'src/greenop.pxd']))

extensions.append(Extension('discretegreenop',
                            sources=['src/discretegreenop.pyx']))

extensions.append(Extension('fft.serial._serial_fft',
                            sources=['src/fft/serial/_serial_fft.pyx',
                                     'src/fft/serial/_serial_fft.pxd',
                                     'src/fft/serial/fftw.pxd'],
                            libraries=['fftw3'],
                            library_dirs=library_dirs,
                            include_dirs=include_dirs))

extensions.append(Extension('tensor',
                            sources=['src/tensor.pyx',
                                     'src/tensor.pxd']))

extensions.append(Extension('local_operator',
                            sources=['src/local_operator.pyx']))

if not(get_platform() in ('win32', 'win-amd64')):
    # TODO improve this uggly hack
    gcc = 'gcc'
    mpicc = '/usr/bin/mpicc'
    os.environ['CC'] = get_config_var('CC').replace(gcc, mpicc)
    os.environ['LDSHARED'] = get_config_var('LDSHARED').replace(gcc, mpicc)

    extensions.append(
               Extension('fft.parallel._parallel_fft',
                         sources=['src/fft/parallel/_parallel_fft.pyx',
                                  'src/fft/parallel/_parallel_fft.pxd',
                                  'src/fft/parallel/fftw_mpi.pxd'],
                         libraries=['fftw3', 'fftw3_mpi'],
                         include_dirs = [mpi4py.get_include(),
                                         '/opt/local/include'],
))

setup(name = 'Homogenization through FFT',
      packages=[''],
      package_dir = {'': 'src'},
      cmdclass = {'build_ext': build_ext},
      ext_modules = extensions)
