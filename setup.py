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
extensions.append(Extension('janus.utils.checkarray',
                            sources=['janus/utils/checkarray.pyx',
                                     'janus/utils/checkarray.pxd']))

extensions.append(Extension('janus.utils.interfaces',
                            sources=['janus/utils/interfaces.pyx',
                                     'janus/utils/interfaces.pxd']))

extensions.append(Extension('janus.utils.tensors',
                            sources=['janus/utils/tensors.pyx',
                                     'janus/utils/tensors.pxd']))

extensions.append(Extension('janus.matprop',
                            sources=['janus/matprop.pyx',
                                     'janus/matprop.pxd']))

extensions.append(Extension('janus.greenop',
                            sources=['janus/greenop.pyx',
                                     'janus/greenop.pxd']))

extensions.append(Extension('janus.discretegreenop',
                            sources=['janus/discretegreenop.pyx']))

extensions.append(Extension('janus.fft.serial._serial_fft',
                            sources=['janus/fft/serial/_serial_fft.pyx',
                                     'janus/fft/serial/_serial_fft.pxd',
                                     'janus/fft/serial/fftw.pxd'],
                            libraries=['fftw3'],
                            library_dirs=library_dirs,
                            include_dirs=include_dirs))

extensions.append(Extension('janus.local_operator',
                            sources=['janus/local_operator.pyx']))

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
