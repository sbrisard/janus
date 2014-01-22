import numpy
import os

from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_config_var
from distutils.util import get_platform

include_dirs = [numpy.get_include(),
                'C:\\opt\\Microsoft_HPC_Pack_2012\\Inc',
                '/opt/local/include']

library_dirs = ['C:\\opt\\Microsoft_HPC_Pack_2012\\Lib\\i386']

try:
    import mpi4py
    include_dirs.append(mpi4py.get_include())
except ImportError:
    pass

checkarray = Extension('checkarray',
                       sources=['src/checkarray.c'])

interfaces = Extension('interfaces', sources=['src/interfaces.c'])

matprop = Extension('matprop',
                    sources=['src/matprop.c'])

greenop = Extension('greenop',
                    sources=['src/greenop.c'])

discretegreenop = Extension('discretegreenop',
                            sources=['src/discretegreenop.c'])

fft_serial = Extension('fft.serial._serial_fft',
                       sources=['src/fft/serial/_serial_fft.c'],
                       libraries=['fftw3'],
                       library_dirs=library_dirs,
                       include_dirs=include_dirs)

tensor = Extension('tensor',
                   sources=['src/tensor.c'])

local_operator = Extension('local_operator',
                           sources=['src/local_operator.c'])

ext_modules = [checkarray,
               interfaces,
               matprop,
               greenop,
               discretegreenop,
               fft_serial,
               tensor,
               local_operator]


if not(get_platform() in ('win32', 'win-amd64')):
    # TODO improve this uggly hack
    gcc = 'gcc'
    mpicc = '/usr/bin/mpicc'
    os.environ['CC'] = get_config_var('CC').replace(gcc, mpicc)
    os.environ['LDSHARED'] = get_config_var('LDSHARED').replace(gcc, mpicc)

    ext_modules.append(
               Extension('fft.parallel._parallel_fft',
                         sources=['src/fft/parallel/_parallel_fft.c'],
                         libraries=['fftw3', 'fftw3_mpi'],
                         include_dirs = [mpi4py.get_include(),
                                         '/opt/local/include'],
))

setup(name = 'Homogenization through FFT',
      packages=[''],
      package_dir = {'': 'src'},
      ext_modules = ext_modules)
