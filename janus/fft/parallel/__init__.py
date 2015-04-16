"""This module is a Python wrapper around the MPI version of FFTW.

.. autofunction:: janus.fft.parallel.create_real

"""

from ._parallel_fft import create_real
from ._parallel_fft import init

init()
