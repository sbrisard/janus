import mpi4py
import subprocess

subprocess.call(['cython', '_parallel_fft.pyx',
                 '-I' + mpi4py.get_include(),
                 '-I../../'])
