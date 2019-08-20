from .fftw_mpi cimport *
from janus.fft.serial._serial_fft cimport _RealFFT2D
from janus.fft.serial._serial_fft cimport _RealFFT3D

from mpi4py import MPI

from janus.fft.serial._serial_fft import FFTW_MEASURE

cpdef init():
    fftw_mpi_init()

def create_real(shape, comm=MPI.COMM_WORLD, flags=FFTW_MEASURE):
    """Return a new FFT object to compute real-to-complex transforms.

    Args:
        shape (tuple): the global shape of the input data.
        comm (mpi4py.MPI.Comm): the MPI communicator.
        flags (int): "or" combination of :ref:`planner-flags`.
    """
    if len(shape) == 2:
        return create_real_2D(shape[0], shape[1], comm, flags)
    elif len(shape) == 3:
        return create_real_3D(shape[0], shape[1], shape[2], comm, flags)
    else:
        msg = 'length of shape can be 2 or 3 (was {0})'
        raise ValueError(msg.format(len(shape)))

cdef create_real_2D(ptrdiff_t n0, ptrdiff_t n1, Comm comm, unsigned flags):
    cdef ptrdiff_t n0_loc, offset0
    cdef ptrdiff_t size = 2 * fftw_mpi_local_size_2d(n0, n1 // 2 + 1,
                                                     comm.ob_mpi,
                                                     &n0_loc,
                                                     &offset0)
    cdef _RealFFT2D fft = _RealFFT2D(n0, n1, n0_loc, offset0)
    fft.buffer = fftw_alloc_real(size)
    fft.plan_r2c = fftw_mpi_plan_dft_r2c_2d(n0, n1,
                                            <double *> fft.buffer,
                                            <fftw_complex *> fft.buffer,
                                            comm.ob_mpi, flags)
    fft.plan_c2r = fftw_mpi_plan_dft_c2r_2d(n0, n1,
                                            <fftw_complex *> fft.buffer,
                                            <double *> fft.buffer,
                                            comm.ob_mpi, flags)
    return fft

cdef create_real_3D(ptrdiff_t n0, ptrdiff_t n1, ptrdiff_t n2, Comm comm,
                    unsigned flags):
    cdef ptrdiff_t n0_loc, offset0
    cdef ptrdiff_t size = 2 * fftw_mpi_local_size_3d(n0, n1, n2 // 2 + 1,
                                                     comm.ob_mpi,
                                                     &n0_loc,
                                                     &offset0)
    cdef _RealFFT3D fft = _RealFFT3D(n0, n1, n2, n0_loc, offset0)
    fft.buffer = fftw_alloc_real(size)
    fft.plan_r2c = fftw_mpi_plan_dft_r2c_3d(n0, n1, n2,
                                            <double *> fft.buffer,
                                            <fftw_complex *> fft.buffer,
                                            comm.ob_mpi, flags)
    fft.plan_c2r = fftw_mpi_plan_dft_c2r_3d(n0, n1, n2,
                                            <fftw_complex *> fft.buffer,
                                            <double *> fft.buffer,
                                            comm.ob_mpi, flags)
    return fft
