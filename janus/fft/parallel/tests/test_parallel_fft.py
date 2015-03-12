import numpy as np
import pytest

import janus.fft.serial
import janus.fft.parallel

from mpi4py import MPI


# TODO Scatterv fails with arrays of unequal size
# (which happens with n0 == 31)
@pytest.mark.parametrize('shape', [#(31, 15),
                                   #(31, 16),
                                   (32, 15),
                                   (32, 16)])
def test_r2c(shape):
    comm = MPI.COMM_WORLD
    root = 0

    janus.fft.parallel.init()
    pfft = janus.fft.parallel.create_real(shape, comm)
    if comm.rank == root:
        np.random.seed(20150312)
        rglob = 2. * np.random.rand(*shape) - 1.
    else:
        rglob = None
    rloc = np.empty(pfft.rshape, dtype=np.float64)
    comm.Scatterv(rglob, rloc, root)
    cloc = np.empty(pfft.cshape, dtype=np.float64)
    pfft.r2c(rloc, cloc)
    if comm.rank == root:
        # TODO See Issue #7
        actual = np.empty((pfft.shape[0],) + pfft.cshape[1:],
                          dtype=np.float64)
    else:
        actual = None
    comm.Gatherv(cloc, actual, root)

    if comm.rank == root:
        sfft = janus.fft.serial.create_real(shape)
        expected = sfft.r2c(rglob)
        norm = np.sqrt(np.sum((actual - expected)**2))
        assert norm == 0.
