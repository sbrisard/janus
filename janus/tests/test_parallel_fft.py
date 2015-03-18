import itertools

import numpy as np
import pytest

import janus.fft.serial
import janus.fft.parallel

from mpi4py import MPI

ULP = np.finfo(np.float64).eps
SIZES = [7, 8, 9, 15, 16, 17, 31, 32, 33]
SHAPES_2D = list(itertools.product(SIZES, SIZES))
SHAPES_3D = list(itertools.product(SIZES, SIZES, SIZES))
SHAPES = SHAPES_2D + SHAPES_3D

@pytest.mark.parametrize('shape', SHAPES)
def test_r2c(shape):
    comm = MPI.COMM_WORLD
    root = 0

    janus.fft.parallel.init()
    pfft = janus.fft.parallel.create_real(shape, comm)
    counts_and_displs = comm.gather(sendobj=(pfft.isize, pfft.idispl,
                                             pfft.osize, pfft.odispl),
                                    root=root)
    if comm.rank == root:
        np.random.seed(20150312)
        rglob = 2. * np.random.rand(*shape) - 1.
        icounts, idispls, ocounts, odispls = zip(*counts_and_displs)
    else:
        rglob = None
        icounts, idispls, ocounts, odispls = None, None, None, None
    rloc = np.empty(pfft.rshape, dtype=np.float64)
    comm.Scatterv([rglob, icounts, idispls, MPI.DOUBLE], rloc, root)
    cloc = np.empty(pfft.cshape, dtype=np.float64)
    pfft.r2c(rloc, cloc)
    if comm.rank == root:
        # TODO See Issue #7
        actual = np.empty((pfft.shape[0],) + pfft.cshape[1:],
                          dtype=np.float64)
    else:
        actual = None
    comm.Gatherv(cloc, [actual, ocounts, odispls, MPI.DOUBLE], root)

    if comm.rank == root:
        sfft = janus.fft.serial.create_real(shape)
        expected = np.asarray(sfft.r2c(rglob))
        norm_err = np.sqrt(np.sum((actual - expected)**2))
        norm_ref = np.sqrt(np.sum(expected**2))
        assert norm_err <= ULP * norm_ref

@pytest.mark.parametrize('shape', SHAPES)
def test_c2r(shape):
    comm = MPI.COMM_WORLD
    root = 0

    janus.fft.parallel.init()
    pfft = janus.fft.parallel.create_real(shape, comm)
    counts_and_displs = comm.gather(sendobj=(pfft.isize, pfft.idispl,
                                             pfft.osize, pfft.odispl),
                                    root=root)
    if comm.rank == root:
        np.random.seed(20150312)
        # TODO See Issue #7
        oshape = (pfft.shape[0],) + pfft.cshape[1:]
        cglob = 2. * np.random.rand(*oshape) - 1.
        icounts, idispls, ocounts, odispls = zip(*counts_and_displs)
    else:
        cglob = None
        icounts, idispls, ocounts, odispls = None, None, None, None
    rloc = np.empty(pfft.rshape, dtype=np.float64)
    cloc = np.empty(pfft.cshape, dtype=np.float64)
    comm.Scatterv([cglob, ocounts, odispls, MPI.DOUBLE], cloc, root)
    pfft.c2r(cloc, rloc)
    if comm.rank == root:
        # TODO See Issue #7
        actual = np.empty(shape, dtype=np.float64)
    else:
        actual = None
    comm.Gatherv(rloc, [actual, icounts, idispls, MPI.DOUBLE], root)

    if comm.rank == root:
        sfft = janus.fft.serial.create_real(shape)
        expected = np.asarray(sfft.c2r(cglob))
        norm_err = np.sqrt(np.sum((actual - expected)**2))
        norm_ref = np.sqrt(np.sum(expected**2))
        assert norm_err <= ULP * norm_ref
