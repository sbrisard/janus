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
PARAMS = list(itertools.product(SHAPES, [False, True]))

@pytest.mark.parametrize('global_ishape, inverse', PARAMS)
def test_transform(global_ishape, inverse):
    comm = MPI.COMM_WORLD
    root = 0

    pfft = janus.fft.parallel.create_real(global_ishape, comm)
    counts_and_displs = comm.gather(sendobj=(pfft.isize, pfft.idispl,
                                             pfft.osize, pfft.odispl),
                                    root=root)
    global_oshape = pfft.global_oshape
    if comm.rank == root:
        icounts, idispls, ocounts, odispls = zip(*counts_and_displs)
        if inverse:
            xshape, xcounts, xdispls = global_oshape, ocounts, odispls
            yshape, ycounts, ydispls = global_ishape, icounts, idispls
        else:
            xshape, xcounts, xdispls = global_ishape, icounts, idispls
            yshape, ycounts, ydispls = global_oshape, ocounts, odispls
        np.random.seed(20150312)
        x = 2. * np.random.rand(*xshape) - 1.
        y = np.empty(yshape, dtype=np.float64)
    else:
        x, xcounts, xdispls = None, None, None
        y, ycounts, ydispls = None, None, None

    if inverse:
        xshape, yshape, transform = pfft.oshape, pfft.ishape, pfft.c2r
    else:
        xshape, yshape, transform = pfft.ishape, pfft.oshape, pfft.r2c

    xloc = np.empty(xshape, dtype=np.float64)
    yloc = np.empty(yshape, dtype=np.float64)
    comm.Scatterv([x, xcounts, xdispls, MPI.DOUBLE], xloc, root)
    transform(xloc, yloc)
    comm.Gatherv(yloc, [y, ycounts, ydispls, MPI.DOUBLE], root)

    if comm.rank == root:
        sfft = janus.fft.serial.create_real(global_ishape)
        yref = sfft.c2r(x) if inverse else sfft.r2c(x)
        yref = np.asarray(yref)
        norm_err = np.sqrt(np.sum((y - yref)**2))
        norm_ref = np.sqrt(np.sum(yref**2))
        assert norm_err <= ULP * norm_ref
