import os.path

import numpy as np

import janus.discretegreenop
import janus.fft.parallel
import janus.greenop as greenop

from mpi4py import MPI
from nose.tools import nottest
from nose.tools import raises
from numpy.testing import assert_array_almost_equal_nulp

from janus.matprop import IsotropicLinearElasticMaterial as Material

@nottest
def do_test_apply(path_to_ref, rel_err):
    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank

    # Load reference data
    n = None
    expected = None
    if rank == root:
        npz_file = np.load(path_to_ref)
        x = npz_file['x']
        expected = npz_file['y']
        n = x.shape[:-1]

    # Broadcast global size of grid
    n = comm.bcast(n, root)

    # Create local FFT object, and send local grid-size to root process
    transform = janus.fft.parallel.create_real(n, comm)
    n_locs = comm.gather((transform.rshape[0], transform.offset0), root)

    mat = Material(0.75, 0.3, len(n))
    green = janus.discretegreenop.truncated(greenop.create(mat), n, 1.,
                                            transform)

    # Scatter x
    x_locs = None
    if rank == root:
        x_locs = [x[offset0:offset0 + n0] for n0, offset0 in n_locs]
    x_loc = comm.scatter(x_locs, root)
    y_loc = np.empty(transform.rshape + (green.oshape[-1],),
                       dtype=np.float64)
    green.apply(x_loc, y_loc)

    # Gather y
    y_locs = comm.gather(y_loc, root)
    if rank == root:
        actual = np.empty_like(expected)
        for y_loc, (n0, offset0) in zip(y_locs, n_locs):
            actual[offset0:offset0 + n0] = y_loc

        error = actual - expected
        error = actual - expected
        ulp = np.finfo(np.float64).eps
        nulp = rel_err / ulp
        assert_array_almost_equal_nulp(expected, np.asarray(actual), nulp)

def test_apply():

    directory = os.path.join('..', 'parallel',
                             os.path.dirname(os.path.realpath(__file__)))

    params = [('truncated_green_operator_200x300_unit_tau_xx_10x10+95+145.npz',
               1.2E-10),
              ('truncated_green_operator_200x300_unit_tau_yy_10x10+95+145.npz',
               6.7E-11),
              ('truncated_green_operator_200x300_unit_tau_xy_10x10+95+145.npz',
               7.7E-11),
              ('truncated_green_operator_40x50x60_unit_tau_xx_10x10x10+15+20+25.npz',
               1.7E-10),
              ('truncated_green_operator_40x50x60_unit_tau_yy_10x10x10+15+20+25.npz',
               7.6E-10),
              ('truncated_green_operator_40x50x60_unit_tau_zz_10x10x10+15+20+25.npz',
               1.6E-9),
              ('truncated_green_operator_40x50x60_unit_tau_yz_10x10x10+15+20+25.npz',
               1.6E-10),
              ('truncated_green_operator_40x50x60_unit_tau_zx_10x10x10+15+20+25.npz',
               6.4E-10),
              ('truncated_green_operator_40x50x60_unit_tau_xy_10x10x10+15+20+25.npz',
               1.6E-9),
               ]
    for filename, rel_err in params:
        path = os.path.join(directory, 'data', filename)
        yield do_test_apply, path, rel_err
