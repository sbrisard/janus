import numpy as np

import discretegreenop
import fft.parallel
import greenop

from mpi4py import MPI
from nose.tools import nottest
from nose.tools import raises
from numpy.testing import assert_array_almost_equal_nulp

from matprop import IsotropicLinearElasticMaterial as Material

@nottest
def do_test_convolve_2D(path_to_ref, rel_err):
    comm = MPI.COMM_WORLD
    root = 0
    rank = comm.rank

    # Load reference data
    n = None
    expected = None
    if rank == root:
        dummy = np.load(path_to_ref)
        n = dummy.shape[:-1]

        tau = dummy[:, :, 0:3]
        expected = dummy[:, :, 3:]

    # Broadcast global size of grid
    n = comm.bcast(n, root)

    # Create local FFT object, and send local grid-size to root process
    transform = fft.parallel.create_real(n, comm)
    n_locs = comm.gather((transform.rshape[0], transform.offset0), root)

    mat = Material(0.75, 0.3, 2)
    green = discretegreenop.create(greenop.create(mat), n, 1., transform)

    # Scatter tau
    tau_locs = None
    if rank == root:
        tau_locs = [tau[offset0:offset0 + n0] for n0, offset0 in n_locs]
    tau_loc = comm.scatter(tau_locs, root)
    eta_loc = np.empty(transform.rshape + (green.nrows,), dtype=np.float64)
    green.convolve(tau_loc, eta_loc)

    # Gather eta
    eta_locs = comm.gather(eta_loc, root)
    if rank == root:
        actual = np.empty_like(expected)
        for eta_loc, (n0, offset0) in zip(eta_locs, n_locs):
            actual[offset0:offset0 + n0] = eta_loc

        error = actual - expected
        error = actual - expected
        ulp = np.finfo(np.float64).eps
        nulp = rel_err / ulp
        assert_array_almost_equal_nulp(expected, np.asarray(actual), nulp)

def test_convolve_2D():

    params = [('truncated_green_operator_200x300_unit_tau_xx_10x10+95+145.npy',
               1.2E-10),
              ('truncated_green_operator_200x300_unit_tau_yy_10x10+95+145.npy',
               6.7E-11),
              ('truncated_green_operator_200x300_unit_tau_xy_10x10+95+145.npy',
               7.7E-11),
               ]
    for path_to_ref, rel_err in params:
        yield do_test_convolve_2D, path_to_ref, rel_err
