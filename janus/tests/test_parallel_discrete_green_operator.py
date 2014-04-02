import os.path

import numpy as np

import janus.discretegreenop
import janus.fft.parallel
import janus.greenop as greenop

from mpi4py import MPI
from nose.tools import nottest
from nose.tools import raises
from numpy.testing import assert_allclose
from numpy.testing import assert_array_almost_equal_nulp

from janus.matprop import IsotropicLinearElasticMaterial as Material

ULP = np.finfo(np.float64).eps

@nottest
def do_test_apply(path_to_ref, rtol):
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

        assert_allclose(actual, expected, rtol, 10 * ULP)

def test_apply():

    directory = os.path.dirname(os.path.realpath(__file__))
    template_2D = ('truncated_green_operator_200x300_'
                   'unit_tau_{0}_10x10+95+145.npz')
    template_3D = ('truncated_green_operator_40x50x60_'
                   'unit_tau_{0}_10x10x10+15+20+25.npz')
    params = [(template_2D.format('xx'), ULP),
              (template_2D.format('yy'), ULP),
              (template_2D.format('xy'), ULP),
              (template_3D.format('xx'), ULP),
              (template_3D.format('yy'), ULP),
              (template_3D.format('zz'), ULP),
              (template_3D.format('yz'), ULP),
              (template_3D.format('zx'), ULP),
              (template_3D.format('xy'), ULP),
               ]

    for filename, rtol in params:
        path = os.path.join(directory, 'data', filename)
        yield do_test_apply, path, rtol
