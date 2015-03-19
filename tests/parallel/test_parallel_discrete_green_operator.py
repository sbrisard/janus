import os.path

import numpy as np

import janus.fft.parallel as fft
import janus.material.elastic.linear.isotropic as material

from mpi4py import MPI
from numpy.testing import assert_allclose

from janus.green import truncated
from janus.green import filtered

ULP = np.finfo(np.float64).eps

class AbstractTestParallelDiscreteGreenOperator:

    def pytest_generate_tests(self, metafunc):
        if metafunc.function.__name__ == 'test_apply':
            if self.operates_in_place:
                flags = [0, 1, 2]
            else:
                flags = [0, 2]
            params = [i + (j,) for i in self.params_test_apply() for j in flags]
            metafunc.parametrize('path_to_ref, rtol, flag', params)


    def test_apply(self, path_to_ref, rtol, flag):
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
        transform = fft.create_real(n, comm)
        n_locs = comm.gather((transform.ishape[0], transform.offset0), root)

        green = self.greend(material.create(0.75, 0.3, len(n))
                                    .green_operator(),
                            n, 1., transform)

        # Scatter x
        x_locs = None
        if rank == root:
            x_locs = [x[offset0:offset0 + n0] for n0, offset0 in n_locs]
        x_loc = comm.scatter(x_locs, root)

        # if flag == 0:
        #     y_loc = green.apply(x_loc)
        # elif flag == 1:
        #     y_loc = green.apply(x_loc, x_loc)
        # elif flag == 2:
        #     base = np.empty(transform.ishape + (green.oshape[-1],),
        #                     dtype=np.float64)
        #     y_loc = green.apply(x_loc, base)
        # else:
        #     raise ValueError()

        y_loc = np.empty(transform.ishape + (green.oshape[-1],),
                        dtype=np.float64)
        green.apply(x_loc, y_loc)

        # Gather y
        y_locs = comm.gather(y_loc, root)
        if rank == root:
            actual = np.empty_like(expected)
            for y_loc, (n0, offset0) in zip(y_locs, n_locs):
                actual[offset0:offset0 + n0] = y_loc
            assert_allclose(actual, expected, rtol, 10 * ULP)

class TestParallelTruncatedGreenOperator(AbstractTestParallelDiscreteGreenOperator):
    operates_in_place = True

    def greend(self, greenc, n, h, transform=None):
        return truncated(greenc, n, h, transform)

    def params_test_apply(self):
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
                  (template_3D.format('xy'), ULP)]
        return [(os.path.join(directory, '..', 'data', filename), rtol)
                for filename, rtol in params]


class TestParallelFilteredGreenOperator(AbstractTestParallelDiscreteGreenOperator):
    operates_in_place = True

    def greend(self, greenc, n, h, transform=None):
        return filtered(greenc, n, h, transform)

    def params_test_apply(self):
        directory = os.path.dirname(os.path.realpath(__file__))
        template_2D = ('filtered_green_operator_200x300_'
                       'unit_tau_{0}_10x10+95+145.npz')
        template_3D = ('filtered_green_operator_40x50x60_'
                       'unit_tau_{0}_10x10x10+15+20+25.npz')
        params = [(template_2D.format('xx'), ULP),
                  (template_2D.format('yy'), ULP),
                  (template_2D.format('xy'), ULP),
                  (template_3D.format('xx'), ULP),
                  (template_3D.format('yy'), ULP),
                  (template_3D.format('zz'), ULP),
                  (template_3D.format('yz'), ULP),
                  (template_3D.format('zx'), ULP),
                  (template_3D.format('xy'), ULP)]
        return [(os.path.join(directory, '..', 'data', filename), rtol)
                for filename, rtol in params]
