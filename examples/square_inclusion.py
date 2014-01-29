import sys
import time

import numpy as np
import petsc4py
import skimage.io

petsc4py.init(sys.argv)
skimage.io.use_plugin('freeimage', 'imread')
skimage.io.use_plugin('freeimage', 'imsave')
sys.path.append('..')

import janus.discretegreenop as discretegreenop
import janus.fft.parallel as fft
import janus.greenop as greenop
import janus.operators as operators

from mpi4py import MPI
from petsc4py import PETSc

from janus.matprop import IsotropicLinearElasticMaterial as Material

class SquareInclusion:
    def __init__(self, a, mat_i, mat_m, mat_0, n, comm=MPI.COMM_WORLD):
        transform = fft.create_real((n, n), comm)
        self.n0 = transform.rshape[0]
        self.n1 = transform.rshape[1]
        self.offset0 = transform.offset0
        self.green = discretegreenop.create(greenop.create(mat_0), (n, n),
                                            1., transform)
        aux_i = operators.isotropic_4(1. / (2. * (mat_i.k - mat_0.k)),
                                      1. / (2. * (mat_i.g - mat_0.g)),
                                      dim=2)
        aux_m = operators.isotropic_4(1. / (2. * (mat_m.k - mat_0.k)),
                                      1. / (2. * (mat_m.g - mat_0.g)),
                                      dim=2)

        ops = np.empty(transform.rshape, dtype=object)

        imax = int(np.ceil(n * a - 0.5))

        for i0 in range(self.n0):
            for i1 in range(self.n1):
                if (self.offset0 + i0 < imax) and (i1 < imax):
                    ops[i0, i1] = aux_i
                else:
                    ops[i0, i1] = aux_m

        self.tau2eps = operators.BlockDiagonalOperator2D(ops)

    def create_vector(self):
        return PETSc.Vec().createMPI(size=(self.n1 * self.n1 * 3),
                                     bsize=(self.n0 * self.n1 * 3))

    def mult(self, mat, x, y):
        x_arr = x.array.reshape((self.n0, self.n1, 3))
        y_arr = y.array.reshape((self.n0, self.n1, 3))
        self.green.convolve(x_arr, y_arr)
        y_arr += np.asarray(self.tau2eps.apply(x_arr))

if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    mat_i = Material(10.0, 0.2, dim=2)
    mat_m = Material(1.0, 0.3, dim=2)
    mat_0 = Material(0.9, 0.3, dim=2)

    a = 0.5
    n = 256

    example = SquareInclusion(a, mat_i, mat_m, mat_0, n)
    local_shape = (example.n0, example.n1, 3)

    x = example.create_vector()
    y = example.create_vector()
    b = example.create_vector()

    x_arr = x.array.reshape(local_shape)
    y_arr = y.array.reshape(local_shape)
    b_arr = b.array.reshape(local_shape)

    a = PETSc.Mat().createPython([x.getSizes(), b.getSizes()])
    a.setPythonContext(example)
    a.setUp()

    eps_macro = np.array([0., 0., 1.])
    for i0 in range(example.n0):
        for i1 in range(example.n1):
            b_arr[i0, i1, :] = eps_macro

    ksp = PETSc.KSP().create()
    ksp.setOperators(a)
    ksp.setType('cg')
    ksp.getPC().setType('none')
    ksp.setFromOptions()

    t1 = time.perf_counter()
    ksp.solve(b, x)
    t2 = time.perf_counter()
    t = t2 - t1
    print("I'm process {}: execution time {} s.".format(comm.rank, t))

    example.tau2eps.apply(x_arr, y_arr)

    # Gather eps
    gathered = comm.gather((y_arr, example.offset0))

    if comm.rank == 0:
        eps = np.empty((n, n, 3), dtype=np.float64)
        for eps_loc, offset0 in gathered:
            n0 = eps_loc.shape[0]
            eps[offset0:(offset0 + n0), :, :] = eps_loc
        eps32 = eps.astype(np.float32)
        path = '/media/sf_brisard/Documents/tmp/'
        skimage.io.imsave(path + 'eps_xx.tif', eps32[:, :, 0])
        skimage.io.imsave(path + 'eps_yy.tif', eps32[:, :, 1])
        skimage.io.imsave(path + 'eps_xy.tif', eps32[:, :, 2])
