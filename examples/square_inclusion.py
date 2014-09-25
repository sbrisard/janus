import sys

import numpy as np
import petsc4py

petsc4py.init(sys.argv)
sys.path.append('..')

import janus.green as green
import janus.fft.parallel as fft
import janus.material.elastic.linear.isotropic as material
import janus.operators as operators

from mpi4py import MPI
from petsc4py import PETSc

class SquareInclusion:
    def __init__(self, a, mat_i, mat_m, mat_0, n, comm=MPI.COMM_WORLD):
        transform = fft.create_real((n, n), comm)
        self.n0 = transform.rshape[0]
        self.n1 = transform.rshape[1]
        self.offset0 = transform.offset0
        self.green = green.truncated(mat_0.green_operator(),
                                               (n, n), 1., transform)
        aux_i = operators.isotropic_4(1. / (2. * (mat_i.k - mat_0.k)),
                                      1. / (2. * (mat_i.g - mat_0.g)),
                                      dim=2)
        aux_m = operators.isotropic_4(1. / (2. * (mat_m.k - mat_0.k)),
                                      1. / (2. * (mat_m.g - mat_0.g)),
                                      dim=2)

        ops = np.empty(transform.rshape, dtype=object)
        ops[:, :] = aux_m
        imax = int(np.ceil(n * a - 0.5))
        ops[:imax - self.offset0, :imax] = aux_i

        self.tau2eps = operators.BlockDiagonalOperator2D(ops)

    def create_vector(self):
        return PETSc.Vec().createMPI(size=(self.n1 * self.n1 * 3),
                                     bsize=(self.n0 * self.n1 * 3))

    def mult(self, mat, x, y):
        x_arr = x.array.reshape((self.n0, self.n1, 3))
        y_arr = y.array.reshape((self.n0, self.n1, 3))
        self.green.apply(x_arr, y_arr)
        y_arr += np.asarray(self.tau2eps.apply(x_arr))

if __name__ == '__main__':
    comm = MPI.COMM_WORLD

    mat_i = material.create(10.0, 0.2, dim=2)
    mat_m = material.create(1.0, 0.3, dim=2)
    mat_0 = material.create(0.9, 0.3, dim=2)

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
    b_arr[:, :, :] = eps_macro

    ksp = PETSc.KSP().create()
    ksp.setTolerances(rtol=1E-10)
    ksp.setOperators(a)
    ksp.setType('cg')
    ksp.getPC().setType('none')
    ksp.setFromOptions()

    t1 = PETSc.Log.getTime()
    ksp.solve(b, x)
    t2 = PETSc.Log.getTime()
    t = t2 - t1

    example.tau2eps.apply(x_arr, y_arr)

    # Gather tau and eps
    if comm.rank == 0:
        tau = np.empty((n, n, 3), dtype=np.float64)
        eps = np.empty((n, n, 3), dtype=np.float64)
    else:
        tau = None
        eps = None
    comm.Gatherv(x_arr, tau, 0)
    comm.Gatherv(y_arr, eps, 0)

    if comm.rank == 0:
        print('Computation time = {} s'.format(t))
        print('Total number of iterations = {}'.format(ksp.getIterationNumber()))
        print('Residual norm = {}'.format(ksp.getResidualNorm()))
        print('Avg tau = {}'.format(tau.mean(axis=(0, 1))))
        print('Avg eps = {}'.format(eps.mean(axis=(0, 1))))
