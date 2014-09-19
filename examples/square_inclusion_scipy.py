import sys
import time

import numpy as np
import scipy.sparse.linalg as spla
import skimage.io


skimage.io.use_plugin('freeimage', 'imread')
skimage.io.use_plugin('freeimage', 'imsave')
sys.path.append('..')

import janus.discretegreenop as discretegreenop
import janus.fft.serial as fft
import janus.material.elastic.linear.isotropic as material
import janus.operators as operators

class SquareInclusion:
    def __init__(self, a, mat_i, mat_m, mat_0, n):
        self.n0 = n
        self.n1 = n
        self.shape = (3 * self.n0 * self.n1,
                      3 * self.n0 * self.n1)
        self.dtype = np.float64
        transform = fft.create_real((self.n0, self.n1))
        self.green = discretegreenop.filtered(mat_0.green_operator(),
                                              transform.rshape,
                                              1., transform)
        aux_i = operators.isotropic_4((mat_i.k - mat_0.k) / 2.,
                                      (mat_i.g - mat_0.g) / 2.,
                                      dim=2)
        aux_m = operators.isotropic_4((mat_m.k - mat_0.k) / 2.,
                                      (mat_m.g - mat_0.g) / 2.,
                                      dim=2)

        op_loc = np.empty(transform.rshape, dtype=object)

        imax = int(np.ceil(n * a - 0.5))

        for i0 in range(self.n0):
            for i1 in range(self.n1):
                if (i0 < imax) and (i1 < imax):
                    op_loc[i0, i1] = aux_i
                else:
                    op_loc[i0, i1] = aux_m

        self.eps2tau = operators.BlockDiagonalOperator2D(op_loc)

    def create_vector(self):
        return np.empty(self.n0 * self.n1 * 3, dtype=np.float64)

    def matvec(self, x):
        xx = x.reshape(self.n0, self.n1, 3)
        return (xx + self.green.apply(self.eps2tau.apply(xx))).reshape(x.shape)

    def rmatvec(self, x):
        xx = x.reshape(self.n0, self.n1, 3)
        y = xx + self.eps2tau.apply(self.green.apply(xx))
        return y.reshape(x.shape)

def callback(r):
    print(r)

if __name__ == '__main__':

    mat_i = material.create(0., 0.2, dim=2)
    mat_m = material.create(1.0, 0.3, dim=2)
    mat_0 = material.create(1.0, 0.3, dim=2)

    a = 0.5
    n = 1024

    example = SquareInclusion(a, mat_i, mat_m, mat_0, n)

    b = example.create_vector()
    b_arr = b.reshape((n, n, 3))

    a = spla.aslinearoperator(example)

    eps_macro = np.array([0., 0., 1.])
    for i0 in range(example.n0):
        for i1 in range(example.n1):
            b_arr[i0, i1, :] = eps_macro

    print((b ** 2).sum())

    t1 = time.perf_counter()
    x, info = spla.gmres(a, b, callback=callback)
    t2 = time.perf_counter()
    t = t2 - t1
    print("Execution time {} s.".format(t))
    print("info = {}".format(info))
    r = b - a * x
    print('r = {}'.format((r**2).sum()))

    x_arr = x.reshape((n, n, 3))
    print('avg eps_xx = {}'.format(x_arr[:, :, 0].mean()))
    print('avg eps_yy = {}'.format(x_arr[:, :, 1].mean()))
    print('avg eps_xy = {}'.format(x_arr[:, :, 2].mean()))

    eps32 = x_arr.astype(np.float32)
    path = 'C:\\Users\\brisard\\Documents\\tmp\\'
    skimage.io.imsave(path + 'eps_xx.tif', eps32[:, :, 0])
    skimage.io.imsave(path + 'eps_yy.tif', eps32[:, :, 1])
    skimage.io.imsave(path + 'eps_xy.tif', eps32[:, :, 2])
