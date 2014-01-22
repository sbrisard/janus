from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound

from utils.checkarray cimport check_shape_1d
from utils.checkarray cimport create_or_check_shape_1d

"""This module defines fourth rank, isotropic tensor, which are
classically decomposed as the sum of a spherical and a deviatoric part

where

"""

def isotropic_4(sph, dev, dim):
    """Create a fourth rank, isotropic tensor.

    Parameters
    ----------
    sph: float
        The spherical projection of the returned tensor.
    dev: float
        The deviatoric projection of the returned tensor.
    dim: int
        The dimension of the physical space on which the returned tensor
        operates.
    """
    if dim == 2:
        return _FourthRankIsotropicTensor2D(sph, dev, dim)
    elif dim == 3:
        return _FourthRankIsotropicTensor3D(sph, dev, dim)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))

cdef class FourthRankIsotropicTensor(Operator):
    @cdivision(True)
    def __cinit__(self, double sph, double dev, int dim):
        """Should not be called directly, as the returned object would not
        properly initialized.

        """
        self.dim = dim
        self.nrows = (self.dim * (self.dim + 1)) / 2
        self.ncols = self.nrows
        self.sph = sph
        self.dev = dev
        self.tr = (sph - dev) / <double> self.dim

    def __repr__(self):
        return ('FourthRankIsotropicTensor(sph={0}, dev={1}, dim={2})'
                .format(self.sph, self.dev, self.dim))

    cdef void c_apply(self, double[:] x, double[:] y):
        pass

    @boundscheck(False)
    @wraparound(False)
    def apply(self, double[:] x, double[:] y=None):
        check_shape_1d(x, self.ncols)
        y = create_or_check_shape_1d(y, self.nrows)
        self.c_apply(x, y)
        return y

cdef class _FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double self_dev = self.dev
        cdef double x0 = x[0]
        cdef double x1 = x[1]
        cdef double aux = self.tr * (x0 + x1)
        y[0] = aux + self_dev * x0
        y[1] = aux + self_dev * x1
        y[2] = self_dev * x[2]

cdef class _FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double self_dev = self.dev
        cdef double x0 = x[0]
        cdef double x1 = x[1]
        cdef double x2 = x[2]
        cdef double aux = self.tr * (x0 + x1 + x2)
        y[0] = aux + self_dev * x0
        y[1] = aux + self_dev * x1
        y[2] = aux + self_dev * x2
        y[3] = self_dev * x[3]
        y[4] = self_dev * x[4]
        y[5] = self_dev * x[5]
