from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound

from checkarray cimport check_shape_1d
from checkarray cimport create_or_check_shape_1d

"""This module defines fourth rank, isotropic tensor, which are
classically decomposed as the sum of a spherical and a deviatoric part

where

"""

def isotropic4(sph, dev, dim):
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

cdef class FourthRankIsotropicTensor:
    @cdivision(True)
    def __cinit__(self, double sph, double dev, int dim):
        """Should not be called directly, as the returned object would not
        properly initialized.

        """
        self.dim = dim
        self.sym = (self.dim * (self.dim + 1)) / 2
        self.sph = sph
        self.dev = dev
        self.tr = (sph - dev) / <double> self.dim

    def __repr__(self):
        return ('FourthRankIsotropicTensor(sph={0}, dev={1}, dim={2})'
                .format(self.sph, self.dev, self.dim))

    cdef void c_apply(self, double[:] x, double[:] y):
        pass

    def apply(self, double[:] x, double[:] y=None):
        check_shape_1d(x, self.sym)
        y = create_or_check_shape_1d(y, self.sym)
        self.c_apply(x, y)
        return y

cdef class _FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):
    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double aux = self.tr * (x[0] + x[1])
        cdef double self_dev = self.dev
        y[0] = aux + self_dev * x[0]
        y[1] = aux + self_dev * x[1]
        y[2] = self_dev * x[2]

cdef class _FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):
    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double aux = self.tr * (x[0] + x[1] + x[2])
        cdef double self_dev = self.dev
        y[0] = aux + self_dev * x[0]
        y[1] = aux + self_dev * x[1]
        y[2] = aux + self_dev * x[2]
        y[3] = self_dev * x[3]
        y[4] = self_dev * x[4]
        y[5] = self_dev * x[5]
