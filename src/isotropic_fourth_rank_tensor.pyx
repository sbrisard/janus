from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound

from checkarray cimport check_shape_1d
from checkarray cimport create_or_check_shape_1d

"""This module defines fourth rank, isotropic tensor, which are
classically decomposed as the sum of a spherical and a deviatoric part

where 

"""

def create(sph, dev, dim):
    """Create a new instance of ``IsotropicFourthRankTensor``.

    Parameters
    ----------
    sph: float
        The spherical projection of the tensor to be returned.
    dev: float
        The deviatoric projection of the tensor to be returned.
    """
    if dim == 2:
        return IsotropicFourthRankTensor2D(sph, dev)
    elif dim == 3:
        return IsotropicFourthRankTensor3D(sph, dev)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))

cdef class IsotropicFourthRankTensor2D:

    @cdivision(True)
    def __cinit__(self, double sph, double dev):
        self.dim = 2
        self.sph = sph
        self.dev = dev
        self.s = (sph - dev) / <double> self.dim

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double aux1 = self.s * (x[0] + x[1])
        cdef double aux2 = self.dev
        y[0] = aux1 + aux2 * x[0]
        y[1] = aux1 + aux2 * x[1]
        y[2] = aux2 * x[2]

    def apply(self, double[:] x, double[:] y=None):
        check_shape_1d(x, 3)
        y = create_or_check_shape_1d(y, 3)
        self.c_apply(x, y)
        return y

cdef class IsotropicFourthRankTensor3D:

    @cdivision(True)
    def __cinit__(self, double sph, double dev):
        self.dim = 3
        self.sph = sph
        self.dev = dev
        self.s = (sph - dev) / <double> self.dim

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double aux1 = self.s * (x[0] + x[1] + x[2])
        cdef double aux2 = self.dev
        y[0] = aux1 + aux2 * x[0]
        y[1] = aux1 + aux2 * x[1]
        y[2] = aux1 + aux2 * x[2]
        y[3] = aux2 * x[3]
        y[4] = aux2 * x[4]
        y[5] = aux2 * x[5]

    def apply(self, double[:] x, double[:] y=None):
        check_shape_1d(x, 6)
        y = create_or_check_shape_1d(y, 6)
        self.c_apply(x, y)
        return y
