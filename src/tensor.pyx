from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound

from checkarray cimport check_shape_1d
from checkarray cimport create_or_check_shape_1d

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

    cdef void c_apply(self, double *x, int sx0, double *y, int sy0):
        pass

    @boundscheck(False)
    @wraparound(False)
    def apply(self, double[:] x, double[:] y=None):
        check_shape_1d(x, self.sym)
        y = create_or_check_shape_1d(y, self.sym)
        self.c_apply(&x[0], x.strides[0], &y[0], y.strides[0])
        return y

cdef class _FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):
    cdef inline void c_apply(self, double *x, int sx0, double *y, int sy0):
        cdef double self_dev = self.dev
        cdef char *xx = <char *> x
        cdef double x0 = (<double *> xx)[0]
        xx += sx0
        cdef double x1 = (<double *> xx)[0]
        cdef double aux = self.tr * (x0 + x1)
        cdef char *yy = <char *> y
        (<double *> yy)[0] = aux + self_dev * x0
        yy += sy0
        (<double *> yy)[0] = aux + self_dev * x1
        xx += sx0
        yy += sy0
        (<double *> yy)[0] = self_dev * (<double *> xx)[0]

cdef class _FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):
    cdef inline void c_apply(self, double *x, int sx0, double *y, int sy0):
        cdef double self_dev = self.dev
        cdef char *xx = <char *> x
        cdef double x0 = (<double *> xx)[0]
        xx += sx0
        cdef double x1 = (<double *> xx)[0]
        xx += sx0
        cdef double x2 = (<double *> xx)[0]
        cdef double aux = self.tr * (x0 + x1 + x2)
        cdef char *yy = <char *>y
        (<double *> yy)[0] = aux + self_dev * x0
        yy += sy0
        (<double *> yy)[0] = aux + self_dev * x1
        yy += sy0
        (<double *> yy)[0] = aux + self_dev * x2
        xx += sx0
        yy += sy0
        (<double *> yy)[0] = self_dev * (<double *> xx)[0]
        xx += sx0
        yy += sy0
        (<double *> yy)[0] = self_dev * (<double *> xx)[0]
        xx += sx0
        yy += sy0
        (<double *> yy)[0] = self_dev * (<double *> xx)[0]
