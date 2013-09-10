from cython cimport boundscheck
from libc.math cimport sqrt

from matprop cimport IsotropicLinearElasticMaterial as Material

cdef class GreenOperator2d:
    cdef readonly int dim
    cdef int sym
    cdef readonly Material mat

    def __cinit__(self, Material mat):
        self.dim = 2
        self.sym = 3
        if (self.dim != mat.dim):
            raise ValueError('plane strain material expected')
        self.mat = mat
        
    @boundscheck(False)
    cpdef apply(self, double[::1] k, double[::1] tau):
        # The following tests are necessary, since bounds checks are removed.
        if k.shape[0] != self.dim:
            raise IndexError('shape of k must be ({0},)'.format(self.dim))

        if tau.shape[0] != self.sym:
            raise IndexError('shape of tau must be ({0},)'.format(self.sym))
        
        cdef double nx = k[0]
        cdef double ny = k[1]
        cdef double scale = 1. / sqrt(nx * nx + ny * ny)
        nx *= scale
        ny *= scale
