cdef class FourthRankIsotropicTensor:
    """
    nrows: int
        The number of rows of the underlying matrix.
    ncols: int
        The number of columns of the underlying matrix.

    """
    cdef readonly int dim, nrows, ncols
    cdef readonly double sph, dev
    cdef double tr
    cdef void c_apply(self, double[:] x, double[:] y)

