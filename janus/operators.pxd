cdef class AbstractOperator:
    cdef readonly int isize
    """The size of the input (`int`, read-only)."""

    cdef readonly int osize
    """The size of the output (`int`, read-only)."""

    cdef void c_apply(self, double[:] x, double[:] y)


cdef class AbstractLinearOperator(AbstractOperator):
    cdef void c_to_memoryview(self, double[:, :] out)


cdef class FourthRankIsotropicTensor(AbstractLinearOperator):
    cdef readonly int dim
    cdef readonly double sph, dev
    cdef double tr


cdef class FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):
    pass


cdef class FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):
    pass

cdef class AbstractStructuredOperator2D:
    cdef readonly int dim
    cdef readonly int shape0, shape1, ishape2, oshape2
    cdef readonly tuple ishape, oshape

    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y)


cdef class AbstractStructuredOperator3D:
    cdef readonly int dim
    cdef readonly int shape0, shape1, shape2, ishape3, oshape3
    cdef readonly tuple ishape, oshape

    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y)


cdef class BlockDiagonalOperator2D(AbstractStructuredOperator2D):
    cdef AbstractOperator[:, :] loc


cdef class BlockDiagonalOperator3D(AbstractStructuredOperator3D):
    cdef AbstractOperator[:, :, :] loc


cdef class BlockDiagonalLinearOperator2D(AbstractStructuredOperator2D):
    cdef double[:, :, :, :] a

    cdef void c_apply_transpose(self, double[:, :, :] x, double[:, :, :] y)


cdef class BlockDiagonalLinearOperator3D(AbstractStructuredOperator3D):
    cdef double[:, :, :, :, :] a

    cdef void c_apply_transpose(self, double[:, :, :, :] x,
                                double[:, :, :, :] y)
