cdef class Operator:
    cdef readonly int isize, osize

    cdef void c_apply(self, double[:] x, double[:] y)


cdef class AbstractLinearOperator(Operator):
    cdef void c_to_memoryview(self, double[:, :] out)


cdef class AbstractStructuredOperator2D:
    cdef readonly int dim
    cdef readonly int ishape0, ishape1, ishape2
    cdef readonly int oshape0, oshape1, oshape2
    cdef readonly tuple ishape, oshape

    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y)


cdef class AbstractStructuredOperator3D:
    cdef readonly int dim
    cdef readonly int ishape0, ishape1, ishape2, ishape3
    cdef readonly int oshape0, oshape1, oshape2, oshape3
    cdef readonly tuple ishape, oshape

    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y)


cdef class BlockDiagonalOperator2D(AbstractStructuredOperator2D):
    cdef Operator[:, :] a_loc


cdef class FourthRankIsotropicTensor(Operator):
    cdef readonly int dim
    cdef readonly double sph, dev
    cdef double tr


cdef class FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):
    pass


cdef class FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):
    pass
