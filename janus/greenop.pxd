from janus.operators cimport AbstractLinearOperator

cdef class AbstractGreenOperator(AbstractLinearOperator):
    cdef readonly int dim

    cdef void c_set_frequency(self, double[:] k)
