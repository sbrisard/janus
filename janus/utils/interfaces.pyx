cdef class Operator:
    def __cinit__(self):
        pass

    cdef void c_apply(self, double[:] x, double[:] y):
        raise NotImplementedError
