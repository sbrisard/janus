"""Various interfaces and base classes.

Classes:

- Operator: a general operator

"""

cdef class Operator:

    """General operator.

    Objects represented by this class map the `ncols`-dimensional real
    space to the `nrows`-dimensional real space.

    Attributes
    ----------
    ncols: int
        The dimension of the domain of the operator. For a linear
        operator, `ncols` is the number of columns of the underlying
        matrix.
    nrows: int
        The dimension of the codomain (range) of the operator. For a
        linear operator, `ncols` is the number of columns of the
        underlying matrix.
        
    """

    def __cinit__(self):
        pass

    cdef void c_apply(self, double[:] x, double[:] y):
        raise NotImplementedError
