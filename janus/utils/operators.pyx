"""Various interfaces and base classes.

Classes:

- Operator: a general operator

"""
from janus.utils.checkarray cimport check_shape_1d
from janus.utils.checkarray cimport create_or_check_shape_1d


cdef class Operator:

    """General operator.

    Objects represented by this class map the `ncols`-dimensional real
    space to the `nrows`-dimensional real space.

    Attributes
    ----------
    ncols : int
        The dimension of the domain of the operator. For a linear
        operator, `ncols` is the number of columns of the underlying
        matrix.
    nrows : int
        The dimension of the codomain (range) of the operator. For a
        linear operator, `ncols` is the number of columns of the
        underlying matrix.

    """

    def __cinit__(self):
        pass

    cdef void c_apply(self, double[:] x, double[:] y):
        raise NotImplementedError

    def apply(self, double[:] x, double[:] y=None):
        """Return the result of applying this operator to `x`.

        Parameters
        ----------
        x : memoryview to float64
            The input vector.
        y : memoryview to float64, optional
            The vector in which the output is to be stored. If ``None``,
            a new array is created.

        Returns
        -------
        y : memoryview to float64
            The value of ``A(x)``, if `A` is the operator. This is a
            view of the input parameter with the same name (if not
            ``None``).
            
        """
        check_shape_1d(x, self.ncols)
        y = create_or_check_shape_1d(y, self.nrows)
        self.c_apply(x, y)
        return y
