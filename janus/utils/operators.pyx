"""Abstract base classes defining operators (:mod:`janus.utils.operators`)
=======================================================================

The abstract classes defined in this module define operators in the most
general sense, that is a mapping from a (real) vector space to another.

Classes:

- :class:`Operator` -- general operator
- :class:`AbstractStructuredOperator2D` -- operator with 2D layout of
  the (tensorial) data

"""
from cython cimport boundscheck
from cython cimport wraparound

from janus.utils.checkarray cimport check_shape_1d
from janus.utils.checkarray cimport check_shape_3d
from janus.utils.checkarray cimport create_or_check_shape_1d
from janus.utils.checkarray cimport create_or_check_shape_3d


cdef class Operator:

    """General operator.

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

cdef class AbstractStructuredOperator2D:

    """Operator applied to data structured in a 2D grid.

    Objects represented by this class map ``isize``-dimensional real data
    to ``osize``-dimensional real data. Input and output datas are
    structured in 2D grids, each grid-cell being a vector itself.
    Therefore, the input (resp. output) is a (``float64``) memoryview of
    shape ``(ishape0, ishape1, ishape2)`` [resp.
    ``(oshape0, oshape1, oshape2)``], with::

        isize = ishape0 * ishape1 * ishape2
        osize = oshape0 * oshape1 * oshape2

    Attributes
    ----------
    ishape0 : int
        The first dimension of the input.
    ishape1 : int
        The second dimension of the input.
    ishape2 : int
        The third dimension of the input.
    oshape0 : int
        The first dimension of the output.
    oshape1 : int
        The second dimension of the output.
    oshape2 : int
        The third dimension of the output.
    ishape : tuple
        The shape of the input [the tuple ``(ishape0, ishape1, ishape2)``].
    oshape : tuple
        The shape of the output [the tuple ``(oshape0, oshape1, oshape2)``].

    """

    def __cinit__(self):
        pass

    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        raise NotImplementedError

    def apply(self, x, y=None):
        """apply(x, y=None)
        
        Return the result of applying this operator to `x`.

        The default implementation calls the (Cython) method
        ``c_apply()``.

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
        check_shape_3d(x, self.ishape0, self.ishape1, self.ishape2)
        y = create_or_check_shape_3d(y, self.oshape0, self.oshape1,
                                     self.oshape2)
        self.c_apply(x, y)
        return y

cdef class BlockDiagonalOperator2D(AbstractStructuredOperator2D):

    def __cinit__(self, Operator[:, :] op):
        self.ishape0 = op.shape[0]
        self.ishape1 = op.shape[1]
        self.ishape2 = op[0, 0].ncols
        self.oshape0 = op.shape[0]
        self.oshape1 = op.shape[1]
        self.oshape2 = op[0, 0].nrows
        self.ishape = (self.ishape0, self.ishape1, self.ishape2)
        self.oshape = (self.oshape0, self.oshape1, self.oshape2)
        self.op = op.copy()

        cdef int i0, i1, ishape2, oshape2
        for i0 in range(self.ishape0):
            for i1 in range(self.ishape1):
                ishape2 = op[i0, i1].ncols
                oshape2 = op[i0, i1].nrows
                if ishape2 != self.ishape2:
                    raise ValueError('invalid dimension block operator input: '
                                     'expected {0}, '
                                     'actual {1}'.format(self.ishape2,
                                                         ishape2))
                if oshape2 != self.oshape2:
                    raise ValueError('invalid dimension block operator output: '
                                     'expected {0}, '
                                     'actual {1}'.format(self.oshape2,
                                                         oshape2))

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int i0, i1
        cdef Operator op

        for i0 in range(self.ishape0):
            for i1 in range(self.ishape1):
                op = self.op[i0, i1]
                op.c_apply(x[i0, i1, :], y[i0, i1, :])
