"""Classes defining operators (:mod:`janus.utils.operators`)
=========================================================

The (concrete and abstract) classes defined in this module define
operators in the most general sense, as a mapping from a (real) vector
space to another.

Classes:

- :class:`Operator` -- general operator
- :class:`AbstractStructuredOperator2D` -- operator with 2D layout of
         the (vectorial) data
- :class:`BlockDiagonalOperator2D` -- block-diagonal, structured operator

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
        """apply(x, y=None)

        Return the result of applying this operator to `x`.

        Parameters
        ----------
        x : 1D memoryview of float64
            The input vector.
        y : 1D memoryview of float64, optional
            The output vector. Its content is altered by this method. If
            ``None``, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        y : 1D memoryview of float64
            The result of applying this operator to `x`. This is a
            view of the parameter `y` (if not ``None``).

        """
        check_shape_1d(x, self.ncols)
        y = create_or_check_shape_1d(y, self.nrows)
        self.c_apply(x, y)
        return y


cdef class AbstractStructuredOperator2D:

    """Operator applied to vectorial data structured in a 2D grid.

    Objects represented by this class map real vectors of dimension
    ``isize`` to real vectors of dimension ``osize``. Input and output
    vectors are structured in 2D grids, each grid-cell being a vector
    itself. Therefore, the input (resp. output) is a (``float64``)
    memoryview of shape ``(ishape0, ishape1, ishape2)`` [resp.
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

    def apply(self, double[:, :, :] x, double[:, :, :] y=None):
        """apply(x, y=None)

        Return the result of applying this operator to `x`.

        The default implementation calls the (Cython) method
        ``c_apply()``.

        Parameters
        ----------
        x : 3D memoryview of float64
            The input vector.
        y : 3D memoryview of float64, optional
            The output vector. Its content is altered by this method. If
            ``None``, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        y : 3D memoryview of float64
            The result of applying this operator to `x`. This is a
            view of the parameter `y` (if not ``None``).

        """
        check_shape_3d(x, self.ishape0, self.ishape1, self.ishape2)
        y = create_or_check_shape_3d(y, self.oshape0, self.oshape1,
                                     self.oshape2)
        self.c_apply(x, y)
        return y


cdef class BlockDiagonalOperator2D(AbstractStructuredOperator2D):

    """Block-diagonal operator with 2D layout of the (vectorial) data.

    This class defines concrete implementations of
    :class:`AbstractStructuredOperator2D`. Such operators are defined by
    a 2D array ``a_loc[:, :]`` of local operators (:class:`Operator`).
    Then, the input ``x[:, :, :]`` is mapped to the output
    ``y[:, :, :]``, such that::

        y[i0, i1, :] = a_loc[i0, i1].apply(x)

    TODO : all local operators must have same input and output dimensions
    TODO : what is the ishape and oshape of such an operator?

    Parameters
    ----------
    a_loc : 2D memoryview of :class:`Operator`
        The array of local operators.

    """

    def __cinit__(self, Operator[:, :] a_loc):
        self.ishape0 = a_loc.shape[0]
        self.ishape1 = a_loc.shape[1]
        self.ishape2 = a_loc[0, 0].ncols
        self.oshape0 = a_loc.shape[0]
        self.oshape1 = a_loc.shape[1]
        self.oshape2 = a_loc[0, 0].nrows
        self.ishape = (self.ishape0, self.ishape1, self.ishape2)
        self.oshape = (self.oshape0, self.oshape1, self.oshape2)
        self.a_loc = a_loc.copy()

        cdef int i0, i1, ishape2, oshape2
        for i0 in range(self.ishape0):
            for i1 in range(self.ishape1):
                ishape2 = a_loc[i0, i1].ncols
                oshape2 = a_loc[i0, i1].nrows
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
                op = self.a_loc[i0, i1]
                op.c_apply(x[i0, i1, :], y[i0, i1, :])
