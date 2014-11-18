# -*- coding: utf-8 -*-

"""This module defines operators in the general sense, as a mapping from a
(real) vector space to another.

Functions defined in this module
--------------------------------

- :func:`isotropic_4` -- create a fourth-rank, isotropic tensor with minor
  symmetries.
- :func:`block_diagonal_operator` -- create a block-diagonal operator.
- :func:`block_diagonal_linear_operator` -- create a block-diagonal, linear
  operator.

Classes defined in this module
------------------------------

- :class:`AbstractOperator` -- general operator
- :class:`AbstractLinearOperator` -- general linear operator
- :class:`FourthRankIsotropicTensor` -- fourth-rank, isotropic tensor with
  minor symmetries
- :class:`FourthRankIsotropicTensor2D` -- specialization of the above to 2D
- :class:`FourthRankIsotropicTensor3D` -- specialization of the above to 3D
- :class:`AbstractStructuredOperator2D` -- operator with 2D layout of
  the data
- :class:`AbstractStructuredOperator3D` -- operator with 3D layout of
  the data
- :class:`BlockDiagonalOperator2D` -- block-diagonal operator with 2D
  layout of the data
- :class:`BlockDiagonalOperator3D` -- block-diagonal operator with 3D
  layout of the data
- :class:`BlockDiagonalLinearOperator2D` -- block-diagonal, linear
  operator with 2D layout of the data
- :class:`BlockDiagonalLinearOperator3D` -- block-diagonal, linear
  operator with 3D layout of the data

"""

from cython cimport boundscheck
from cython cimport cdivision
from cython cimport wraparound

from janus.utils.checkarray cimport check_shape_1d
from janus.utils.checkarray cimport check_shape_3d
from janus.utils.checkarray cimport check_shape_4d
from janus.utils.checkarray cimport create_or_check_shape_1d
from janus.utils.checkarray cimport create_or_check_shape_2d
from janus.utils.checkarray cimport create_or_check_shape_3d
from janus.utils.checkarray cimport create_or_check_shape_4d


def isotropic_4(sph, dev, dim):
    """isotropic_4(sph, dev, dim)

    Create a fourth rank, isotropic tensor.

    Parameters
    ----------
    sph: float
        The spherical projection of the returned tensor.
    dev: float
        The deviatoric projection of the returned tensor.
    dim: int
        The dimension of the physical space on which the returned tensor
        operates.

    """
    if dim == 2:
        return FourthRankIsotropicTensor2D(sph, dev, dim)
    elif dim == 3:
        return FourthRankIsotropicTensor3D(sph, dev, dim)
    else:
        raise ValueError('dim must be 2 or 3 (was {0})'.format(dim))


def block_diagonal_operator(loc):
    """block_diagonal_operator(loc)

    Create a block-diagonal operator.

    Parameters
    ----------
    loc : 2D memoryview of :class:`AbstractOperator`
        The array of local operators. The returned instance keeps a
        *shallow* copy of `loc`.

    """
    dim = len(loc.shape)
    if dim == 2:
        return BlockDiagonalOperator2D(loc)
    elif dim == 3:
        return BlockDiagonalOperator3D(loc)
    else:
        raise ValueError('number of dimensions of loc must be 2 or 3 '
                         '(was {0})'.format(dim))


def block_diagonal_linear_operator(a):
    """block_diagonal_linear_operator(a)

    Create a block-diagonal, linear operator.

    Parameters
    ----------
    a : 4D or 5D memoryview of `float64`.
        The array of local matrices. The returned instance keeps a
        *shallow* copy of `a`.

    """
    dim = len(a.shape) - 2
    if dim == 2:
        return BlockDiagonalLinearOperator2D(a)
    elif dim == 3:
        return BlockDiagonalLinearOperator3D(a)
    else:
        raise ValueError('number of dimensions of a must be 4 or 5 '
                         '(was {0})'.format(dim))


cdef class AbstractOperator:

    """General operator.

    Concrete instances of this class map arrays of size `isize` to
    arrays of size `osize`.

    Attributes
    ----------
    isize : int
        The size of the input.
    osize : int
        The size of the output.

    """

    def __cinit__(self):
        pass

    def init_sizes(self, int isize, int osize):
        """init_sizes(isize, osize)

        Initialize the values of the `isize` and `osize` attributes.

        This method is provided as a convenience to users who want to
        create pure Python implementations of this class. It should be
        called only *once* at the initialization of the instance, as it
        can potentially modify the `isize` and `osize` attributes (which
        are otherwise read-only).

        Parameters
        ----------
        isize : int
            The size of the input.
        osize : int
            The size of the output.

        """
        if isize <= 0:
            raise ValueError('isize should be > 0 (was {})'.format(isize))
        if osize <= 0:
            raise ValueError('osize should be > 0 (was {})'.format(osize))
        self.isize = isize
        self.osize = osize

    cdef void c_apply(self, double[:] x, double[:] y):
        pass

    def apply(self, double[:] x, double[:] y=None):
        """apply(x, y=None)

        Return the result of applying this operator to `x`.

        Parameters
        ----------
        x : 1D memoryview of float64
            The input vector.
        y : 1D memoryview of float64, optional
            The output vector. Its content is altered by this method. If
            `None`, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        y : 1D memoryview of float64
            The result of applying this operator to `x`. This is a
            view of the parameter `y` (if not `None`).

        """
        check_shape_1d(x, self.isize)
        y = create_or_check_shape_1d(y, self.osize)
        self.c_apply(x, y)
        return y


cdef class AbstractLinearOperator(AbstractOperator):

    """Specialization of :class:`AbstractOperator` to linear operators.

    This abstract class defines the method :func:`to_memoryview` which
    returns the matrix of this linear operator as a memoryview.
    """

    cdef void c_to_memoryview(self, double[:, :] out):
        raise NotImplementedError

    def to_memoryview(self, double[:, :] out=None):
        """to_memoryview(out=None)

        Return the matrix of this operator as a memoryview.

        Parameters
        ----------
        out : 2D memoryview of float64
            The output array, in which the matrix is to be stored. If
            `None`, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        out : 2D memoryview of float64
            The matrix of this operator. This is a view of the parameter
            `out` (if not `None`).

        """
        out = create_or_check_shape_2d(out, self.osize, self.isize)
        self.c_to_memoryview(out)
        return out


cdef class FourthRankIsotropicTensor(AbstractLinearOperator):

    """
    Fourth rank, isotropic tensor with minor symmetries.

    Such a tensor is defined by its spherical and deviatoric
    projections. **Warning:** this class should *not* be instantiated
    directly, as the object returned by ``__cinit__`` would not be in a
    legal state. Use :func:`isotropic_4` instead.

    Parameters
    ----------
    sph : float64
        The spherical projection of the tensor.
    dev : float64
        The deviatoric projection of the tensor.
    dim : int
        The dimension of the space on which the tensor operates.

    See also
    --------
    isotropic_4

    Notes
    -----
    Any fourth rank, isotropic tensor ``T`` is a linear combination of
    the spherical projection tensor ``J`` and the deviatoric projection
    tensor ``K`` ::

        T = sph * J + dev * K.

    ``sph`` and ``dev`` are the two coefficients which are passed to the
    initializer of this class. The components of ``J`` are ::

        J_ijkl = δ_ij * δ_kl / d,

    where ``δ_ij`` denotes the Kronecker symbol, and ``d`` is the
    dimension of the space on which the tensor operates. The components
    of ``K`` are found from the identity ``K = I - J``, where ``I`` is
    the fourth-rank identity tensor ::

        I_ijkl = (δ_ik * δ_jl + δ_il * δ_jk) / 2.

    """

    @cdivision(True)
    def __cinit__(self, double sph, double dev, int dim):
        cdef int sym = (dim * (dim + 1)) / 2
        self.init_sizes(sym, sym)
        self.dim = dim
        self.sph = sph
        self.dev = dev
        self.tr = (sph - dev) / <double> self.dim

    def __repr__(self):
        return ('FourthRankIsotropicTensor(sph={0}, dev={1}, dim={2})'
                .format(self.sph, self.dev, self.dim))


cdef class FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):

    """
    Specialization of :class:`FourthRankIsotropicTensor` to 2D.

    The present implementation allows for
    :ref:`in-place operations <in-place-operations>`.

    """

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double self_dev = self.dev
        cdef double x0 = x[0]
        cdef double x1 = x[1]
        cdef double aux = self.tr * (x0 + x1)
        y[0] = aux + self_dev * x0
        y[1] = aux + self_dev * x1
        y[2] = self_dev * x[2]

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_to_memoryview(self, double[:, :] out):
        cdef double aux = (self.sph + self.dev) / 2.0
        out[0, 0] = aux
        out[1, 1] = aux
        aux = 0.5 * (self.sph - self.dev)
        out[0, 1] = aux
        out[1, 0] = aux
        out[2, 2] = self.dev
        out[2, 0] = 0.0
        out[2, 1] = 0.0
        out[0, 2] = 0.0
        out[1, 2] = 0.0


cdef class FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):

    """
    Specialization of :class:`FourthRankIsotropicTensor` to 3D.


    The present implementation allows for
    :ref:`in-place operations <in-place-operations>`.

    """

    @boundscheck(False)
    @wraparound(False)
    cdef inline void c_apply(self, double[:] x, double[:] y):
        cdef double self_dev = self.dev
        cdef double x0 = x[0]
        cdef double x1 = x[1]
        cdef double x2 = x[2]
        cdef double aux = self.tr * (x0 + x1 + x2)
        y[0] = aux + self_dev * x0
        y[1] = aux + self_dev * x1
        y[2] = aux + self_dev * x2
        y[3] = self_dev * x[3]
        y[4] = self_dev * x[4]
        y[5] = self_dev * x[5]

    @boundscheck(False)
    @cdivision(True)
    @wraparound(False)
    cdef void c_to_memoryview(self, double[:, :] out):
        cdef double aux = (self.sph + 2.0 * self.dev) / 3.0
        out[0, 0] = aux
        out[1, 1] = aux
        out[2, 2] = aux
        aux = (self.sph - self.dev) / 3.0
        out[0, 1] = aux
        out[0, 2] = aux
        out[1, 0] = aux
        out[1, 2] = aux
        out[2, 0] = aux
        out[2, 1] = aux
        out[3, 3] = self.dev
        out[4, 4] = self.dev
        out[5, 5] = self.dev
        out[0, 3] = 0.0
        out[0, 4] = 0.0
        out[0, 5] = 0.0
        out[1, 3] = 0.0
        out[1, 4] = 0.0
        out[1, 5] = 0.0
        out[2, 3] = 0.0
        out[2, 4] = 0.0
        out[2, 5] = 0.0
        out[3, 0] = 0.0
        out[3, 1] = 0.0
        out[3, 2] = 0.0
        out[3, 4] = 0.0
        out[3, 5] = 0.0
        out[4, 0] = 0.0
        out[4, 1] = 0.0
        out[4, 2] = 0.0
        out[4, 3] = 0.0
        out[4, 5] = 0.0
        out[5, 0] = 0.0
        out[5, 1] = 0.0
        out[5, 2] = 0.0
        out[5, 3] = 0.0
        out[5, 4] = 0.0


cdef class AbstractStructuredOperator2D:

    """Operator applied to vectorial data structured in a 2D grid.

    Structured operators have been defined
    :ref:`above <structured-operators>`; they map real vectors of
    dimension ``isize`` to real vectors of dimension ``osize``. Input
    and output vectors are structured in 2D grids, each grid-cell being
    a vector itself. Therefore, the input (resp. output) is a
    (``float64``) memoryview of shape ``(shape0, shape1, ishape2)``
    [resp. ``(shape0, shape1, oshape2)``], with ::

        isize = shape0 * shape1 * ishape2
        osize = shape0 * shape1 * oshape2

    Attributes
    ----------
    dim : int
        The dimension of the structured grid (``dim == 2``).
    shape0 : int
        The first dimension of the input and output.
    shape1 : int
        The second dimension of the input and output.
    ishape2 : int
        The third dimension of the input.
    oshape2 : int
        The third dimension of the output.
    ishape : tuple
        The shape of the input [the tuple ``(shape0, shape1, ishape2)``].
    oshape : tuple
        The shape of the output [the tuple ``(shape0, shape1, oshape2)``].

    """

    def __cinit__(self):
        self.dim = 2

    def init_shapes(self, int shape0, int shape1, int ishape2, int oshape2):
        """init_shapes(shape0, shape1, ishape2, oshape2)

        Initialize the values of the `ishape`, `oshape`, `shape0`,
        `shape1`, `ishape2` and `oshape2` attributes.

        This method is provided as a convenience to users who want to
        create pure Python implementations of this class. It should be
        called only *once* at the initialization of the instance, as it
        can potentially modify these attributes (which are otherwise
        read-only).

        Parameters
        ----------
        shape0 : int
            The first dimension of the input and output.
        shape1 : int
            The second dimension of the input and output.
        ishape2 : int
            The third dimension of the input.
        oshape2 : int
            The third dimension of the output.

        """
        if shape0 <= 0:
            raise ValueError('shape0 should be > 0 (was {})'.format(shape0))
        if shape1 <= 0:
            raise ValueError('shape1 should be > 0 (was {})'.format(shape1))
        if ishape2 <= 0:
            raise ValueError('ishape2 should be > 0 (was {})'.format(ishape2))
        if oshape2 <= 0:
            raise ValueError('oshape2 should be > 0 (was {})'.format(oshape2))
        self.shape0 = shape0
        self.shape1 = shape1
        self.ishape2 = ishape2
        self.oshape2 = oshape2
        self.ishape = (shape0, shape1, ishape2)
        self.oshape = (shape0, shape1, oshape2)

    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        pass

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
        check_shape_3d(x, self.shape0, self.shape1, self.ishape2)
        y = create_or_check_shape_3d(y, self.shape0, self.shape1,
                                     self.oshape2)
        self.c_apply(x, y)
        return y


cdef class AbstractStructuredOperator3D:

    """Operator applied to vectorial data structured in a 3D grid.

    Structured operators have been defined
    :ref:`above <structured-operators>`; they map real vectors of
    dimension ``isize`` to real vectors of dimension ``osize``. Input
    and output vectors are structured in 3D grids, each grid-cell being
    a vector itself. Therefore, the input (resp. output) is a
    (``float64``) memoryview of shape
    ``(shape0, shape1, shape2, ishape3)``
    [resp. ``(shape0, shape1, shape2, shape3)``], with ::

        isize = shape0 * shape1 * shape2 * ishape3
        osize = shape0 * shape1 * shape2 * oshape3

    Attributes
    ----------
    dim : int
        The dimension of the structured grid (``dim == 2``).
    shape0 : int
        The first dimension of the input and output.
    shape1 : int
        The second dimension of the input and output.
    shape2 : int
        The third dimension of the input and output.
    ishape3 : int
        The fourth dimension of the input.
    oshape3 : int
        The fourth dimension of the output.
    ishape : tuple
        The shape of the input [the tuple
        ``(shape0, shape1, shape2, ishape3)``].
    oshape : tuple
        The shape of the output [the tuple
        ``(shape0, shape1, shape2, oshape3)``].

    """

    def __cinit__(self):
        self.dim = 3

    def init_shapes(self, int shape0, int shape1, int shape2,
                    int ishape3, int oshape3):
        """init_shapes(shape0, shape1, shape2, ishape3, oshape3)

        Initialize the values of the `ishape`, `oshape`, `shape0`,
        `shape1`, `shape2`, `ishape3` and `oshape3` attributes.

        This method is provided as a convenience to users who want to
        create pure Python implementations of this class. It should be
        called only *once* at the initialization of the instance, as it
        can potentially modify these attributes (which are otherwise
        read-only).

        Parameters
        ----------
        shape0 : int
            The first dimension of the input and output.
        shape1 : int
            The second dimension of the input and output.
        shape2 : int
            The third dimension of the input and output.
        ishape3 : int
            The fourth dimension of the input.
        oshape3 : int
            The fourth dimension of the output.

        """
        if shape0 <= 0:
            raise ValueError('shape0 should be > 0 (was {})'.format(shape0))
        if shape1 <= 0:
            raise ValueError('shape1 should be > 0 (was {})'.format(shape1))
        if shape2 <= 0:
            raise ValueError('shape2 should be > 0 (was {})'.format(shape2))
        if ishape3 <= 0:
            raise ValueError('ishape3 should be > 0 (was {})'.format(ishape3))
        if oshape3 <= 0:
            raise ValueError('oshape3 should be > 0 (was {})'.format(oshape3))
        self.shape0 = shape0
        self.shape1 = shape1
        self.shape2  =shape2
        self.ishape3 = ishape3
        self.oshape3 = oshape3
        self.ishape = (shape0, shape1, shape2, ishape3)
        self.oshape = (shape0, shape1, shape2, oshape3)

    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y):
        pass

    def apply(self, double[:, :, :, :] x, double[:, :, :, :] y=None):
        """apply(x, y=None)

        Return the result of applying this operator to `x`.

        The default implementation calls the (Cython) method
        ``c_apply()``.

        Parameters
        ----------
        x : 4D memoryview of float64
            The input vector.
        y : 4D memoryview of float64, optional
            The output vector. Its content is altered by this method. If
            ``None``, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        y : 4D memoryview of float64
            The result of applying this operator to `x`. This is a
            view of the parameter `y` (if not ``None``).

        """
        check_shape_4d(x,
                       self.shape0, self.shape1,
                       self.shape2, self.ishape3)
        y = create_or_check_shape_4d(y,
                                     self.shape0, self.shape1,
                                     self.shape2, self.oshape3)
        self.c_apply(x, y)
        return y


cdef class BlockDiagonalOperator2D(AbstractStructuredOperator2D):

    """Block-diagonal operator with 2D layout of the (vectorial) data.

    Block-diagonal operators have been defined
    :ref:`above <block-diagonal-operators>`.

    If the local operators `loc` allow for :ref:`in-place operations
    <in-place-operations>`, then the block-diagonal operator also
    allows for in-place operations.

    Parameters
    ----------
    loc : 2D memoryview of :class:`AbstractOperator`
        The array of local operators. The returned instance keeps a
        *shallow* copy of `loc`.

    """

    def __cinit__(self, AbstractOperator[:, :] loc):
        if loc is None:
            raise ValueError('loc should not be None')
        self.init_shapes(loc.shape[0], loc.shape[1],
                         loc[0, 0].isize, loc[0, 0].osize)
        self.loc = loc

        cdef int i0, i1, ishape2, oshape2
        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                ishape2 = loc[i0, i1].isize
                oshape2 = loc[i0, i1].osize
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
        cdef AbstractOperator op

        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                op = self.loc[i0, i1]
                op.c_apply(x[i0, i1, :], y[i0, i1, :])


cdef class BlockDiagonalOperator3D(AbstractStructuredOperator3D):

    """Block-diagonal operator with 3D layout of the (vectorial) data.

    Block-diagonal operators have been defined
    :ref:`above <block-diagonal-operators>`.


    If the local operators `loc` allow for :ref:`in-place operations
    <in-place-operations>`, then the block-diagonal operator also
    allows for in-place operations.

    Parameters
    ----------
    loc : 3D memoryview of :class:`AbstractOperator`
        The array of local operators. The returned instance keeps a
        *shallow* copy of `loc`.

    """

    def __cinit__(self, AbstractOperator[:, :, :] loc):
        if loc is None:
            raise ValueError('loc should not be None')
        self.init_shapes(loc.shape[0], loc.shape[1], loc.shape[2],
                         loc[0, 0, 0].isize, loc[0, 0, 0].osize)
        self.loc = loc

        cdef int i0, i1, i2, ishape3, oshape3
        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                for i2 in range(self.shape2):
                    ishape3 = loc[i0, i1, i2].isize
                    oshape3 = loc[i0, i1, i2].osize
                    if ishape3 != self.ishape3:
                        raise ValueError('invalid dimension block operator '
                                         'input: expected {0}, '
                                         'actual {1}'.format(self.ishape3,
                                                             ishape3))
                    if oshape3 != self.oshape3:
                        raise ValueError('invalid dimension block operator '
                                         'output: expected {0}, '
                                         'actual {1}'.format(self.oshape3,
                                                             oshape3))

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y):
        cdef int i0, i1, i2
        cdef AbstractOperator op

        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                for i2 in range(self.shape2):
                    op = self.loc[i0, i1, i2]
                    op.c_apply(x[i0, i1, i2, :], y[i0, i1, i2, :])


cdef class BlockDiagonalLinearOperator2D(AbstractStructuredOperator2D):

    """Block-diagonal linear operator with 2D layout of the data.

    Such operators have been defined
    :ref:`above <block-diagonal-linear-operators>`.

    Parameters
    ----------
    a : 4D memoryview of ``float64``
        The array of local matrices. The returned instance keeps a
        *shallow* copy of `a`.

    """

    def __cinit__(self, double[:, :, :, :] a):
        if a is None:
            raise ValueError('a should not be None')
        self.init_shapes(a.shape[0], a.shape[1], a.shape[3], a.shape[2])
        self.a = a

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int i0, i1, i2, j2
        cdef double[:, :] a_loc
        cdef double[:] x_loc
        cdef double[:] y_loc

        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                a_loc = self.a[i0, i1, :, :]
                x_loc = x[i0, i1, :]
                y_loc = y[i0, i1, :]
                for i2 in range(self.oshape2):
                    yy = 0.
                    for j2 in range(self.ishape2):
                        yy += a_loc[i2, j2] * x_loc[j2]
                    y_loc[i2] = yy

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply_transpose(self, double[:, :, :] x, double[:, :, :] y):
        cdef int i0, i1, i2, j2
        cdef double[:, :] a_loc
        cdef double[:] x_loc
        cdef double[:] y_loc

        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                a_loc = self.a[i0, i1, :, :]
                x_loc = x[i0, i1, :]
                y_loc = y[i0, i1, :]
                for i2 in range(self.ishape2):
                    yy = 0.
                    for j2 in range(self.oshape2):
                        yy += a_loc[j2, i2] * x_loc[j2]
                    y_loc[i2] = yy

    def apply_transpose(self, double[:, :, :] x, double[:, :, :] y=None):
        """apply_transpose(x, y=None)

        Return the result of applying the transpose of this operator to `x`.

        The output is defined as follows ::

            y[i0, i1, i2] = sum(a[i0, i1, j2, i2] * x[i0, i1, j2], j2)

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
        check_shape_3d(x, self.shape0, self.shape1, self.oshape2)
        y = create_or_check_shape_3d(y, self.shape0, self.shape1,
                                     self.ishape2)
        self.c_apply_transpose(x, y)
        return y


cdef class BlockDiagonalLinearOperator3D(AbstractStructuredOperator3D):

    """Block-diagonal linear operator with 3D layout of the data.

    Such operators have been defined
    :ref:`above <block-diagonal-linear-operators>`.

    Parameters
    ----------
    a : 5D memoryview of ``float64``
        The array of local matrices. The returned instance keeps a
        *shallow* copy of `a`.

    """

    def __cinit__(self, double[:, :, :, :, :] a):
        if a is None:
            raise ValueError('a should not be None')
        self.init_shapes(a.shape[0], a.shape[1], a.shape[2],
                         a.shape[4], a.shape[3])
        self.a = a

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y):
        cdef int i0, i1, i2, i3, j3
        cdef double[:, :] a_loc
        cdef double[:] x_loc
        cdef double[:] y_loc

        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                for i2 in range(self.shape2):
                    a_loc = self.a[i0, i1, i2, :, :]
                    x_loc = x[i0, i1, i2, :]
                    y_loc = y[i0, i1, i2, :]
                    for i3 in range(self.oshape3):
                        yy = 0.
                        for j3 in range(self.ishape3):
                            yy += a_loc[i3, j3] * x_loc[j3]
                        y_loc[i3] = yy

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply_transpose(self, double[:, :, :, :] x,
                                double[:, :, :, :] y):
        cdef int i0, i1, i2, i3, j3
        cdef double[:, :] a_loc
        cdef double[:] x_loc
        cdef double[:] y_loc

        for i0 in range(self.shape0):
            for i1 in range(self.shape1):
                for i2 in range(self.shape2):
                    a_loc = self.a[i0, i1, i2, :, :]
                    x_loc = x[i0, i1, i2, :]
                    y_loc = y[i0, i1, i2, :]
                    for i3 in range(self.ishape3):
                        yy = 0.
                        for j3 in range(self.oshape3):
                            yy += a_loc[j3, i3] * x_loc[j3]
                        y_loc[i3] = yy

    def apply_transpose(self, double[:, :, :, :] x, double[:, :, :, :] y=None):
        """apply_transpose(x, y=None)

        Return the result of applying the transpose of this operator to `x`.

        The output is defined as follows ::

            y[i0, i1, i2, i3] = sum(a[i0, i1, i2, j3, i3] *
                                    x[i0, i1, i2, j3], j3)

        Parameters
        ----------
        x : 4D memoryview of float64
            The input vector.
        y : 4D memoryview of float64, optional
            The output vector. Its content is altered by this method. If
            ``None``, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        y : 4D memoryview of float64
            The result of applying this operator to `x`. This is a
            view of the parameter `y` (if not ``None``).

        """
        check_shape_4d(x,
                       self.shape0, self.shape1,
                       self.shape2, self.oshape3)
        y = create_or_check_shape_4d(y,
                                     self.shape0, self.shape1,
                                     self.shape2, self.ishape3)
        self.c_apply_transpose(x, y)
        return y
