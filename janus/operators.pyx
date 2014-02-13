# -*- coding: utf-8 -*-

"""Classes defining operators (:mod:`janus.operators`)
===================================================

The (concrete and abstract) classes defined in this module define
operators in the most general sense, as a mapping from a (real) vector
space to another. If the mapping is *linear*, then the resulting
operator is a *tensor*.

.. _structured-operators:

Beyond general operators (:class:`Operator`), this module also
introduces *structured* operators
(:class:`AbstractStructuredOperator2D`,
:class:`AbstractStructuredOperator3D`), for wich the input and output of
the operators are structured in n-dimensional grids. The content of each
cell might be tensorial, so that the input and output of 2D structured
operators are *3-dimensional* arrays. Likewise, the input and output
arrays of 3D structured operators are *4-dimensional arrays*. The
*local* input and output then refer to data contained in one specific
cell. For example, let ``x[:, :, :]`` (resp. ``y[:, :, :]``) be the
input (resp. output) of a 2D structured operator; the local input (resp.
output) of cell ``(i0, i1)`` is the array ``x[i0, i1, :]`` (resp
``y[i0, i1, :]``). Extension of these definitions to higher spatial
dimensions is straightforward.

.. _block-diagonal-operators:

*Block-diagonal* operators (:class:`BlockDiagonalOperator2D`,
:class:`BlockDiagonalOperator3D`) are defined as structured operators
for which the local output depends on the local input only. Such
operators can be represented as arrays of local operators. For example,
a 2D block-diagonal operator can be defined through an array
``op_loc[:, :]``, of :class:`Operator` such that the input
``x[:, :, :]`` is mapped to the output ``y[:, :, :]`` as follows::

    y[i0, i1, :] = op_loc[i0, i1].apply(x[i0, i1, :])

(extension to higher spatial dimensions being trivial)

.. _block-diagonal-linear-operators:

This can be further simplified in the case of linear, block-diagonal
operators (:class:`BlockDiagonalLinearOperator2D`,
:class:`BlockDiagonalLinearOperator3D`). Indeed, ``op_loc`` is then an
array of matrices, which can be viewed as a higher-dimension array.
Therefore, a 2D, block-diagonal linear operator can be defined through
a four-dimensional array ``a[:, :, :, :]`` such that::

    y[i0, i1, i2] = sum(a[i0, i1, i2, j2] * x[i0, i1, j2], j2)

(extension to higher spatial dimensions being trivial)

Functions:

- :func:`isotropic_4` -- create a fourth-rank, isotropic tensor with minor
        symmetries.

Classes:

- :class:`Operator` -- general operator
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

    Create a fourth rank, isotrpic tensor.

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


cdef class Operator:

    """General operator.

    Attributes
    ----------
    isize : int
        The dimension of the input of the operator.
    osize : int
        The dimension of the output of the operator.

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
        check_shape_1d(x, self.isize)
        y = create_or_check_shape_1d(y, self.osize)
        self.c_apply(x, y)
        return y


cdef class AbstractLinearOperator(Operator):

    """Specialization of :class:`Operator` to linear operators.

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
            ``None``, a new array is created and returned; otherwise,
            this method returns a view of this object.

        Returns
        -------
        out : 2D memoryview of float64
            The matrix of this operator. This is a view of the parameter
            `out` (if not ``None``).

        """
        out = create_or_check_shape_2d(out, self.osize, self.isize)
        self.c_to_memoryview(out)
        return out


cdef class FourthRankIsotropicTensor(Operator):

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
    Any fourth rank, isotropic tensor `T` is a linear combination of the
    spherical projection tensor `J` and the deviatoric projection tensor
    `K`::

        T = sph * J + dev * K.

    `sph` and `dev` are the two coefficients which are passed to the
    constructor of this class. The components of `J` are::

        J_ijkl = δ_ij * δ_kl / d,

    where ``δ_ij`` denotes the Kronecker symbol, and `d` is the
    dimension of the space on which the tensor operates. The components
    of `K` are found from the identity ``K = I - J``, where `I` is the
    fourth-rank identity tensor::

        I_ijkl = (δ_ik * δ_jl + δ_il * δ_jk) / 2.

    """

    @cdivision(True)
    def __cinit__(self, double sph, double dev, int dim):
        self.dim = dim
        self.isize = (self.dim * (self.dim + 1)) / 2
        self.osize = self.isize
        self.sph = sph
        self.dev = dev
        self.tr = (sph - dev) / <double> self.dim

    def __repr__(self):
        return ('FourthRankIsotropicTensor(sph={0}, dev={1}, dim={2})'
                .format(self.sph, self.dev, self.dim))


cdef class FourthRankIsotropicTensor2D(FourthRankIsotropicTensor):

    """Specialization of :class:`FourthRankIsotropicTensor` to 2D."""

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


cdef class FourthRankIsotropicTensor3D(FourthRankIsotropicTensor):

    """Specialization of :class:`FourthRankIsotropicTensor` to 3D."""

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


cdef class AbstractStructuredOperator2D:

    """Operator applied to vectorial data structured in a 2D grid.

    Structured operators have been defined
    :ref:`above <structured-operators>`; they map real vectors of
    dimension ``isize`` to real vectors of dimension ``osize``. Input
    and output vectors are structured in 2D grids, each grid-cell being
    a vector itself. Therefore, the input (resp. output) is a
    (``float64``) memoryview of shape ``(ishape0, ishape1, ishape2)``
    [resp. ``(oshape0, oshape1, oshape2)``], with::

        isize = ishape0 * ishape1 * ishape2
        osize = oshape0 * oshape1 * oshape2

    Attributes
    ----------
    dim : int
        The dimension of the structured grid (``dim == 2``).
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
        self.dim = 2

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


cdef class AbstractStructuredOperator3D:

    """Operator applied to vectorial data structured in a 3D grid.

    Structured operators have been defined
    :ref:`above <structured-operators>`; they map real vectors of
    dimension ``isize`` to real vectors of dimension ``osize``. Input
    and output vectors are structured in 3D grids, each grid-cell being
    a vector itself. Therefore, the input (resp. output) is a
    (``float64``) memoryview of shape
    ``(ishape0, ishape1, ishape2, ishape3)``
    [resp. ``(oshape0, oshape1, oshape2, oshape3)``], with::

        isize = ishape0 * ishape1 * ishape2 * ishape3
        osize = oshape0 * oshape1 * oshape2 * oshape3

    Attributes
    ----------
    dim : int
        The dimension of the structured grid (``dim == 2``).
    ishape0 : int
        The first dimension of the input.
    ishape1 : int
        The second dimension of the input.
    ishape2 : int
        The third dimension of the input.
    ishape3 : int
        The fourth dimension of the input.
    oshape0 : int
        The first dimension of the output.
    oshape1 : int
        The second dimension of the output.
    oshape2 : int
        The third dimension of the output.
    oshape3 : int
        The fourth dimension of the output.
    ishape : tuple
        The shape of the input [the tuple
        ``(ishape0, ishape1, ishape2, ishape3)``].
    oshape : tuple
        The shape of the output [the tuple
        ``(oshape0, oshape1, oshape2, oshape3)``].

    """

    def __cinit__(self):
        self.dim = 3

    cdef void c_apply(self, double[:, :, :, :] x, double[:, :, :, :] y):
        raise NotImplementedError

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
                       self.ishape0, self.ishape1,
                       self.ishape2, self.ishape3)
        y = create_or_check_shape_4d(y,
                                     self.oshape0, self.oshape1,
                                     self.oshape2, self.oshape3)
        self.c_apply(x, y)
        return y


cdef class BlockDiagonalOperator2D(AbstractStructuredOperator2D):

    """Block-diagonal operator with 2D layout of the (vectorial) data.

    Block-diagonal operators have been defined
    :ref:`above <block-diagonal-operators>`.

    TODO -- all local operators must have same input and output dimensions

    TODO -- what is the ishape and oshape of such an operator?

    Parameters
    ----------
    op_loc : 2D memoryview of :class:`Operator`
        The array of local operators.

    """

    def __cinit__(self, Operator[:, :] op_loc):
        self.ishape0 = op_loc.shape[0]
        self.ishape1 = op_loc.shape[1]
        self.ishape2 = op_loc[0, 0].isize
        self.oshape0 = op_loc.shape[0]
        self.oshape1 = op_loc.shape[1]
        self.oshape2 = op_loc[0, 0].osize
        self.ishape = (self.ishape0, self.ishape1, self.ishape2)
        self.oshape = (self.oshape0, self.oshape1, self.oshape2)
        self.op_loc = op_loc.copy()

        cdef int i0, i1, ishape2, oshape2
        for i0 in range(self.ishape0):
            for i1 in range(self.ishape1):
                ishape2 = op_loc[i0, i1].isize
                oshape2 = op_loc[i0, i1].osize
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
                op = self.op_loc[i0, i1]
                op.c_apply(x[i0, i1, :], y[i0, i1, :])


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
        self.ishape0 = a.shape[0]
        self.ishape1 = a.shape[1]
        self.ishape2 = a.shape[3]
        self.oshape0 = a.shape[0]
        self.oshape1 = a.shape[1]
        self.oshape2 = a.shape[2]
        self.ishape = (self.ishape0, self.ishape1, self.ishape2)
        self.oshape = (self.oshape0, self.oshape1, self.oshape2)
        self.a = a

    @boundscheck(False)
    @wraparound(False)
    cdef void c_apply(self, double[:, :, :] x, double[:, :, :] y):
        cdef int i0, i1
        cdef double[:, :] a_loc
        cdef double[:] x_loc
        cdef double[:] y_loc

        for i0 in range(self.ishape0):
            for i1 in range(self.ishape1):
                a_loc = self.a_loc[i0, i1, :, :]
                x_loc = x[i0, i1, :]
                y_loc = y[i0, i1, :]
                for i2 in range(self.oshape2):
                    yy = 0.
                    for j2 in range(self.ishape2):
                        yy += a_loc[i2, j2] * x_loc[j2]
                    y_loc[i2] = yy
