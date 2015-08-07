.. -*- coding: utf-8-unix -*-

*********
Operators
*********

Classes and functions for the definition of operators are provided in the :mod:`janus.operators` module. Operator should be understood in the most general sense, as a mapping from a (real) vector space to another. If the mapping is *linear*, then the resulting operator is a *tensor*.

The root of the hierarchy tree of operators is the class :class:`AbstractOperator <janus.operators.AbstractOperator>`. Instances of this class have two attributes, ``isize`` and ``osize`` which are the sizes of the input and output of the operator, respectively (in other words, these attributes are the dimensions of the domain and codomain, respectively).

Operator ``op`` is mapped to the vector ``x`` through the method :meth:`AbstractOperator.apply <janus.operators.AbstractOperator.apply>`::

    y = op.apply(x)

where ``x`` (resp. ``y``) is a 1D array of length ``op.isize`` (resp. ``op.osize``).

.. _structured-operators:

Structured operators
====================

Structured operators are operators whose input and output are structured in multi-dimensional grids. The content of each cell might be tensorial, so that the input and output of 2D structured operators are *3-dimensional* arrays. Likewise, the input and output arrays of 3D structured operators are *4-dimensional* arrays. The *local* input and output then refer to data contained in one specific cell. For example, let ``x[:, :, :]`` (resp. ``y[:, :, :]``) be the input (resp. output) of a 2D structured operator; the local input (resp. output) of cell ``(i0, i1)`` is the 1D array ``x[i0, i1, :]`` (resp ``y[i0, i1, :]``).

Two- and three- dimensional structured operators are defined in this module through the classes :class:`AbstractStructuredOperator2D <janus.operators.AbstractStructuredOperator2D>` and :class:`AbstractStructuredOperator3D <janus.operators.AbstractStructuredOperator3D>`). Instances of these classes have two attributes, ``ishape`` and ``oshape`` which are the shapes (tuples of dimensions) of the input and output of the operator, respectively. It should be noted that the data layout (the dimensions of the *spatial* grid) of the input and output are identical. In other words::

  op.ishape[:-1] == op.oshape[:-1]

for any structured operator ``op``. Of course, ``op.ishape[-1]`` and ``op.oshape[-1]`` may differ. Structured operators are applied to multidimensional arrays as follows::

  y = op.apply(x)

where ``x.shape == op.ishape`` and ``y.shape == op.oshape``.

.. _block-diagonal-operators:

Block-diagonal operators
========================

Block-diagonal operators (:class:`BlockDiagonalOperator2D <janus.operators.BlockDiagonalOperator2D>`, :class:`BlockDiagonalOperator3D <janus.operators.BlockDiagonalOperator3D>`) are defined as structured operators for which the local output depends on the local input only. Any block diagonal operator can be represented as an array of local operators (of type :class:`AbstractOperator <janus.operators.AbstractOperator>`) ``loc``. Then, the input ``x`` is mapped to the output ``y`` as follows ::

    y[i0, i1, :] = loc[i0, i1].apply(x[i0, i1, :])

in 2D, and ::

    y[i0, i1, i2, :] = loc[i0, i1, i2].apply(x[i0, i1, i2, :])

in 3D.

.. _block-diagonal-linear-operators:

Block-diagonal linear operators
-------------------------------

This can be further simplified in the case of linear, block-diagonal operators (:class:`BlockDiagonalLinearOperator2D <janus.operators.BlockDiagonalLinearOperator2D>`, :class:`BlockDiagonalLinearOperator3D <janus.operators.BlockDiagonalLinearOperator3D>`). Indeed, ``loc`` is then an array of matrices, which can be viewed as a higher-dimension array. Therefore, a block-diagonal linear operator can be defined through a `float64` array ``a`` such that ::

    y[i0, i1, i2] = sum(a[i0, i1, i2, j2] * x[i0, i1, j2], j2)

in 2D, and ::

    y[i0, i1, i2, i3] = sum(a[i0, i1, i2, i3, j3] * x[i0, i1, i2, j3], j3)

in 3D.

Block-diagonal linear operators are created with the function :func:`block_diagonal_linear_operator <janus.operators.block_diagonal_linear_operator>`, which takes as an input an array, whose last two dimensions correspond to the matrix of the local operator.

>>> import numpy as np
>>> import janus.operators as operators
>>> a = np.arange(120., dtype=np.float64).reshape(2, 3, 4, 5)
>>> op = operators.block_diagonal_linear_operator(a)
>>> x = np.arange(30., dtype=np.float64).reshape(2, 3, 5)
>>> y = op.apply(x)
>>> yy = np.sum(a * x[:, :, np.newaxis, :], axis=-1)
>>> np.sqrt(np.sum((yy - y)**2))
0.0

.. _in-place-operations:

Performing in-place operations
==============================

All types of operators define a method ``apply(x, y)``, where ``x`` is a memoryview of the input and ``y`` is a memoryview of the output. If ``y`` is ``None``, then ``apply`` returns a newly created memoryview. If ``y`` is not ``None``, then ``apply`` returns a reference to ``y``.

Depending on the implementation, some operators allow for in-place operations, which can further reduce memory allocations. In other words, ``apply(x, x)`` is valid for such operators and returns the expected value. Whether or not an operator allows for in-place operations is implementation dependent, and should be specified in the documentation. **Unless otherwise stated, it should be assumed that in-place operations are not supported.**

If relevant, the above also applies to the Cython method ``c_apply(x, y)``.

API of module :mod:`janus.operators`
====================================

.. automodule:: janus.operators
   :members:
   :undoc-members:
   :show-inheritance:
