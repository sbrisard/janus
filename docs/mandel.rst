.. -*- coding: utf-8-unix -*-

******************
On Mandel notation
******************

Janus makes havy use of Mandel's representation of symmetric, second rank tensors as column vectors, and fourth rank tensors with minor symmetries as matrices. This representation is defined below. Furthermore, the properties of the matrix representation are summarized (see also `Wikipedia <https://en.wikipedia.org/wiki/Voigt_notation#Mandel_notation>`_).

Mandel notation in 3D
=====================

In this section, the notation is introduced for tensors of the three dimensional space.

Second rank, symmetric tensors
------------------------------

Let :math:`\tens\strain` be a second rank, symmetric tensor

.. math::

   \strain_{ij}=\strain_{ji}.

Its Mandel representation :math:`[\tens\strain]` (as a column-vector) is defined as follows

.. math::

   [\tens\strain] = [
       \strain_{11}, \strain_{22}, \strain_{33},
       \sqrt2\strain_{23}, \sqrt2\strain_{31}, \sqrt2\strain_{12}
   ]^\T,

where the cross-component :math:`\strain_{ij}` (:math:`i\neq j`) appears at the :math:`3+k`-th line, with :math:`k\neq i\neq j`. The :math:`\sqrt 2` prefactors ensure that the standard scalar product of column vectors coincides with the double contraction of tensors. Indeed

.. math::
   :nowrap:

   \begin{align*}
   \tens\stress:\tens\strain&=\stress_{ij}\strain_{ij},\\
   &=\stress_{11}\strain_{11}+\stress_{22}\strain_{22}+\stress_{33}\strain_{33}+2\stress_{23}\strain_{23}+2\stress_{31}\strain_{31}+2\stress_{12}\strain_{12},\\
   &=[\tens\stress]^\T\cdot[\tens\strain].
   \end{align*}

Fourth rank tensors with minor symmetries
-----------------------------------------

Let :math:`\tens[4]\stiffness` be a fourth rank tensor with minor symmetries

.. math::

   \stiffness_{ijkl}=\stiffness_{jikl}=\stiffness_{ijlk}.

Its Mandel representation :math:`[\tens[4]\stiffness]` (as a square matrix) is defined as follows

.. math::

   [\tens[4]\stiffness] =
   \begin{bmatrix}
       \stiffness_{1111} & \stiffness_{1122} & \stiffness_{1133} & \sqrt{2}\stiffness_{1123} & \sqrt{2}\stiffness_{1131} & \sqrt{2}\stiffness_{1112}\\
       \stiffness_{2211} & \stiffness_{2222} & \stiffness_{2233} & \sqrt{2}\stiffness_{2223} & \sqrt{2}\stiffness_{2231} & \sqrt{2}\stiffness_{2212}\\
       \stiffness_{3311} & \stiffness_{3322} & \stiffness_{3333} & \sqrt{2}\stiffness_{3323} & \sqrt{2}\stiffness_{3331} & \sqrt{2}\stiffness_{3312}\\
       \sqrt{2}\stiffness_{2311} & \sqrt{2}\stiffness_{2322} & \sqrt{2}\stiffness_{2333} & 2\stiffness_{2323} & 2\stiffness_{2331} & 2\stiffness_{2312}\\
       \sqrt{2}\stiffness_{3111} & \sqrt{2}\stiffness_{3122} & \sqrt{2}\stiffness_{3133} & 2\stiffness_{3123} & 2\stiffness_{3131} & 2\stiffness_{3112}\\
       \sqrt{2}\stiffness_{1211} & \sqrt{2}\stiffness_{1222} & \sqrt{2}\stiffness_{1233} & 2\stiffness_{1223} & 2\stiffness_{1231} & 2\stiffness_{1212}
   \end{bmatrix},

where the numbering of the cross-components :math:`\stiffness_{ijkl}` with :math:`i\neq j` or :math:`k\neq l` is consistent with the numbering of cross-components of second rank tensors. Again, the :math:`\sqrt 2` and 2 prefactors ensure that matrix-matrix and matrix-vector products coincide with the double contraction of tensors.

More precisely, the Mandel representation of the second rank tensor :math:`\tens\stress=\tens[4]\stiffness:\tens\strain` is the column vector

.. math::
   [\tens\stress]=[\tens[4]\stiffness:\tens\strain]=\stiffness_{ijkl}\strain_{kl}=[\tens\stiffness]\cdot[\tens\strain].

Likewise, if :math:`\tens[4]\Compliance` is another fourth rank tensor with minor symmetries, then

.. math::
   [\tens[4]\stiffness:\tens[4]\Compliance]=[\tens[4]\stiffness]\cdot[\tens[4]\Compliance],

where it is recalled that the (i, j, k, l) component of :math:`\tens\stiffness:\tens\Compliance` is :math:`\stiffness_{ijmn}\Compliance_{mnkl}`. It results from the above formula that the Mandel representation of the inverse of a fourth rank tensor is the inverse of the Mandel representation of this tensor

.. math::
   [\tens[4]\stiffness^{-1}]=[\tens[4]\stiffness]^{-1}.

Finally, it is readily verified that the Mandel representation of the transpose is the transpose of the Mandel representation

.. math::
   [\tens[4]\stiffness^\T]=[\tens[4]\stiffness]^\T.

Mandel notation in 2D
=====================

The above formulas are readily extended to two dimensions, so that we only recall the matrix representation of second rank, symmetric tensors and fourth rank tensors with minor symmetries. The properties of these matrix representations are unchanged.

Second rank, symmetric tensors
------------------------------

The Mandel representation :math:`[\tens\strain]` (as a column-vector) of any second rank, symmetric tensor :math:`\tens\strain` is defined as follows

.. math::

   [\tens\strain] = [\strain_{11}, \strain_{22}, \sqrt2\strain_{12}]^\T.

Fourth rank tensors with minor symmetries
-----------------------------------------

The Mandel representation :math:`[\tens[4]\stiffness]` (as a square matrix) of any fourth rank tensor :math:`\tens[4]\stiffness` with minor symmetries is defined as follows

.. math::

   [\tens[4]\stiffness] =
   \begin{bmatrix}
       \stiffness_{1111} & \stiffness_{1122} & \sqrt{2}\stiffness_{1112}\\
       \stiffness_{2211} & \stiffness_{2222} & \sqrt{2}\stiffness_{2212}\\
       \sqrt{2}\stiffness_{1211} & \sqrt{2}\stiffness_{1222} & 2\stiffness_{1212}
   \end{bmatrix}.
