.. -*- coding: utf-8-unix -*-

.. _Mandel_notation:

***************
Mandel notation
***************

Janus makes havy use of Mandel's representation of symmetric, second rank tensors as column vectors, and fourth rank tensors with minor symmetries as matrices. This representation is defined below. Furthermore, the properties of the matrix representation are summarized (see also `Wikipedia <https://en.wikipedia.org/wiki/Voigt_notation#Mandel_notation>`_).

Mandel notation in 3D
=====================

In this section, the notation is introduced for tensors of the three dimensional space.

Second rank, symmetric tensors
------------------------------

Let :math:`\tens\varepsilon` be a second rank, symmetric tensor

.. math::

   \varepsilon_{ij}=\varepsilon_{ji}.

Its Mandel representation :math:`[\tens\varepsilon]` (as a column-vector) is defined as follows

.. math::

   [\tens\varepsilon] = [
       \varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33},
       \sqrt2\varepsilon_{23}, \sqrt2\varepsilon_{31}, \sqrt2\varepsilon_{12}
   ]^\T,

where the cross-component :math:`\varepsilon_{ij}` (:math:`i\neq j`) appears at the :math:`3+k`-th line, with :math:`k\neq i\neq j`. The :math:`\sqrt 2` prefactors ensure that the standard scalar product of column vectors coincides with the double contraction of tensors. Indeed

.. math::
   :nowrap:

   \begin{align*}
   \tens\sigma:\tens\varepsilon&=\sigma_{ij}\varepsilon_{ij},\\
   &=\sigma_{11}\varepsilon_{11}+\sigma_{22}\varepsilon_{22}+\sigma_{33}\varepsilon_{33}+2\sigma_{23}\varepsilon_{23}+2\sigma_{31}\varepsilon_{31}+2\sigma_{12}\varepsilon_{12},\\
   &=[\tens\sigma]^\T\cdot[\tens\varepsilon].
   \end{align*}

Fourth rank tensors with minor symmetries
-----------------------------------------

Let :math:`\tens[4]C` be a fourth rank tensor with minor symmetries

.. math::

   C_{ijkl}=C_{jikl}=C_{ijlk}.

Its Mandel representation :math:`[\tens[4]C]` (as a square matrix) is defined as follows

.. math::

   [\tens[4]C] =
   \begin{bmatrix}
       C_{1111} & C_{1122} & C_{1133} & \sqrt{2}C_{1123} & \sqrt{2}C_{1131} & \sqrt{2}C_{1112}\\
       C_{2211} & C_{2222} & C_{2233} & \sqrt{2}C_{2223} & \sqrt{2}C_{2231} & \sqrt{2}C_{2212}\\
       C_{3311} & C_{3322} & C_{3333} & \sqrt{2}C_{3323} & \sqrt{2}C_{3331} & \sqrt{2}C_{3312}\\
       \sqrt{2}C_{2311} & \sqrt{2}C_{2322} & \sqrt{2}C_{2333} & 2C_{2323} & 2C_{2331} & 2C_{2312}\\
       \sqrt{2}C_{3111} & \sqrt{2}C_{3122} & \sqrt{2}C_{3133} & 2C_{3123} & 2C_{3131} & 2C_{3112}\\
       \sqrt{2}C_{1211} & \sqrt{2}C_{1222} & \sqrt{2}C_{1233} & 2C_{1223} & 2C_{1231} & 2C_{1212}
   \end{bmatrix},

where the numbering of the cross-components :math:`C_{ijkl}` with :math:`i\neq j` or :math:`k\neq l` is consistent with the numbering of cross-components of second rank tensors. Again, the :math:`\sqrt 2` and 2 prefactors ensure that matrix-matrix and matrix-vector products coincide with the double contraction of tensors.

More precisely, the Mandel representation of the second rank tensor :math:`\tens\sigma=\tens[4]C:\tens\varepsilon` is the column vector

.. math::
   [\tens\sigma]=[\tens[4]C:\tens\varepsilon]=C_{ijkl}\varepsilon_{kl}=[\tens C]\cdot[\tens\varepsilon].

Likewise, if :math:`\tens[4]\Compliance` is another fourth rank tensor with minor symmetries, then

.. math::
   [\tens[4]C:\tens[4]\Compliance]=[\tens[4]C]\cdot[\tens[4]\Compliance],

where it is recalled that the (i, j, k, l) component of :math:`\tens C:\tens\Compliance` is :math:`C_{ijmn}\Compliance_{mnkl}`. It results from the above formula that the Mandel representation of the inverse of a fourth rank tensor is the inverse of the Mandel representation of this tensor

.. math::
   [\tens[4]C^{-1}]=[\tens[4]C]^{-1}.

Finally, it is readily verified that the Mandel representation of the transpose is the transpose of the Mandel representation

.. math::
   [\tens[4]C^\T]=[\tens[4]C]^\T.

Mandel notation in 2D
=====================

The above formulas are readily extended to two dimensions, so that we only recall the matrix representation of second rank, symmetric tensors and fourth rank tensors with minor symmetries. The properties of these matrix representations are unchanged.

Second rank, symmetric tensors
------------------------------

The Mandel representation :math:`[\tens\varepsilon]` (as a column-vector) of any second rank, symmetric tensor :math:`\tens\varepsilon` is defined as follows

.. math::

   [\tens\varepsilon] = [\varepsilon_{11}, \varepsilon_{22}, \sqrt2\varepsilon_{12}]^\T.

Fourth rank tensors with minor symmetries
-----------------------------------------

The Mandel representation :math:`[\tens[4]C]` (as a square matrix) of any fourth rank tensor :math:`\tens[4]C` with minor symmetries is defined as follows

.. math::

   [\tens[4]C] =
   \begin{bmatrix}
       C_{1111} & C_{1122} & \sqrt{2}C_{1112}\\
       C_{2211} & C_{2222} & \sqrt{2}C_{2212}\\
       \sqrt{2}C_{1211} & \sqrt{2}C_{1222} & 2C_{1212}
   \end{bmatrix}.
