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

Let :math:`\boldsymbol\varepsilon` be a second rank, symmetric tensor

.. math::

   \varepsilon_{ij}=\varepsilon_{ji}.

Its Mandel representation :math:`[\boldsymbol\varepsilon]` (as a column-vector) is defined as follows

.. math::

   [\boldsymbol\varepsilon] = [
       \varepsilon_{11}, \varepsilon_{22}, \varepsilon_{33},
       \sqrt2\varepsilon_{23}, \sqrt2\varepsilon_{31}, \sqrt2\varepsilon_{12}
   ]^\mathrm{T},

where the cross-component :math:`\varepsilon_{ij}` (:math:`i\neq j`) appears at the :math:`3+k`-th line, with :math:`k\neq i\neq j`. The :math:`\sqrt 2` prefactors ensure that the standard scalar product of column vectors coincides with the double contraction of tensors. Indeed

.. math::
   :nowrap:

   \begin{align*}
   \boldsymbol\sigma:\boldsymbol\varepsilon&=\sigma_{ij}\varepsilon_{ij},\\
   &=\sigma_{11}\varepsilon_{11}+\sigma_{22}\varepsilon_{22}+\sigma_{33}\varepsilon_{33}+2\sigma_{23}\varepsilon_{23}+2\sigma_{31}\varepsilon_{31}+2\sigma_{12}\varepsilon_{12},\\
   &=[\boldsymbol\sigma]^\mathrm{T}\cdot[\boldsymbol\varepsilon].
   \end{align*}

Fourth rank tensors with minor symmetries
-----------------------------------------

Let :math:`\mathbf C` be a fourth rank tensor with minor symmetries

.. math::

   C_{ijkl}=C_{jikl}=C_{ijlk}.

Its Mandel representation :math:`[\mathbf C]` (as a square matrix) is defined as follows

.. math::

   [\mathbf C] =
   \begin{bmatrix}
       C_{1111} & C_{1122} & C_{1133} & \sqrt{2}C_{1123} & \sqrt{2}C_{1131} & \sqrt{2}C_{1112}\\
       C_{2211} & C_{2222} & C_{2233} & \sqrt{2}C_{2223} & \sqrt{2}C_{2231} & \sqrt{2}C_{2212}\\
       C_{3311} & C_{3322} & C_{3333} & \sqrt{2}C_{3323} & \sqrt{2}C_{3331} & \sqrt{2}C_{3312}\\
       \sqrt{2}C_{2311} & \sqrt{2}C_{2322} & \sqrt{2}C_{2333} & 2C_{2323} & 2C_{2331} & 2C_{2312}\\
       \sqrt{2}C_{3111} & \sqrt{2}C_{3122} & \sqrt{2}C_{3133} & 2C_{3123} & 2C_{3131} & 2C_{3112}\\
       \sqrt{2}C_{1211} & \sqrt{2}C_{1222} & \sqrt{2}C_{1233} & 2C_{1223} & 2C_{1231} & 2C_{1212}
   \end{bmatrix},

where the numbering of the cross-components :math:`C_{ijkl}` with :math:`i\neq j` or :math:`k\neq l` is consistent with the numbering of cross-components of second rank tensors. Again, the :math:`\sqrt 2` and 2 prefactors ensure that matrix-matrix and matrix-vector products coincide with the double contraction of tensors.

More precisely, the Mandel representation of the second rank tensor :math:`\boldsymbol\sigma=\mathbf C:\boldsymbol\varepsilon` is the column vector

.. math::
   [\boldsymbol\sigma]=[\mathbf C:\boldsymbol\varepsilon]=C_{ijkl}\varepsilon_{kl}=[\mathbf C]\cdot[\boldsymbol\varepsilon].

Likewise, if :math:`\mathbf S` is another fourth rank tensor with minor symmetries, then

.. math::
   [\mathbf C:\mathbf S]=[\mathbf C]\cdot[\mathbf S],

where it is recalled that the (i, j, k, l) component of :math:`\mathbf C:\mathbf S` is :math:`C_{ijmn}S_{mnkl}`. It results from the above formula that the Mandel representation of the inverse of a fourth rank tensor is the inverse of the Mandel representation of this tensor

.. math::
   [\mathbf C^{-1}]=[\mathbf C]^{-1}.

Finally, it is readily verified that the Mandel representation of the transpose is the transpose of the Mandel representation

.. math::
   [\mathbf C^\mathrm{T}]=[\mathbf C]^\mathrm{T}.

Mandel notation in 2D
=====================

The above formulas are readily extended to two dimensions, so that we only recall the matrix representation of second rank, symmetric tensors and fourth rank tensors with minor symmetries. The properties of these matrix representations are unchanged.

Second rank, symmetric tensors
------------------------------

The Mandel representation :math:`[\boldsymbol\varepsilon]` (as a column-vector) of any second rank, symmetric tensor :math:`\boldsymbol\varepsilon` is defined as follows

.. math::

   [\boldsymbol\varepsilon] = [\varepsilon_{11}, \varepsilon_{22}, \sqrt2\varepsilon_{12}]^\mathrm{T}.

Fourth rank tensors with minor symmetries
-----------------------------------------

The Mandel representation :math:`[\mathbf C]` (as a square matrix) of any fourth rank tensor :math:`\mathbf C` with minor symmetries is defined as follows

.. math::

   [\mathbf C] =
   \begin{bmatrix}
       C_{1111} & C_{1122} & \sqrt{2}C_{1112}\\
       C_{2211} & C_{2222} & \sqrt{2}C_{2212}\\
       \sqrt{2}C_{1211} & \sqrt{2}C_{1222} & 2C_{1212}
   \end{bmatrix}.
