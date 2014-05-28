************
Introduction
************

Janus is a Python library which allows the discretization of large variational problems where the bilinear form is the sum of a *local* bilinear form and a *convolution* bilinear form. After discretization, the variational problem reduces to a linear system, where the matrix is the sum of a block-diagonal matrix and a block-circulant matrix. Since these matrices are too large to be stored, they are implemented in Janus as matrix-free linear-operators (see :doc:`Operators<./operators>`). Block-circulant operators are diagonal in Fourier space, and therefore computed through FFT.

.. TODO Provide more details and references once a general interface for block-circulant operators is defined.

To be more precise, we consider here *periodic* functions :math:`u`, :math:`v` defined over the unit cell :math:`\Omega=(0,L_1)\times\cdots\times(0,L_d)\subset\mathbb R^d`. :math:`\mathbb V` denotes a space of such functions (with specified regularity). The variational problem to be solved reads

.. math:: \text{Find }u\in\mathbb V\text{ such that }\mathcal a(u, v) = \mathcal L(v)\text{ for all }v\in\mathbb V,

where :math:`\mathcal L` is a linear form, and :math:`\mathcal A` is a bilinear form with the following decomposition

.. math:: \mathcal A(u, v) = \mathcal D(u, v) + \mathcal C(u, v),

with

.. math:: \mathcal D(u, v) = \int_{x\in\mathcal U}u(x)^TD(x)v(x)dx

(:math:`D(x)` is a matrix) and

.. math:: \mathcal C(u, v) = \int_{x,y\in\Omega}u(x)^TC(x-y)v(y)dxdy

A prototypical example of such problems is the Lippmann--Schwinger equation in linear elasticity.
