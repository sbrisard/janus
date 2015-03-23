.. coding: utf-8

#####
Janus
#####

Janus is a Python library which facilitates the implementation of large linear systems where the matrix is the sum of a block-diagonal matrix and a block-circulant matrix. Such systems typically arise from the discretization of variational problems where the bilinear form is the sum of a *local* bilinear form and a *convolution* bilinear form. The Lippmann--Schwinger equation in linear elasticity is an example of such variational problem.

The matrices involved in such problems are too large to be stored and a matrix-free strategy must be adopted. Janus provides a general framework to implement block-diagonal and block-circulant operators (and many more). It interfaces with FFTW to compute matrix-vector products involving block-circulant operators (which are diagonal in Fourier space).

History of changes
==================

2015-023-23 -- Reconciliation of the APIs of FFT objects and operators
----------------------------------------------------------------------

The following attributes of FFT objects were renamed (incompatible changes)

  - `rshape` → `ishape`: the shape of the *local* input array,
  - `cshape` → `oshape`: the shape of the *local* output array,
  - `shape` → `global_ishape`: the shape of the *global* input array.

Besides, the following attribute was added

  - `global_oshape`: the shape of the *global* output array.
