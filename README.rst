#####
Janus
#####

Janus is a Python library which facilitates the implementation of large linear systems where the matrix is the sum of a block-diagonal matrix and a block-circulant matrix. Such systems typically arise from the discretization of variational problems where the bilinear form is the sum of a *local* bilinear form and a *convolution* bilinear form. The Lippmann--Schwinger equation in linear elasticity is an example of such variational problem.

The matrices involved in such problems are too large to be stored and a matrix-free stratgey must be adopted. Janus provides a general framework to implement block-diagonal and block-circulant operators (and many more). It interfaces with FFTW to compute matrix-vector products involving block-circulant operators (which are diagonal in Fourier space).
