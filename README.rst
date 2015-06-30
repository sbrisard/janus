.. coding: utf-8

#####
Janus
#####

Introduction
============

Janus is a Python library dedicated to the discretization of the Lippmann–Schwinger equation with periodic boundary conditions. The matrix of the resulting linear system is the sum of a block-diagonal matrix and a block-circulant matrix. Following the ideas initially introduced by Moulinec and Suquet (1998) matrix-vector products can then be computed efficiently by means of the Fast Fourier Transform. A matrix-free strategy is then adopted to solve the linear system iteratively, e.g. (non-exhaustive list)

- fixed-point iterations (Moulinec & Suquet, 1998),
- accelerated schemes (Eyre & Milton, 1999),
- augmented Lagrangians (Michel, Moulinec & Suquet, 2001),
- Krylov subspace linear sovers (Brisard & Dormieux, 2010),
- polarization-based schemes (Monchiet & Bonnet, 2012),

see also Moulinec & Silva (2014) for a comparison of some of these iterative schemes.

The library provides tools to define the linear operator associated to the discretized Lippmann–Schwinger equation, and to compute the necessary matrix-vector products. Third-party iterative linear solvers ([Scipy](http://docs.scipy.org/doc/scipy-0.15.1/reference/sparse.linalg.html#solving-linear-problems), [petsc4py](https://bitbucket.org/petsc/petsc4py)) can then be invoked to compute the solution.

The library is designed with performance in mind. It is fully parallelized (using MPI and mpi4py), and the critical parts of the code are written in Cython.

History of major changes
========================

2015-02-23 — Reconciliation of the APIs of FFT objects and operators
--------------------------------------------------------------------

The following attributes of FFT objects were renamed (incompatible changes)

- ``rshape`` → ``ishape``: the shape of the *local* input array,
- ``cshape`` → ``oshape``: the shape of the *local* output array,
- ``shape`` → ``global_ishape``: the shape of the *global* input array.

Besides, the following attribute was added

- ``global_oshape``: the shape of the *global* output array.