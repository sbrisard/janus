.. -*- coding: utf-8 -*-

************
Introduction
************

Janus is a Python library dedicated to the discretization of the Lippmann--Schwinger equation with periodic boundary conditions. The matrix of the resulting linear system is the sum of a block-diagonal matrix and a block-circulant matrix. Following the ideas initially introduced by :ref:`Moulinec & Suquet (1998) <MOUL1998>` matrix-vector products can then be computed efficiently by means of the `Fast Fourier Transform <http://en.wikipedia.org/wiki/Fast_Fourier_transform>`_. A matrix-free strategy is then adopted to solve the linear system iteratively, e.g. (non-exhaustive list)

  - fixed-point iterations (:ref:`Moulinec & Suquet, 1998 <MOUL1998>`),
  - accelerated schemes (:ref:`Eyre & Milton, 1999 <EYRE1999>`),
  - augmented Lagrangians (:ref:`Michel et al., 2001 <MICH2001>`),
  - Krylov subspace linear sovers (:ref:`Brisard & Dormieux, 2010 <BRIS2010A>`),
  - polarization-based schemes (:ref:`Monchiet & Bonnet, 2012 <MONC2012>`),

see also :ref:`Moulinec & Silva (2014) <MOUL2014>` for a comparison of some of these iterative schemes.

The library provides tools to define the linear operator associated to the discretized Lippmann--Schwinger equation, and to compute the necessary matrix-vector products. Third-party iterative linear solvers (`Scipy <http://docs.scipy.org/doc/scipy-0.15.1/reference/sparse.linalg.html#solving-linear-problems>`_, `petsc4py <https://bitbucket.org/petsc/petsc4py>`_) can then be invoked to compute the solution.

The library is designed with performance in mind. It is fully parallelized (using MPI and `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_), and the critical parts of the code are written in `Cython <http://cython.org/>`_.
