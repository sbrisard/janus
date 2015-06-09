.. -*- coding: utf-8 -*-

************
Introduction
************

Janus is a Python library dedicated to the discretization of the Lippmann--Schwinger equation with periodic boundary conditions. The matrix of the resulting linear system is the sum of a block-diagonal matrix and a block-circulant matrix. Following the ideas initially introduced by Moulinec and Suquet [MS98]_ matrix-vector products can then be computed efficiently by means of the `Fast Fourier Transform <http://en.wikipedia.org/wiki/Fast_Fourier_transform>`_. A matrix-free strategy is then adopted to solve the linear system iteratively, e.g. (non-exhaustive list)

  - fixed-point iterations [MS98]_,
  - accelerated schemes [EM99]_,
  - augmented Lagrangians [MMS01]_,
  - Krylov subspace linear sovers [BD10]_,
  - polarization-based schemes [MB12]_,

see also [MS14]_ for a comparison of some of these iterative schemes.

The library provides tools to define the linear operator associated to the discretized Lippmann--Schwinger equation, and to compute the necessary matrix-vector products. Third-party iterative linear solvers (`Scipy <http://docs.scipy.org/doc/scipy-0.15.1/reference/sparse.linalg.html#solving-linear-problems>`_, `petsc4py <https://bitbucket.org/petsc/petsc4py>`_) can then be invoked to compute the solution.

The library is designed with performance in mind. It is fully parallelized (using MPI and `mpi4py <https://bitbucket.org/mpi4py/mpi4py>`_), and the critical parts of the code are written in `Cython <http://cython.org/>`_.

.. rubric :: References

.. [BD10] S. Brisard and L. Dormieux. FFT-based methods for the mechanics of composites: A general variational framework. *Computational Materials Science*, 49(3):663--671, 2010.

.. [EM99] D. J. Eyre and G. W. Milton. A fast numerical scheme for computing the response of composites using grid refinement. *European Physical Journal-Applied Physics*, 6(1):41--47, 1999.

.. [MB12] V. Monchiet and G. Bonnet. A polarization-based FFT iterative scheme for computing the effective properties of elastic composites with arbitrary contrast. *International Journal for Numerical Methods in Engineering*, 89(11):1419--1436, 2012.

.. [MMS01] J.C. Michel, H. Moulinec, and P. Suquet. A computational scheme for linear and non-linear composites with arbitrary phase contrast. *International Journal for Numerical Methods in Engineering*, 52(1--2):139--160, 2001.

.. [MS98] H. Moulinec and P. Suquet. A numerical method for computing the overall response of nonlinear composites with complex microstructure. *Computer Methods in Applied Mechanics and Engineering*, 157(1-2):69--94, 1998.

.. [MS14] H. Moulinec and F. Silva. Comparison of three accelerated fft-based schemes for computing the mechanical response of composite materials. *International Journal for Numerical Methods in Engineering*, 97(13):960--985, 2014.
