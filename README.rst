.. -*- coding: utf-8 -*-

#####
Janus
#####

|tests| |docs|

.. |tests| image:: https://github.com/sbrisard/janus/actions/workflows/tests.yml/badge.svg?branch=master
   :target: https://github.com/sbrisard/janus/actions/workflows/tests.yml
   :alt: Tests

.. |docs| image:: https://img.shields.io/badge/docs-sbrisard.github.io%2Fjanus-blue
   :target: https://sbrisard.github.io/janus/
   :alt: Documentation

Janus is a Python library dedicated to the discretization of the Lippmann--Schwinger equation with periodic boundary conditions. The library is designed to be as flexible as possible. In order to ensure reasonable performance, the critical parts of the code are written in Cython.

Janus is released under a BSD 3-clause license (see ``LICENSE.txt``).

The documentation can be found at https://sbrisard.github.io/janus/.

History of major changes
========================

2026-09-23 — This code no longer supports ``MPI``
-------------------------------------------------

Janus no longer targets large simulations. As explained in the `roadmap <https://sbrisard.github.io/janus/roadmap.html>`_, support for parallel computing through MPI has therefore been removed, which makes installation simpler.

The last version that supports MPI is tagged ``Farewell_MPI`` in the Git repository.

As a consequence, the FFT objects and the discrete Green operators no longer distinguish local and global shapes: the attributes ``offset0``, ``global_shape0``, ``global_ishape`` and ``global_oshape`` were removed (incompatible changes). Use ``shape0``, ``ishape`` and ``oshape`` instead.


2015-07-09 — This code is now licensed under BSD 3-clause license
-----------------------------------------------------------------

See LICENSE.txt.

2015-02-23 — Reconciliation of the APIs of FFT objects and operators
--------------------------------------------------------------------

The following attributes of FFT objects were renamed (incompatible changes)

- ``rshape`` → ``ishape``: the shape of the *local* input array,
- ``cshape`` → ``oshape``: the shape of the *local* output array,
- ``shape`` → ``global_ishape``: the shape of the *global* input array.

Besides, the following attribute was added

- ``global_oshape``: the shape of the *global* output array.
