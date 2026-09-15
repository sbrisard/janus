.. -*- coding: utf-8 -*-

#####
Janus
#####

Janus is a Python library dedicated to the discretization of the Lippmann--Schwinger equation with periodic boundary conditions. The library is designed to be as flexible as possible. In order to ensure reasonable performance, the critical parts of the code are written in Cython.

Janus is released under a BSD 3-clause license (see ``LICENSE.txt``).

The documentations can be found at http://sbrisard.github.io/janus/.

History of major changes
========================

2026-09-15 — This code no longer supports ``MPI``
-------------------------------------------------

Janus no longer targets large simulations. As explained in the `roadmap <https://sbrisard.github.io/janus/roadmap.html>`_, support for parallel computing through MPI is therefore being removed, which will make installation simpler.

The last version that supports MPI is tagged ``Farewell_MPI`` in the Git repository.

The removal is being carried out in the ``MPI-ectomy`` branch.


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
