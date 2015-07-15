.. -*- coding: utf-8 -*-

#####
Janus
#####

Janus is a Python library dedicated to the discretization of the Lippmann--Schwinger equation with periodic boundary conditions. The library is designed with performance in mind. It is fully parallelized, and the critical parts of the code are written in Cython.

Janus is released under a BSD 3-clause license (see ``LICENSE.txt``).

The documentations can be found on `RTD <http://janus.readthedocs.org/>`_.

History of major changes
========================

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
