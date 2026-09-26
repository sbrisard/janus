.. -*- coding: utf-8 -*-

.. _roadmap:

***************
Roadmap to v1.0
***************

This chapter was written by Claude at the request of the author (see :doc:`claude`, task *The road to v1.0*). It analyses the evolutions that would turn Janus into a flexible, easily maintainable and extensible library, and proposes an ordered sequence of releases leading to version 1.0. It was revised on 2026-09-25 and 2026-09-26, after the decision to keep Cython (see :doc:`claude`, sections *The continuum Green operator in the new architecture* and *Automatic differentiation with JAX*); the revised sections carry a note. It is a working document: milestone 0.2 (E1, and the continuous integration part of E11) is done; the other evolutions have not been implemented yet.

Goals
=====

.. note:: These goals were revised on 2026-09-26, for consistency with the decision to keep Cython (see E3, and :doc:`claude`, section *The continuum Green operator in the new architecture*). They initially required new discretizations and new physics to be implementable in pure Python, without compilation, and asked that the architecture should not rule out GPU execution.

Janus was initially designed for two purposes: rapid prototyping, and large (distributed) simulations. The second purpose is now obsolete, because new developments in the community will soon offer excellent alternatives. Janus should therefore focus on prototyping:

- installation should be as easy as possible (ideally ``pip install``, without compiler, thanks to precompiled wheels);
- new discretizations and new physics (conductivity, Darcy flow, finite strain hyperelasticity…) are written in Cython, within the library, mainly by the author; the local operators (constitutive laws, description of the microstructure) are written without compilation, in NumPy, or in Numba where NumPy is not suitable;
- the constitutive laws of interest are linear laws, and hyperelastic laws at finite strain; laws with internal variables (plasticity, damage, viscosity…) are out of scope;
- GPU execution is not a priority; automatic differentiation is not required, but remains possible through an adapter of the discrete Green operator (E13);
- any dependency that is distributed as binary wheels is acceptable;
- breaking changes are allowed.

The speed estimates given below are *not* backed by benchmarks (at the author's request, the analysis relies on the code and on the known properties of the tools); benchmarks are part of the proposed milestones.

Where does the friction come from?
==================================

The following observations are based on a complete reading of the current code base (about 2300 lines of Cython, in ``janus/operators.pyx``, ``janus/green.pyx``, ``janus/material/elastic/linear/isotropic.pyx`` and ``janus/fft/``).

1. **Extensions must be written in Cython.** All the methods that do the actual work (``c_apply``, ``c_set_frequency``, ``c_apply_by_freq``, ``c_to_memoryview``) are ``cdef`` methods, which cannot be overridden in Python. The docstrings of ``init_sizes`` and ``init_shapes`` suggest that pure Python subclasses are supported, but this only holds for direct calls to ``apply``. Inside the library, the Cython methods are called, and Python overrides are *silently ignored*. For example, a Python subclass of :class:`AbstractOperator <janus.operators.AbstractOperator>` which doubles its input returns the expected result when called directly, but when used as a local operator of a :class:`BlockDiagonalOperator2D <janus.operators.BlockDiagonalOperator2D>`, the output array is left untouched (the no-op ``AbstractOperator.c_apply`` is called instead); this was checked with the current build. Likewise, a continuum Green operator (for new physics) must be a Cython subclass of ``AbstractGreenOperator``.

2. **Operators are stateful.** A continuum Green operator is evaluated in two steps, which share a hidden state: ``set_frequency(k)``, then ``apply(tau, eta)``; the discrete Green operators loop over all frequencies in Cython. Evaluating the physics one wave-vector at a time is not a problem in itself: it follows the mathematics, it is efficient in Cython, and it allocates nothing (this design is kept, see E3 and E4). The hidden state is the problem: an operator cannot be shared between threads, so that the loop over frequencies cannot be parallelized, and the result of ``apply`` depends on a previous call. The local operators have a similar weight: ``BlockDiagonalOperator2D/3D`` holds one Cython object per cell, called indirectly at each cell, and describes the microstructure cell by cell rather than by its phases (see E8). One consequence of the per-frequency design remains: a physics written in Python would be called once per frequency (e.g. 8 million calls for a 256³ grid), so that it must be compiled to be fast (see E3).

   .. note:: This item was revised on 2026-09-26, for consistency with the decision to keep Cython (see :doc:`claude`, section *The continuum Green operator in the new architecture*). It initially presented the per-frequency design itself as the problem, and concluded that any Python-level extension mechanism requires operators that act on whole arrays of wave-vectors or cells at once.

3. **Everything is duplicated for 2D and 3D.** Almost every class exists in two versions (``AbstractStructuredOperator2D/3D``, ``BlockDiagonalOperator2D/3D``, ``TruncatedGreenOperator2D/3D``, ``FilteredGreenOperator2D/3D``, ``FiniteDifferences2D/3D``, ``_RealFFT2D/3D``, ``_GreenOperatorForStrains2D/3D``, ``FourthRankIsotropicTensor2D/3D``…), and the 3D versions of some operators are missing (``FourthRankCubicTensor3D``).

4. **Discretizations are tied to the physics.** The filtered Green operators hard-code symmetric 3×3 (2D) and 6×6 (3D) matrices, i.e. strain-based elasticity; their code is generated by ``scripts/gencode.py``. The filtered discretization cannot be reused for, say, conductivity (2×2 or 3×3 matrices).

5. **The data model is restricted.** Local data are vectors of fixed size, symmetric tensors are stored in Mandel–Voigt form, only ``float64`` is supported, and complex Fourier coefficients are stored as interleaved real numbers, the Green operator being applied separately to the real and imaginary parts (which assumes a *real* Fourier symbol). Finite strain mechanics requires non-symmetric second-rank tensors (9 components in 3D), and some discretizations have complex symbols.

6. **Distributed memory leaks into the serial code.** Discrete Green operators and FFT objects distinguish local and global shapes (``offset0``, ``global_shape0``, ``global_ishape``…), although only the serial version is used in practice.

7. **The build chain is fragile.** Compilation requires a C compiler, a hand-written ``setup.cfg`` with machine-specific paths to FFTW, and a ``setup.py`` that relies on ``distutils`` (see :doc:`installation` and the task *Installation on a windows machine* in :doc:`claude`).

Minor issues found along the way illustrate the maintenance burden of the current code: ``TruncatedGreenOperator3D.c_apply`` calls the Python-level ``set_frequency`` (with argument checks) in its innermost loop, whereas the 2D version calls ``c_set_frequency``; some variables in ``FiniteDifferences3D.c_set_frequency`` are untyped (hence Python objects); the error messages of ``_RealFFT3D`` report wrong shapes (``TODO`` in the code).

Analysis of the proposed evolutions
===================================

Each evolution is analysed in terms of gains, losses, consequences on the code and implementation problems, and ends with a recommendation. Evolutions E1 to E3 were proposed by the author; E4 to E12 are additional suggestions; E13 was added after a later discussion.

E1. Removal of the MPI dependency
---------------------------------

**Gains.**

- Installation: no ``mpi4py``, no MPI-enabled FFTW, no parsing of ``mpicc -show`` in ``setup.py``. This also removes a fragile step on Windows (``setup.py`` fails if ``mpi4py`` is installed but ``mpicc`` is missing).
- Code: the distinction between local and global shapes disappears from the FFT objects and the discrete Green operators (``offset0``, ``global_shape0``, ``global_ishape``, ``global_oshape``, ``n0_loc``). ``janus/fft/parallel/`` (about 110 lines), ``tests/parallel/``, ``sphinx/parallel_fft_tutorial.py`` and the ``[fftw_mpi]`` configuration can be deleted.
- Documentation: the API of ``janus.fft.parallel`` is currently missing from the documentation built on Windows; this problem disappears.

**Losses.** Distributed-memory simulations are no longer possible, which is consistent with the new goals. Shared-memory parallelism remains available (multithreaded FFTs, ``prange`` loops in Cython or Numba): 3D grids of 256³ to 512³ cells remain within reach of a workstation (a single 512³ field of symmetric tensors, 6 components in double precision, requires 6.4 GB).

**Consequences on the code.** Breaking change for users of ``janus.fft.parallel`` (presumably few). The constructors of the discrete Green operators are simplified.

**Implementation problems.** None: this is mostly code deletion. If distributed computing were needed again, it could be reintroduced at the level of the FFT backend (e.g. ``mpi4py-fft``) without affecting the rest of the architecture, provided that the FFT is kept behind a small interface (see E2).

**Recommendation.** Do it first (milestone 0.2): it is cheap, and it reduces the amount of code to be ported by the other evolutions.

E2. Interfacing with FFTW
-------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decision to keep Cython (see E3, and :doc:`claude`, section *The continuum Green operator in the new architecture*, *Follow-up: milestones*).

The hand-written wrapper (``janus/fft/serial/``, about 370 lines of Cython) could be replaced by `pyFFTW <https://github.com/pyFFTW/pyFFTW>`_. Another candidate should be considered: ``scipy.fft``.

**pyFFTW.**

- Gains: removes the Cython wrapper and, above all, the need to locate and link FFTW at build time (``setup.cfg``), which is the most fragile step of the current installation. pyFFTW is distributed as binary wheels (version 0.15.1 provides wheels for Python 3.14, including Windows), and exposes the planner flags, wisdom, multithreading and aligned arrays of FFTW.
- Losses: an additional compiled dependency, with a slow release cadence (the latest release, 0.15.1, dates from October 2025). FFTW is licensed under the GPL, which constrains the distribution of software that links it; note that Janus (BSD-3-Clause) is *already* in this situation.

**scipy.fft.**

- Gains: no *compiled* dependency to be located at build time; SciPy becomes a runtime dependency of Janus (currently, NumPy is its only one), but it is distributed as binary wheels, and users need it anyway for the iterative solvers; BSD license, real-to-complex transforms along arbitrary axes of an array (``rfftn(x, axes=...)``, hence directly applicable to fields of tensors), multithreading (``workers`` argument), no planning step. Moreover, pyFFTW can be plugged in as a *backend* of ``scipy.fft`` (``scipy.fft.set_backend(pyfftw.interfaces.scipy_fft)``), without any change to the code that calls ``scipy.fft``.
- Losses: typically somewhat slower than FFTW with ``FFTW_MEASURE``; no wisdom. This loss of the fine tuning of FFTW is measured by the benchmark of milestone 0.3.

**Keeping the current wrapper.** Since Cython is kept (E3), the current wrapper could be kept as well, and the FFT called from Cython. This is not worth it: the FFT is called once per field (or once per component of a field), so that calling it from Python has a negligible cost, the loop over frequencies remaining in Cython (E4); on the other hand, the link to FFTW is what requires ``setup.cfg`` on some machines, and it would complicate the production of precompiled wheels (E11), which would have to embed FFTW.

**Consequences on the code.** The FFT objects (``_RealFFT2D/3D``, with their explicit copies to and from an internal buffer) and the planner flags (``janus.fft.FFTW_*``) disappear. The current wrapper copies the data to and from an FFTW buffer at each call anyway, so no zero-copy optimization is lost. Complex Fourier coefficients become genuine complex arrays, instead of interleaved real arrays.

**Implementation problems.**

- The layout of the output of real-to-complex transforms (last axis of length ``n//2 + 1``) and the handling of the Nyquist frequency for even grid sizes must be reproduced exactly, so that the new implementation matches the reference data in ``tests/data``.
- ``scipy.fft`` has no ``out`` argument: each transform allocates its result, whereas the current discrete Green operators preallocate two Fourier buffers (two complex fields), and the in-place semantics of ``apply(x, y)`` is kept (E7). To limit the temporaries, the whole field is transformed at once (``rfftn`` over the grid axes), the Green operator is applied *in place* on the Fourier coefficients (with a small temporary vector per frequency, which removes the second buffer), and the inverse transform is written component by component into ``y``. The temporaries then amount to one complex field and one real component; ``apply(x, x)`` remains valid, since ``x`` is entirely transformed before ``y`` is written. The layout of the output of ``rfftn`` seen by the Cython loop (contiguity, strides) is to be checked.

**Recommendation.** Use ``scipy.fft``, called from Python, the loop over frequencies remaining in Cython; document pyFFTW as an optional accelerator, via the ``scipy.fft`` backend mechanism, and do not make it a hard dependency. The new core uses ``scipy.fft`` from milestone 0.3 on; the current wrapper, the link to FFTW and ``setup.cfg`` are deleted at the switch (milestone 0.4).

E3. Replacing Cython with other (dynamic) tools
-----------------------------------------------

.. note:: This section was revised on 2026-09-25, after a discussion with the author (see :doc:`claude`, section *The continuum Green operator in the new architecture*). Its initial recommendation, to replace Cython entirely, was reversed.

Two kinds of loops are compiled in the current code: the frequency-wise application of the Green operator, and the cell-wise application of local operators. The alternatives to Cython fall into two families: vectorized array expressions (NumPy, JAX), where no loop over frequencies is written at all, and just-in-time compilation of explicit loops (Numba).

**Pure NumPy (vectorized array expressions).**

- Gains: no compilation at all, runs everywhere, easiest to read, write and debug.
- Losses: a Python loop over frequencies is prohibitive (at least a microsecond per iteration, hence minutes per application of the operator on a 512³ grid, against less than a second in Cython), so that everything must be expressed as operations on whole arrays. This has a cost in memory, which the discussion of 2026-09-25 made explicit: either the symbol is formed for all frequencies (in 3D, a 6×6 matrix per frequency of a 256³ grid requires 2.4 GB), or it is applied in closed form, but it must then receive the wave-vectors of the whole grid. Storing them costs a quarter of a field in 3D elasticity (1.6 GB for a 512³ grid, against 6.4 GB for the Fourier coefficients of the polarization), and half of a field in 2D conductivity; the current code stores nothing. Array expressions also create temporaries of the size of the grid. Workarounds exist (wave-vectors passed as broadcastable components, evaluation by blocks of frequencies), but they are symptoms of the vectorized design, and move the code away from its mathematical form.

**Numba.**

- Gains: loops written in Python syntax and compiled at run time (``@njit``, ``parallel=True``, ``cache=True``), with a performance similar to Cython, and no compilation at installation (Numba 0.67 provides wheels for Python 3.14, including Windows). The physics can be written for one wave-vector at a time, as in Cython, and passed to a generic kernel.
- Losses: only a subset of Python and NumPy is supported; passing user-defined physics to generic kernels is possible, but awkward (jitted functions as arguments, material parameters to be passed explicitly or frozen in closures); compilation latency at the first call; harder debugging; no automatic differentiation; GPU support is limited to NVIDIA hardware, through a separate package.

**JAX.**

- Gains: NumPy-like API; ``jit`` compiles and *fuses* array expressions (which removes the temporaries of the pure NumPy approach); ``vmap`` turns a function of one wave-vector or one cell into a function of the whole grid; automatic differentiation; GPU execution. jaxlib 0.11 provides wheels for Python 3.14, including Windows (CPU only).
- Losses: arrays are immutable, so that the current in-place API (``apply(x, y)``, ``apply(x, x)``) is not possible; single precision by default; tracing constraints; compilation latency; a large dependency.

**Keeping Cython.** The alternatives above are worth their cost only if new physics must be written *without compiling*, or if GPU execution and automatic differentiation are required. The author has stated that new physics will be written mainly by the author, within the library, and that GPU execution is not a priority. Under these conditions, compiling is not an obstacle, and most friction points listed above are *design* problems, which can be fixed without changing the tool:

- stateful operators (friction point 2): see E4;
- Python subclasses silently ignored (friction point 1): ``cpdef`` methods, or an explicit error, so that a Python implementation is either used (slowly, but correctly) or rejected;
- restricted data model (friction point 5): Cython's fused types cover ``float32``, ``float64`` and complex types, and nothing in Cython imposes the Mandel–Voigt form or real symbols;
- duplication between 2D and 3D (friction point 3): can be reduced by looping over a flattened frequency index, the dimension being a run-time value, at the cost of the hand-unrolled expressions; this is where Cython helps least (see E5);
- fragile build chain (friction point 7): already largely addressed by the optional ``setup.cfg`` and by continuous integration on Linux and Windows (milestone 0.2); precompiled wheels (e.g. built with ``cibuildwheel``) would remove the need for a compiler on the user's side.

What remains specific to Cython is that a new physics must be compiled to be fast (a physics written in pure Python would be called once per frequency, hence as slowly as a Python loop), and that GPU execution is out of reach. Automatic differentiation, on the other hand, remains possible through a small adapter, the discrete Green operator being the only part of the computation that JAX cannot see (see E13).

**Recommendation.** Keep Cython, and fix its friction points by design (E4, and possibly E5). The existing code remains the reference, and the reference data in ``tests/data`` provide a regression oracle for the rewrite. A Numba prototype (generic kernel, truncated scheme, 2D conductivity, timed against the Cython code) may be written later, if the choice of Cython is to be confirmed by measurements.

E4. Stateless operators
-----------------------

.. note:: This section was revised on 2026-09-25, together with E3 (see :doc:`claude`, section *The continuum Green operator in the new architecture*). It initially proposed vectorized operators, acting on arrays of wave-vectors.

**Proposal.** A continuum Green operator exposes a stateless ``cdef`` method that applies its symbol at *one* wave-vector: given :math:`\mathbf k` and :math:`\hat{\boldsymbol\tau}`, it computes :math:`\hat{\boldsymbol\eta} = \hat\Gamma(\mathbf k) \cdot \hat{\boldsymbol\tau}` in closed form, without forming the matrix of :math:`\hat\Gamma(\mathbf k)`, and without any ``set_frequency`` state. The discrete Green operators loop over the frequencies, and compute each wave-vector (or each modified wave-vector, see E6) on the fly: no array of wave-vectors is stored, and no temporary array is created.

The zero frequency is part of the definition: :math:`\hat\Gamma(\mathbf 0) = \mathbf 0`, i.e. the Green operator acts on fields with zero mean. This value does not depend on the loading: the macroscopic loading (imposed mean strain, mean stress, or mixed conditions) is handled by adapting the equation to be solved, not the Green operator.

For example, the Green operator of isotropic conductivity (conductivity ``k0``), :math:`\hat\Gamma(\mathbf k) \cdot \hat{\boldsymbol\tau} = \mathbf k \, (\mathbf k \cdot \hat{\boldsymbol\tau}) / (k_0 \, |\mathbf k|^2)`, could be written as follows (a sketch: the exact signature is still open, see below)::

    cdef void apply(self, const double *k, const double complex *tau,
                    double complex *eta) noexcept nogil:
        cdef int i
        cdef double k2 = 0.
        cdef double complex k_tau = 0.
        for i in range(self.dim):
            k2 += k[i] * k[i]
            k_tau += k[i] * tau[i]
        for i in range(self.dim):
            eta[i] = 0. if k2 == 0. else k[i] * k_tau / (self.k0 * k2)

**Gains.** The code follows the mathematics (the symbol is applied at a wave-vector); memory is minimal, as in the current code; the operators are thread-safe, so that the loop over frequencies can be parallelized (``prange``, without the GIL); the same method can be evaluated at modified wave-vectors (E6). The matrix of :math:`\hat\Gamma(\mathbf k)`, which is useful for testing, is obtained by applying the operator to the vectors of the canonical basis: it need not be implemented by each physics.

**Losses.** A new physics must be written in Cython to be efficient (see E3). A notation such as ``symbol(k) @ tau``, closer to the mathematics, is not provided: it would require an intermediate object (a matrix, or a matrix-free linear operator) at each call.

**Consequences.** Rewrite of ``janus/green.pyx`` and ``janus/material/``. The API changes from ``green.set_frequency(k); green.apply(tau, eta)`` to a single call that takes the wave-vector as an argument.

**Implementation problems.**

- Signature of the ``cdef`` method: raw pointers or memoryviews (slicing a memoryview at each frequency has a non-negligible cost in an inner loop); real or complex wave-vectors (some schemes involve complex modified wave-vectors, see E6); representation of the local tensors (see E9), which fixes the sizes of ``tau`` and ``eta``.
- Python-level access: a ``def`` (or ``cpdef``) counterpart is needed for the tests and for interactive use, and Python subclasses must not be silently ignored (friction point 1).
- Nyquist frequencies, and modified wave-vectors that vanish at a nonzero discrete frequency: with the above definition, the Green operator is zero there. Check that this is the intended behaviour for each scheme. On grids of even sizes, the truncated scheme of the current code is not a projector, presumably because of the Nyquist frequencies (see E12).

**Recommendation.** Core of the new architecture (milestone 0.3).

E5. Dimension-generic implementation
------------------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decision to keep Cython (see E3, and :doc:`claude`, section *The continuum Green operator in the new architecture*). It initially assumed a core written in NumPy or JAX.

**Proposal.** A single implementation for 2D and 3D (and possibly 1D, useful for tests), the dimension being a run-time value. This concerns the Cython part of the library, i.e. the discrete Green operators and the physics: local operators written in NumPy (E8) act on arrays of shape ``(*grid_shape, n)``, and are dimension-generic at no cost.

**Gains.** Roughly halves the Cython code, and removes the risk of inconsistencies between the 2D and 3D versions (see the minor issues listed above); missing 3D variants (``FourthRankCubicTensor3D``) come for free when the underlying formula is dimension-independent.

**Losses.** Hand-unrolled 3×3/6×6 expressions are replaced by loops whose bounds (dimension, number of components) are only known at run time, which the C compiler cannot unroll; the resulting slowdown is to be measured.

**Consequences.** Classes suffixed by ``2D``/``3D`` disappear from the API. The ``apply`` method of the physics takes the dimension as a parameter from milestone 0.3 on, so that E5 does not change its interface (see E4).

**Implementation problems.** The number of dimensions of a Cython memoryview is fixed at compile time: the discrete Green operators therefore loop over a *flattened* frequency index, on a view of shape ``(N, n)`` of the Fourier coefficients (``N`` frequencies, ``n`` components), which requires the layout of the output of ``rfftn`` to be known (see E2). The discrete frequency, hence the wave-vector, must then be recovered from the flattened index, either by integer divisions at each frequency, or by incrementing a multi-index; the cost of either option is to be measured. This is where Cython helps least (see E3).

**Recommendation.** Milestone 0.5, after the switch: 0.3 is already the largest milestone, and E5 does not change the interface of the physics.

E6. Separation of discretization and physics
--------------------------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decision to keep Cython (see E3 and E4). It initially stated that new schemes could be added in a few lines of Python.

**Proposal.** A discretization scheme is described independently of the physics, as a list of *weighted modified wave-vectors* associated with each discrete frequency; the discrete Green operator is then the weighted sum of the continuum symbol evaluated at these wave-vectors:

- truncated scheme: one term (the wave-vector of the discrete frequency itself, with weight 1);
- filtered scheme: 4 terms in 2D and 8 terms in 3D (neighbouring wave-vectors, with ``cos²`` weights), as currently implemented in ``FilteredGreenOperator2D/3D``;
- finite differences (``willot2015``): one term, with a modified wave-vector.

Like the physics (E4), a scheme is written in Cython, and is stateless: for a given discrete frequency, it computes its modified wave-vectors and their weights on the fly, in buffers provided by the caller. The loop over frequencies, the calls to the physics and the weighted sum are generic.

**Gains.** Any scheme works with any physics (e.g. the filtered scheme for conductivity), and a new scheme only requires the computation of its wave-vectors and weights, in a few lines of Cython; ``scripts/gencode.py`` becomes useless.

**Losses.** Since the matrix of the symbol is no longer formed (E4), the filtered scheme applies the continuum symbol 4 or 8 times per frequency, and sums the results. The current code also evaluates the symbol 4 or 8 times per frequency, forming a matrix each time, and then combining the matrices; no significant loss is expected. The calls from the generic loop to the scheme and to the physics are indirect (``cdef`` methods), and cannot be inlined by the C compiler. Both points are to be checked by the benchmark of milestone 0.3.

**Consequences.** New public concept (``Scheme``), and new signature of the discrete Green operators (continuum operator + scheme + grid).

**Implementation problems.** Some schemes involve complex modified wave-vectors (the current implementation of ``willot2015`` uses an equivalent real form); symbols may therefore have to accept complex wave-vectors and use conjugates where required (e.g. ``k ⊗ conj(k)``), which is one of the open points of the signature of the physics (see E4). Schemes that are not of the "weighted sum" type (e.g. staggered grids) may require a more general interface; this should be kept in mind, but not designed for upfront.

**Recommendation.** Adopt in milestone 0.3, and validate it by reproducing the three existing schemes against the reference data.

E7. Array API and functional interface
--------------------------------------

.. note:: This evolution was *not adopted* on 2026-09-25, as a consequence of the revision of E3 (see :doc:`claude`, section *The continuum Green operator in the new architecture*). The section is kept so that the numbering of the evolutions does not change.

**Proposal.** Write the library against the `Python array API standard <https://data-apis.org/array-api/>`_ (via ``array-api-compat``), in functional style: ``y = op(x)`` rather than ``op.apply(x, y)``, so that the same code runs on NumPy, JAX, and possibly CuPy or PyTorch.

**Why it was not adopted.** This evolution only made sense for a core written in array expressions, meant to run on JAX later. Since Cython is kept (E3), and since GPU execution is not a priority, it has lost its purpose; automatic differentiation remains possible without it, through an adapter of the discrete Green operator (E13). Its main cost, on the other hand, remains: in-place operations and output arguments would disappear from the API.

**Consequences.** None: the current in-place semantics of the operators (``apply(x, y)``, ``apply(x, x)``, see :ref:`in-place-operations`) is kept.

E8. Local operators as fields
-----------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decisions on local operators (see :doc:`claude`, section *The continuum Green operator in the new architecture*, *Follow-up: local operators* and *Follow-up: scope of the constitutive laws*, and section *Automatic differentiation with JAX*). It initially proposed constitutive functions of one cell, vectorized in NumPy or by ``vmap`` in JAX, extensible to general nonlinear laws.

**Proposal.** Replace block-diagonal operators made of arrays of objects (``BlockDiagonalOperator2D`` of ``FourthRankIsotropicTensor``, as in the :doc:`tutorial <tutorials/square_basic/square_basic>`: one Cython object per cell, one indirect call per cell) with *functions of fields*: a local operator maps a field :math:`\boldsymbol\tau` to a field :math:`\boldsymbol\eta`, both arrays of shape ``(*grid_shape, n)``. The microstructure is described by its phases, not cell by cell.

Local operators are not part of the library: they are written in complete examples, without compilation. The library only provides small helper functions, which return the Mandel–Voigt matrices of usual fourth-rank tensors (e.g. isotropic, cubic), in NumPy. The interface between the local operators and the discrete Green operator only consists of fields: the operator passed to the Krylov solver combines them in Python, with one call per field for each part, so that both parts need not be written with the same tool.

- Linear laws with few phases: pure NumPy, with a phase map (array of integers) and one precomputed matrix per phase, and a Python loop over the *phases*, not over the cells. Temporaries are limited to the size of each phase.
- Many phases (e.g. polycrystals with thousands of grains), where a Python loop over the phases is slow, and where storing one matrix per cell would cost :math:`n^2` reals per cell (six times a field in 3D): Numba, with an ``@njit`` loop over the cells, which reads the matrix of each cell from a table indexed by the phase, and creates no temporary.
- Hyperelastic laws at finite strain (milestone 0.6), whose stress and tangent operator are naturally evaluated cell by cell: Numba as well; the tangent operators are written by hand.
- With JAX, the same functions of fields can be written with ``jax.numpy``, and differentiated through an adapter of the discrete Green operator (E13).

**Gains.** The goal of prototyping without compiling is recovered for the local operators, i.e. for the constitutive laws and the microstructure, which are the parts most often modified; Cython is only kept where it is required, for the Green operator. No object per cell, and no indirect call per cell. Numba is a dependency of some examples only, not of the library.

**Losses.** Arbitrary heterogeneous objects per cell are no longer supported (they could not be written in Python anyway, see the first friction point). Laws with internal variables are out of scope (see *Goals*).

**Consequences.** ``BlockDiagonalOperator2D/3D`` and the fourth-rank tensor classes of ``janus/operators.pyx`` are deleted at the switch (milestone 0.4), together with the rest of the current API; the fourth-rank tensor classes are replaced by the helper functions mentioned above (e.g. in ``janus/mandelvoigt.py``), and the tutorials are ported to local operators written in NumPy.

**Implementation problems.** The representation of hyperelastic laws (stress and tangent operator as functions of the deformation gradient, material parameters per phase) is to be decided in milestone 0.6.

**Recommendation.** Milestone 0.4: the deletion of ``BlockDiagonalOperator2D/3D`` requires the tutorials to be ported to local operators written in NumPy. An example with many phases (e.g. a polycrystal), whose local operator is written with Numba, in milestone 0.5; local operators written with Numba for hyperelastic laws in milestone 0.6.

E9. Generic tensor representations
----------------------------------

**Proposal.** Describe the local data of a field by a small object (vector of size ``dim``, symmetric second-rank tensor in Mandel form, full second-rank tensor…), instead of assuming a vector of Mandel–Voigt components. The layout ``(*grid_shape, *local_shape)`` (grid axes first, local components last) should be kept: it is the current layout, and it is natural for NumPy broadcasting (``matmul`` acts on the last two axes).

**Gains.** Conductivity and Darcy flow (vector fields), and finite strain (non-symmetric deformation gradients, 9 components in 3D) become possible. The Mandel form (``janus/mandelvoigt.py``) remains available for symmetric tensors; being orthonormal, it keeps transposes and scalar products simple.

**Losses.** Some added complexity in the API.

**Recommendation.** Milestone 0.5 (vectors), and 0.6 (full tensors, for the finite strain prototype).

E10. Solvers and interoperability
---------------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decisions of 2026-09-25 and 2026-09-26 (see :doc:`claude`, sections *The continuum Green operator in the new architecture* and *Automatic differentiation with JAX*). It initially proposed adapters to the JAX solvers, and a Newton–Krylov loop for general nonlinear problems.

**Proposal.** Janus does not provide solvers, on purpose. For prototyping, however, a thin layer would reduce friction: the library provides an adapter of the discrete Green operator to ``scipy.sparse.linalg.LinearOperator`` (flattened vectors). Reference implementations of the basic scheme of Moulinec & Suquet and of a Newton–Krylov loop for hyperelastic problems at finite strain (with tangent operators written by hand) are written in the examples, not in the library, like the local operators (E8). JAX solvers are not adapted by the library: the example with JAX handles its own solver, differentiated implicitly (see E13).

**Gains.** The tutorial becomes shorter; users compare schemes on a common basis, and can copy them from the examples.

**Losses.** Scope creep: the adapter must remain minimal.

**Recommendation.** Adapter in milestone 0.4; example with the basic scheme in 0.5; example with a Newton–Krylov loop in 0.6, with the finite strain prototype.

E11. Packaging, continuous integration and distribution
-------------------------------------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decision to keep Cython (see E3, and :doc:`claude`, section *The continuum Green operator in the new architecture*). It initially assumed that Janus would become a pure Python package, distributed as a universal wheel.

**Proposal.** Janus remains a compiled package (Cython extensions), but no longer links to FFTW after the switch (E2). Configuration in ``pyproject.toml`` as far as possible (``setup.cfg`` disappears with the link to FFTW); precompiled wheels for Linux, Windows and macOS, and for each supported version of Python, built with ``cibuildwheel``, together with a source distribution; publication on PyPI (and conda-forge); continuous integration on GitHub Actions (tests, doctests, execution of the examples, see E12, possibly the deployment of the documentation to GitHub Pages).

**Gains.** ``pip install`` in seconds, without compiler, on the platforms for which wheels are provided; the fragile steps identified in :doc:`claude` (locating FFTW, ``setup.cfg``) disappear.

**Losses.** A build matrix (platforms × versions of Python) to be maintained, and longer builds in continuous integration; on platforms without wheels, installing from the source distribution still requires a C compiler (but no longer FFTW).

**Implementation problems.**

- The name ``janus`` is already taken on PyPI (by an unrelated package), so that a distribution name must be chosen (the import name can remain ``janus``, although a distinct import name avoids clashes in environments that contain both packages).
- Minimum supported versions of Python and NumPy should follow a policy such as `SPEC 0 <https://scientific-python.org/specs/spec-0000/>`_; each supported version of Python adds a set of wheels.
- ``pyproject.toml`` does not declare the runtime dependencies yet (NumPy, and SciPy after E2).
- The extensions use typed memoryviews, not the C API of NumPy: the wheels therefore do not depend on the version of NumPy they are built against.

**Recommendation.** Continuous integration as early as milestone 0.2 (done: tests and doctests on Linux and Windows, at each push and once a week, with the current code); macOS, wheels and publication in milestone 0.4, once the link to FFTW is removed.

E12. Testing strategy for the rewrite
-------------------------------------

.. note:: This section was revised on 2026-09-26, for consistency with the decision to keep Cython and to develop the new core alongside the current code (see :doc:`claude`, section *The continuum Green operator in the new architecture*). It initially required the results not to depend on the backend, and stated that the discrete Green operator is a projector, which only holds for some schemes (see below).

**Proposal.** Keep the reference data (``tests/data/*.npz``) as a regression oracle. Until the switch (milestone 0.4), the new core is developed alongside the current code, which remains an executable reference: both are compared on the same grids; afterwards, the last release with the current API (tagged at 0.4) can still generate additional references if needed. Add tests of mathematical properties, independent of the implementation:

- the discrete Green operator is symmetric, for every scheme (a property of its interface, on which automatic differentiation relies, see E13);
- it annihilates uniform fields;
- the continuum Fourier symbol is homogeneous of degree zero;
- :math:`\Gamma_0 \, \mathbf C_0 \, \Gamma_0 = \Gamma_0` (projector), for the schemes for which it holds.

The last property does not hold for every scheme. On 2026-09-26, it was checked on the current code (isotropic elasticity, 2D and 3D): it holds to within about :math:`10^{-16}` for ``willot2015`` on all grids, and for the truncated scheme on grids of odd sizes; it fails for the truncated scheme on grids of even sizes (relative error of order :math:`10^{-1}`, presumably because of the Nyquist frequencies, see E4), and for the filtered scheme on all grids (a weighted average of projectors is not a projector). The symmetry and the annihilation of uniform fields hold for the six discrete Green operators. Each scheme must therefore state which properties it satisfies, and the tests must be parametrized accordingly.

Since the local operators and the reference schemes are written in the examples (E8, E10), the examples are run by continuous integration (E11), so that they do not silently break.

**Gains.** Confidence in the rewrite, and tests that remain valid for new physics and new schemes.

**Recommendation.** From milestone 0.3 on.

E13. Automatic differentiation through an adapter
-------------------------------------------------

.. note:: This evolution was added on 2026-09-26, after a discussion with the author (see :doc:`claude`, section *Automatic differentiation with JAX*).

**Context.** Once Cython is kept (E3), the discrete Green operator is the only part of the computation that the library provides: local operators are functions of fields (E8), written by the user in NumPy or Numba, and the solvers come from other libraries (E10). An example in which the local operators and the solver are written with JAX therefore only needs JAX to see through *one* opaque function, the discrete Green operator.

**Proposal.** A small adapter wraps the discrete Green operator for JAX: ``jax.pure_callback`` calls the Cython code on concrete arrays (also under ``jit``), and ``jax.custom_vjp`` provides its derivative, which JAX then combines with the rest of the computation by the chain rule. Since the discrete Green operator is linear and symmetric, its vector-Jacobian product with respect to the polarization is the operator itself: no new Cython code is needed::

    @jax.custom_vjp
    def gamma(tau):
        return jax.pure_callback(
            lambda t: np.asarray(op.apply(np.asarray(t))),
            jax.ShapeDtypeStruct(tau.shape, tau.dtype), tau)

    def gamma_fwd(tau):
        return gamma(tau), None

    def gamma_bwd(_, eta_bar):
        return (gamma(eta_bar),)

    gamma.defvjp(gamma_fwd, gamma_bwd)

The adapter is first written in an example, which shows that automatic differentiation remains possible with a Cython core (e.g. sensitivity of the effective response to the properties of the phases). It may move to an optional module of the library later, if several examples need it; JAX is not a dependency of the library.

**Gains.** Automatic differentiation of a complete computation, without giving up the Cython core, and at the cost of a few lines of user code.

**Losses.** Reverse mode only: JAX cannot transpose a ``pure_callback``, so that forward mode (``jvp``, ``jacfwd``) is not available; higher-order derivatives in reverse mode remain possible, since the backward pass calls the same wrapped function. The Green operator runs on the host CPU, blocks the fusion of the surrounding code by XLA, and is applied sequentially under ``vmap``; ``jax_enable_x64`` must be set to work in double precision. Registering the Cython kernel as an XLA custom call (``jax.ffi``) would lift some of these limits, at a much higher cost.

**Consequences on the library.** The symmetry of the discrete Green operator becomes part of its *interface*: it is guaranteed by a test, for each scheme, and documented (see E12). The current code satisfies it: on 2026-09-26, :math:`\langle \mathbf y, \Gamma \mathbf x \rangle = \langle \Gamma \mathbf y, \mathbf x \rangle` was checked to within about :math:`10^{-16}` (relative) for the six discrete Green operators (truncated, filtered, ``willot2015``, in 2D and 3D), including on grids of odd sizes. Symmetry relies on the orthonormal Mandel form (E9), and must be preserved by the new core and by new schemes (E6).

**Implementation problems.**

- The solver must not be differentiated through its iterations. The robust approach is implicit differentiation: a ``custom_vjp`` around the whole solve, whose backward pass solves the adjoint problem, whose operator (e.g. :math:`\mathbf I + \delta\mathbf C : \Gamma_0` for the Lippmann–Schwinger operator :math:`\mathbf I + \Gamma_0 \, \delta\mathbf C`) only involves the Green operator and the local operator. Whether the solvers of ``jax.scipy.sparse.linalg`` can be differentiated directly when their operator contains the adapter (they rely on ``custom_linear_solve``, which needs to linearize or transpose that operator) is to be checked by a prototype.
- Derivatives with respect to the parameters of the reference medium would require new Cython kernels (e.g. :math:`\partial \Gamma / \partial \mu_0`). They are simple in closed form, but rarely needed, the reference medium being a numerical parameter.

**Recommendation.** An example with JAX, in milestone 0.5, once local operators are described as fields (E8); the symmetry test from milestone 0.3 on (E12).

Milestones
==========

.. note:: The milestones were revised on 2026-09-25, following the decision to keep Cython (see E3, and :doc:`claude`, section *The continuum Green operator in the new architecture*).

The evolutions above are ordered as follows. Each milestone leaves the library in a usable, tested state.

**0.2 — Clean-up (done).** MPI was removed (E1), and the last commit with MPI was tagged ``Farewell_MPI``. Continuous integration runs on Linux and Windows with the current Cython code, at each push and once a week (E11).

**0.3 — New core, in Cython.** New implementation, developed *alongside* the current code (e.g. in a new subpackage), so that the current code remains an executable reference throughout the rewrite:

- stateless operators, applied one frequency at a time, with :math:`\hat\Gamma(\mathbf 0) = \mathbf 0` by definition (E4); the ``apply`` method of the physics takes the dimension as a parameter, so that E5 does not change it later;
- separation of discretization and physics (E6); ``scripts/gencode.py`` is deleted;
- FFT through ``scipy.fft``, called from Python once per field, the loop over frequencies remaining in Cython (E2);
- isotropic linear elasticity with the three existing schemes (truncated, filtered, ``willot2015``), validated against the reference data and by the mathematical properties of the Green operator, and compared with the current code on the same grids (E12);
- benchmark against the current code, on 2D and 3D grids, including the FFT. Since the tool is unchanged, the acceptance criterion is the absence of regression.

**0.4 — Switch and distribution.** The last release with the current API is tagged; the new core becomes the only one, and the current API (``set_frequency``, generated classes, FFTW wrapper, ``BlockDiagonalOperator2D/3D`` and the fourth-rank tensor classes) is deleted, together with the link to FFTW and ``setup.cfg``. Local operators become functions of fields, written in NumPy, with a phase map and a loop over the phases, and the fourth-rank tensor classes are replaced by helper functions returning Mandel–Voigt matrices (E8); the tutorials and the documentation are ported accordingly; adapter to SciPy solvers (E10). Precompiled wheels for Linux, Windows and macOS are built with ``cibuildwheel``, and published on PyPI (E11).

**0.5 — Second physics.** Conductivity or Darcy flow (vector fields, E9), written in Cython, as a validation of the extension mechanism, with a user guide on adding physics and discretizations. Dimension-generic code (E5). An example with many phases (e.g. a polycrystal), whose local operator is written with Numba (E8); an example with the basic scheme of Moulinec & Suquet, as a reference (E10). An example with JAX, differentiating a complete computation through an adapter of the discrete Green operator (E13).

**0.6 — Finite strain.** Prototype with full (non-symmetric) second-rank tensors (E9), hyperelastic constitutive laws (no internal variables), and a Newton–Krylov loop (E10); the tangent operators are written by hand. The representation of hyperelastic laws is decided, and their local operators are written with Numba (E8).

**1.0 — Stabilization.** API freeze, complete documentation, removal of deprecated code, release on PyPI and conda-forge.

The order matters in three places. The new core is developed alongside the current code, rather than by refactoring it in place, so that both can be compared on the same grids until the switch. The FFT moves to ``scipy.fft`` as early as 0.3, so that the benchmark covers it, and so that the build no longer depends on FFTW when the wheels are produced in 0.4. The dimension-generic code (E5) comes after the switch, because 0.3 is already the largest milestone, and because E5 does not change the interface of the physics, which takes the dimension as a parameter from the start.

Decisions required from the author
==================================

.. note:: This list was revised on 2026-09-26: the decisions on the default backend (NumPy with optional JAX, or JAX only) and on the acceptable slowdown of a pure Python implementation became obsolete when Cython was kept (see E3).

- Distribution name on PyPI (``janus`` is taken), and import name.
- Data layout: keep grid axes first and local components last (recommended, see E9)?
- Precision: ``float64`` only, or ``float32`` as well (which halves the memory footprint of large grids; Cython's fused types allow it, see E3)?
- Minimum supported versions of Python, NumPy and SciPy (see E11).
- Tolerance of the benchmark: strict absence of regression with respect to the current code (acceptance criterion of milestone 0.3), or an accepted margin, in exchange for the genericity introduced by E6 (indirect calls to the scheme and to the physics) and E5 (loops with run-time bounds)?
