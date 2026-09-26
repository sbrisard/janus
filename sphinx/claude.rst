*********************************
Claude's contributions to `Janus`
*********************************

As of august 2026, `Janus` is revived by the author with the help of `Claude Code`. This page will collect all interactions between the author and Claude.


2026-09-26 • Make the roadmap consistent with the decisions of 2026-09-25
=========================================================================

Priority task. The decisions recorded in *The continuum Green operator in the new architecture* (below) were applied to sections *E3*, *E4*, *E7* and *Milestones* of the :doc:`roadmap` only. Update the other parts of the roadmap, which still assume that Cython is replaced by NumPy/JAX. The list established at the end of that section is the starting point:

- *Goals*: new physics implementable without compilation, GPU not to be ruled out;
- friction point 2, which presents the per-frequency design as a problem;
- *E2*: its recommendation contradicts milestone 0.3 (it refers to the array namespace of *E7*, and does not state that the link to FFTW disappears with ``scipy.fft``);
- *E5*, *E6* (new schemes "in a few lines of Python"), *E8*, *E10*, *E11*, *E12*;
- *E8* and milestones 0.4 and 0.5 must also reflect the decision on local operators (see *Follow-up: local operators*): functions of fields, reference implementation in pure NumPy (phase map, loop over the phases), Numba as an optional dependency, removal of ``BlockDiagonalOperator2D/3D`` at milestone 0.4;
- the scope of the constitutive laws (see *Follow-up: scope of the constitutive laws*): linear laws, and hyperelasticity at finite strain, without internal variables. Remove or qualify the mentions of general nonlinear laws (*Goals*, *E8*: "extensible to nonlinear laws", tangent operators by automatic differentiation; *E10*);
- *Decisions required from the author*: *Default backend* and *Acceptable slowdown* are obsolete;
- the introduction (see *Status of the roadmap*);
- the mentions of JAX and of automatic differentiation (*Goals*, *E1*, *E5*, *E8*, *E10*, *E12*), to be aligned with *E13* (see *Automatic differentiation with JAX*): JAX is no longer a backend, but a use case shown by an example.

Start with *E2*. Report the proposed changes before applying the ones that go beyond consistency (i.e. that would express a new decision).

Claude's report
---------------

**E2** (2026-09-26). Changes applied, for consistency only:

- a note states that the section was revised, and why;
- *Gains* of ``scipy.fft``: "no additional dependency (SciPy is needed anyway for iterative solvers)" was inaccurate: NumPy is currently the only runtime dependency of Janus (see ``environment.yml``; SciPy is not even installed in the ``janus`` environment). The text now says that SciPy becomes a runtime dependency, distributed as wheels, and needed anyway by the users for the solvers;
- *Losses* of ``scipy.fft``: the loss of the fine tuning of FFTW is measured by the benchmark of milestone 0.3;
- new paragraph *Keeping the current wrapper*: with Cython kept, this option had to be discussed; it records the reasons of *Follow-up: milestones* (negligible cost of a call from Python once per field, link to FFTW required by ``setup.cfg`` and awkward for wheels);
- *Recommendation*: the reference to the array namespace of *E7* was removed; the FFT is called from Python, the loop over frequencies remaining in Cython; ``scipy.fft`` from milestone 0.3 on, deletion of the wrapper, of the link to FFTW and of ``setup.cfg`` at milestone 0.4.

Point going beyond consistency, submitted to the author, approved and applied: ``scipy.fft`` has no ``out`` argument, so that each transform allocates its result, whereas the current discrete Green operators preallocate two Fourier buffers (``dft_x``, ``dft_y``, i.e. two complex fields) and the in-place semantics of ``apply(x, y)`` is kept (*E7*). Claude proposed to add to the *Implementation problems* of *E2* (now done): transform the whole field at once (``rfftn`` over the grid axes), apply the Green operator *in place* on the Fourier coefficients (with a small temporary vector per frequency, which removes the second buffer), and write the inverse transform component by component into ``y``, so that the temporaries are one complex field and one real component; the layout of the output of ``rfftn`` (contiguity, strides) seen by the Cython loop is to be checked.

**Goals** (2026-09-26). Changes applied, for consistency only; a note records the initial wording:

- installation: "without compiler" now relies on precompiled wheels;
- new discretizations and new physics are written in Cython, within the library, mainly by the author (decision 4); the local operators are written without compilation, in NumPy, or in Numba where NumPy is not suitable (*Follow-up: local operators*);
- new item on the scope of the constitutive laws: linear laws and hyperelasticity at finite strain; laws with internal variables are out of scope (*Follow-up: scope of the constitutive laws*); "finite strain solid mechanics" became "finite strain hyperelasticity";
- GPU execution: the requirement that the architecture should not rule it out was dropped, GPU execution being "not a priority" (decision 4); automatic differentiation remains possible through *E13*.

Found along the way, not in the initial list: the *Gains* of *E6* state that "new schemes can be added in a few lines of Python", whereas, with *E4* as revised, the modified wave-vectors are computed on the fly in the Cython loop, so that a new scheme is written in Cython. To be corrected at the same time as the other evolutions.

**Friction point 2** (2026-09-26). Changes applied, for consistency only; a note records the initial wording. The title "Operators are scalar and stateful" became "Operators are stateful": evaluating the physics one wave-vector at a time is the design retained by decision 3, and is no longer presented as a problem. The problem is the hidden state shared by ``set_frequency`` and ``apply`` (no sharing between threads, hence no parallel loop over frequencies; result depending on a previous call); this is also how *E3* already refers to this friction point. The local operators, already mentioned by the initial wording ("cells"), are described as in *Follow-up: local operators* (one Cython object per cell, indirect call per cell, microstructure described cell by cell). The remaining consequence of the per-frequency design (a physics must be compiled to be fast) is kept, with a reference to *E3*.

**E5** (2026-09-26). Changes applied; a note records that the section initially assumed a core written in NumPy or JAX:

- *Proposal*: the dimension is a run-time value; E5 concerns the Cython part only (discrete Green operators and physics), the local operators written in NumPy (*E8*) being dimension-generic at no cost;
- *Losses*: the argument on NumPy and JAX was replaced by its Cython counterpart (loops with run-time bounds, which the C compiler cannot unroll; slowdown to be measured);
- *Consequences*: the ``apply`` method of the physics takes the dimension as a parameter from milestone 0.3 on (*Follow-up: milestones*);
- *Recommendation*: "Adopt together with E4" contradicted the milestones; it became "milestone 0.5, after the switch", with the reasons of *Follow-up: milestones*.

New paragraph *Implementation problems*, which develops the approach already stated in *E3* (loop over a flattened frequency index) without deciding anything: the number of dimensions of a memoryview is fixed at compile time, hence a view of shape ``(N, n)`` of the Fourier coefficients (which depends on the layout of the output of ``rfftn``, see *E2*), and the wave-vector recovered from the flattened index by integer divisions or by an incremented multi-index, whose costs are to be measured.

**E6** (2026-09-26). Changes applied; a note records the initial wording:

- *Proposal*: new paragraph stating that a scheme, like the physics, is written in Cython and is stateless (wave-vectors and weights computed on the fly for each discrete frequency, in buffers provided by the caller), the loop over frequencies, the calls to the physics and the weighted sum being generic; this transposes *E4* to the schemes;
- *Gains*: "a few lines of Python" became "a few lines of Cython";
- *Losses*: the initial argument ("instead of combining precomputed matrices; this is also the case in the current code") did not describe the current code accurately: ``FilteredGreenOperator2D/3D`` forms the matrix of the symbol at the 4 or 8 wave-vectors, combines these matrices, then applies the result. With *E4*, the symbol is applied 4 or 8 times and the results are summed, which should cost about the same. Claude added that the calls from the generic loop to the scheme and to the physics are indirect (``cdef`` methods), hence not inlined; both points are to be checked by the benchmark of milestone 0.3;
- *Implementation problems*: complex wave-vectors are presented as one of the open points of the signature of the physics (*E4*), rather than as a requirement ("should" became "may have to").

The *Recommendation* (milestone 0.3) was already consistent with the milestones.

**E8, milestones 0.4, 0.5 and 0.6** (2026-09-26). Changes applied, for consistency; a note records the initial wording of *E8*:

- *E8* rewritten according to *Follow-up: local operators*: local operators are functions of fields, on arrays of shape ``(*grid_shape, n)``; pure NumPy for linear laws with few phases (phase map, one matrix per phase, loop over the phases); Numba where NumPy is not suitable (many phases, hyperelasticity); the interface with the Green operator only consists of fields. The mentions of general nonlinear laws, of ``vmap`` and of tangent operators obtained by automatic differentiation were removed; tangent operators are written by hand, and JAX is mentioned through *E13* only;
- following the author's statement of 2026-09-26 (*Automatic differentiation with JAX*), local operators are written in complete examples, *not* in the library; Numba is therefore a dependency of some examples, not of the library. The *Follow-up: local operators* of 2026-09-25 spoke of a "reference implementation in pure NumPy", without saying where it would live; *E8* now follows the later statement;
- *Recommendation* of *E8*: milestone 0.4 instead of 0.5, since the deletion of ``BlockDiagonalOperator2D/3D`` at the switch requires the tutorials to be ported to local operators written in NumPy;
- milestone 0.4: ``BlockDiagonalOperator2D/3D`` and the fourth-rank tensor classes are listed among the deleted API; local operators become functions of fields, and the tutorials are ported accordingly;
- milestone 0.5: "Local operators described as fields, e.g. phase maps (E8)" was removed (moved to 0.4), and "reference iterative schemes (E10)" became a sentence of its own;
- milestone 0.6: the representation of hyperelastic laws is decided there, and their local operators are written with Numba.

Points going beyond consistency, *not* applied, submitted to the author:

- an example with many phases (e.g. a polycrystal), whose local operator is written with Numba, in milestone 0.5. The decisions say *why* Numba is needed, not *when* such an example is written. (Claude had first applied it, then reverted it, since the task asks for such changes to be proposed first);
- the fourth-rank tensor classes (``FourthRankIsotropicTensor2D/3D``, ``FourthRankCubicTensor2D``) are deleted at 0.4 with the rest of the local operators written in Cython, as stated by the author on 2026-09-26 (this is applied). The initial *E8* proposed to turn them into small functions returning Mandel–Voigt matrices, i.e. to keep helpers in the library; whether the library should provide such helpers (e.g. in ``janus/mandelvoigt.py``), or leave them to the examples, is not decided.

The author validated these changes on 2026-09-26, and decided both points: the example with many phases, written with Numba, is planned in milestone 0.5; the library provides helper functions returning the Mandel–Voigt matrices of usual fourth-rank tensors. Both decisions were applied to *E8* (*Proposal*, *Consequences*, *Recommendation*) and to milestones 0.4 (helper functions replacing the fourth-rank tensor classes) and 0.5 (example with Numba).

**E10** (2026-09-26). Changes applied, for consistency; a note records the initial wording:

- the adapters to the JAX solvers were removed: JAX is no longer a backend, and the example with JAX handles its own solver, differentiated implicitly (*E13*);
- "a Newton–Krylov loop for nonlinear problems" became "for hyperelastic problems at finite strain, with tangent operators written by hand" (*Follow-up: scope of the constitutive laws*);
- *Recommendation*: "reference schemes in 0.5" was split: basic scheme in 0.5, Newton–Krylov loop in 0.6, since the latter needs the finite strain prototype; milestones 0.4 ("adapter"), 0.5 (basic scheme) and 0.6 (reference to *E10*) were aligned.

Point *not* applied, submitted to the author: on 2026-09-26, the author stated that "the rest of the chain" (everything but the Green operator) will no longer be developed in the library, but in complete examples. Taken literally, this also applies to the reference schemes (basic scheme, Newton–Krylov loop), which *E10* still places in the library, and possibly to the adapter to ``scipy.sparse.linalg.LinearOperator``. Claude proposes: the adapter stays in the library, since it only wraps the discrete Green operator; the reference schemes are written in the examples.

The author validated these changes on 2026-09-26, and approved the proposal, which was applied to *E10* (*Proposal*, *Gains*, *Losses*, *Recommendation*) and to milestone 0.5 (example with the basic scheme).

**E11** (2026-09-26). Changes applied, for consistency; a note records the initial wording ("once Cython is gone", pure Python package, universal wheel):

- *Proposal*: Janus remains a compiled package, without link to FFTW after the switch; precompiled wheels for Linux, Windows and macOS and for each supported version of Python (``cibuildwheel``), and a source distribution, as decided in *Follow-up: milestones*;
- *Gains*: ``pip install`` without compiler holds on the platforms for which wheels are provided;
- new *Losses*: build matrix to be maintained; a C compiler (but no longer FFTW) is still needed to install from the source distribution;
- *Recommendation*: continuous integration is marked as done at 0.2, as it actually runs (``.github/workflows/tests.yml``: tests and doctests on Linux and Windows, at each push and once a week); macOS, wheels and publication at 0.4, in line with the milestones.

Two facts found while checking the code, added to the *Implementation problems*: ``pyproject.toml`` does not declare any runtime dependency yet (NumPy, and SciPy after *E2*); the extensions use typed memoryviews and not the C API of NumPy, so that the wheels do not depend on the version of NumPy they are built against.

**E12** (2026-09-26). Changes applied; a note records the initial wording:

- "the results do not depend on the backend" was removed (there is no backend any more); it is replaced by the comparison of the new core with the current code on the same grids until the switch, as stated in milestone 0.3, the tagged release of 0.4 taking over afterwards;
- factual correction: the initial text stated that the discrete Green operator is a projector (:math:`\Gamma_0 \mathbf C_0 \Gamma_0 = \Gamma_0`). Claude checked it on the current code (isotropic elasticity, grids of even and odd sizes, 2D and 3D): it holds for ``willot2015`` on all grids and for the truncated scheme on grids of odd sizes, but fails for the truncated scheme on grids of even sizes (relative error of order :math:`10^{-1}`, presumably because of the Nyquist frequencies) and for the filtered scheme on all grids. Symmetry and annihilation of uniform fields hold for all six operators. The list of properties is now a bulleted list, the projector property being restricted to the schemes for which it holds; each scheme must state which properties it satisfies.

Points *not* applied, submitted to the author:

- the failure of the truncated scheme on grids of even sizes is directly related to the open point of *E4* on the Nyquist frequencies; Claude proposes to add a reference to this observation in the *Implementation problems* of *E4*;
- since local operators and reference schemes now live in the examples (*E8*, *E10*), the examples should be run by continuous integration, so that they do not silently break; Claude proposes to add this to *E12* (and to the continuous integration of *E11*).

The author validated these changes on 2026-09-26, and approved both proposals, which were applied: *E4* (Nyquist frequencies) refers to the observation of *E12*; *E12* states that the examples are run by continuous integration, and *E11* lists their execution among the jobs of continuous integration.

**Decisions required from the author** (2026-09-26). Changes applied, for consistency; a note records the removed items:

- *Default backend* and *Acceptable slowdown* (of a pure Python implementation) were removed, as obsolete;
- *Precision*: "relevant on GPU" became "which halves the memory footprint of large grids; Cython's fused types allow it", GPU execution being no longer a priority;
- *Minimum supported versions*: SciPy was added, since it becomes a runtime dependency (*E2*, *E11*).

Point *not* applied, submitted to the author: the acceptance criterion of milestone 0.3 is the absence of regression with respect to the current code. However, the genericity introduced in 0.3 (*E6*: indirect calls to the scheme and to the physics, not inlined) and in 0.5 (*E5*: loops with run-time bounds) may cost some performance, which *E5* and *E6* leave "to be measured". Claude proposes to add a decision: *Tolerance of the benchmark*: strict absence of regression, or an accepted margin (to be set by the author) in exchange for genericity?

The author validated these changes on 2026-09-26, and approved the proposal: the decision *Tolerance of the benchmark* was added to the list.

**Introduction, E1 and E7** (2026-09-26). The last sentence of the introduction was replaced, as decided by the author (see *Status of the roadmap*, formerly *TO DISCUSS*). In *E1*, the last mentions of JAX were removed ("JAX on GPU" became "``prange`` loops in Cython or Numba"; "JAX sharding" was dropped). In *E7*, the reasons why the evolution was not adopted no longer state that automatic differentiation is not a priority, but that it remains possible without *E7* (*E13*).

Summary
~~~~~~~

All the items of the task were processed, in this order: *E2*, *Goals*, friction point 2, *E5*, *E6* (added to the list on the way), *E8* with milestones 0.4, 0.5 and 0.6, *E10*, *E11*, *E12*, *Decisions required from the author*, and the introduction, together with the last mentions of JAX in *E1* and *E7*. Each revised section carries a note that records its initial wording. A final search of the :doc:`roadmap` for "JAX", "backend", "nonlinear", "vmap", "pure Python", "array namespace" and "GPU" only finds the historical analysis of *E3*, the non-adopted *E7*, *E13*, and the notes.

Besides consistency, the author took the following decisions during the task, all applied:

- *E2*: handling of the temporaries of ``scipy.fft`` (whole field transformed at once, Green operator applied in place, inverse transform component by component);
- *E8*: an example with many phases, written with Numba, in milestone 0.5; helper functions returning the Mandel–Voigt matrices of usual fourth-rank tensors are provided by the library;
- *E10*: only the adapter to ``scipy.sparse.linalg.LinearOperator`` is in the library; the reference schemes are written in the examples;
- *E4*/*E12*: reference to the Nyquist observation; the examples are run by continuous integration (*E11*);
- new decision to be taken: *Tolerance of the benchmark*.

Two checks were made on the current code, with results recorded in the :doc:`roadmap`: the six discrete Green operators are symmetric and annihilate uniform fields (*E12*, *E13*); the projector property holds for ``willot2015``, and for the truncated scheme on grids of odd sizes only, never for the filtered scheme (*E12*). Two facts were found in the configuration: SciPy is not a dependency yet (*E2*), and ``pyproject.toml`` declares no runtime dependency (*E11*).

Open points left to later milestones (not part of this task): signature of the ``cdef`` methods and representation of the local tensors (*E4*, *E9*), behaviour at the Nyquist frequencies (*E4*), layout of the output of ``rfftn`` (*E2*, *E5*), representation of hyperelastic laws (*E8*, milestone 0.6), and whether the solvers of ``jax.scipy.sparse.linalg`` can be differentiated through the adapter (*E13*).

The documentation builds without warning; ``docs/`` was not regenerated, and nothing was committed. The author validated this summary on 2026-09-26, and the task was marked as done.


2026-09-26 • Automatic differentiation with JAX
===============================================

Not a task: record of a discussion held in the console. The author asked whether, Cython being kept, automatic differentiation with JAX could be recovered by implementing the missing derivatives of the Green operators by hand, JAX then applying the chain rule.

Discussion
----------

Claude answered that it can: JAX cannot differentiate code it does not see, but ``jax.pure_callback`` makes the Cython code callable from JAX (also under ``jit``), and ``jax.custom_vjp`` provides its derivative, which JAX combines with the rest of the computation by the chain rule. With respect to the polarization, the discrete Green operator is linear, so that its vector-Jacobian product is its transpose; Claude checked numerically that the six current discrete Green operators (truncated, filtered, ``willot2015``, in 2D and 3D) are symmetric, to within about :math:`10^{-16}` (relative), including on grids of odd sizes. The derivative to be provided is therefore the operator itself: no new Cython code is needed. Only the derivatives with respect to the reference medium (:math:`\mu_0`, :math:`\nu_0`) would require new kernels, but they are rarely needed.

Claude listed the limits: reverse mode only (JAX cannot transpose a ``pure_callback``), execution on the host CPU, no fusion by XLA across the call, sequential ``vmap``, ``jax_enable_x64`` required; the solver should be differentiated implicitly (a ``custom_vjp`` around the whole solve, the backward pass solving the adjoint problem), not through its iterations. Claude also objected that the rest of the chain, in particular the local operators, would have to be visible by JAX, whereas the decision of 2026-09-25 is to write them in NumPy.

The author answered that this is not a problem: the library will only provide the Green operator, the local operators written in Cython disappearing; the local operators will be written in complete *examples*, in NumPy or Numba, and nothing prevents an example based on JAX, precisely to show that it is possible. Claude agreed: in such an example, the local operators are written with ``jax.numpy``, and the discrete Green operator is the only opaque function. The remaining points belong to the example, not to the architecture:

- the adapter (a dozen lines) is written in the example, and may move to an optional module of the library later; JAX is not a dependency of the library;
- the solver is differentiated implicitly, in the example; whether the solvers of ``jax.scipy.sparse.linalg`` can be differentiated directly when their operator contains the adapter is to be checked by a prototype;
- the example relies on the symmetry of the discrete Green operator, which must therefore become a property of its interface, guaranteed by a test for each scheme and preserved by the new core.

JAX thus becomes a use case, shown by an example, and not a backend; this is consistent with the abandonment of *E7*.

Claude's report
---------------

At the author's request, the discussion was integrated into the :doc:`roadmap`:

- new evolution *E13. Automatic differentiation through an adapter* (added at the end, so that the numbering does not change): context, adapter code, limits, consequence on the library (symmetry of the discrete Green operator as a property of the interface, with the check made on the current code), implementation problems (implicit differentiation of the solver, derivatives with respect to the reference medium), and recommendation: an example with JAX in milestone 0.5, once local operators are described as fields (*E8*), and the symmetry test from milestone 0.3 on;
- *E3*: the last paragraph no longer states that automatic differentiation is out of reach with Cython, and refers to *E13*;
- *E12*: symmetry is marked as a property of the interface, on which *E13* relies;
- milestone 0.5: the JAX example is added. Its placement in 0.5 is a proposal of Claude;
- the introduction of *Analysis of the proposed evolutions* mentions *E13*.

The other mentions of JAX and of automatic differentiation in the :doc:`roadmap` (*Goals*, *E1*, *E5*, *E8*, *E10*) were not modified: they were added to the list of the task *Make the roadmap consistent with the decisions of 2026-09-25* (above). Note that the *Goals* ("GPU execution and automatic differentiation are not required, but the architecture should not rule them out") are now consistent with the decisions as far as automatic differentiation is concerned; only GPU execution is ruled out.


2026-09-25 • The continuum Green operator in the new architecture
=================================================================

Not a task: record of a discussion held in the console, started by the author with the question "what are the inputs and outputs of the function ``symbol(k)``?" (see *E4* of the :doc:`roadmap`). The discussion led to the decisions below, in this order; the last one reverses the recommendation of *E3*.

Decisions
---------

1. **Zero frequency.** :math:`\hat\Gamma_0(\mathbf 0) = \mathbf 0` is part of the *definition* of the Green operator, which acts on fields with zero mean. It is not a convenience value, and it does not depend on the loading. The macroscopic loading (imposed mean strain, mean stress, or mixed conditions) is handled by adapting the *equation*, not the Green operator. Claude had first argued that, under an imposed mean stress, the zero mode of the operator should be :math:`\mathbf C_0^{-1}`; the author objected that the corresponding term, :math:`\mathbf C_0^{-1} : (\boldsymbol\Sigma - \langle\boldsymbol\tau\rangle)`, belongs to the equation, which Claude agreed with.

2. **Apply, do not form the matrix.** The primitive of a continuum Green operator is the application :math:`\hat{\boldsymbol\eta} = \hat\Gamma(\mathbf k) \cdot \hat{\boldsymbol\tau}`, computed in closed form, not a function ``symbol(k)`` that returns a matrix: the matrix is not needed in general, and forming it costs time and memory. The matrix can be recovered by applying the operator to the vectors of the canonical basis, which is enough for the tests. A notation such as ``symbol(k) @ tau`` would require an intermediate object (a matrix, or a matrix-free linear operator), and was not retained.

3. **One frequency at a time.** Within a vectorized (NumPy) design, ``apply(k, tau)`` must receive the wave-vectors of the whole grid. Storing them costs about a quarter of a field in 3D elasticity (1.6 GB for a 512³ grid, against 6.4 GB for the Fourier coefficients of the polarization), and half of a field in 2D conductivity, whereas the current Cython code computes each wave-vector on the fly and stores nothing. Claude proposed workarounds (wave-vectors passed as a tuple of broadcastable components, evaluation by blocks), which the author found unconvincing: they are symptoms of the ban on loops over frequencies that comes with abandoning compiled code. The physics is therefore written for *one* wave-vector at a time, as in the mathematics and in the current code.

4. **Cython is kept.** The author will write new physics mainly by themselves, within the library, and GPU execution is not a priority. Under these conditions, compiling new physics is not an obstacle, and Cython remains a good choice: the existing code works and serves as a reference, it is fast and allocates nothing, and most friction points of the :doc:`roadmap` (stateful operators, Python subclasses silently ignored, restricted data model, and, to a large extent, the fragile build) are *design* problems, which can be fixed without changing the tool. This departs from the context of the task *The road to v1.0* below, which asked that new physics be implementable without compiling.

Open points
-----------

- Modified wave-vectors that vanish at a nonzero discrete frequency (e.g. at the Nyquist frequencies, for some schemes): with decision 1, the Green operator is zero there. Check that this is the intended behaviour for each scheme, and not a side effect.
- Representation of the local tensors (flattened Mandel–Voigt vectors, or full tensors), which fixes the sizes seen by ``apply``.
- Signature of the ``cdef`` method: raw pointers or memoryviews (slicing a memoryview at each frequency has a non-negligible cost in an inner loop); real or complex wave-vectors.
- Dimension-generic code in Cython (*E5*): this is where Cython helps least.
- A Numba prototype (generic kernel, truncated scheme, 2D conductivity, timed against the Cython code) was considered, then postponed: it may be done later, if the choice of Cython is to be confirmed by measurements.

Claude's report
---------------

At the author's request, sections *E3*, *E4* and *E7* of the :doc:`roadmap` were rewritten according to the above decisions.

- *E3*: the analysis of the three options (NumPy, Numba, JAX) is kept, shortened, and completed with the reasons why the vectorized design was rejected; the recommendation is reversed: Cython is kept, and its friction points are fixed by design.
- *E4*, renamed *Stateless operators*: the continuum Green operator exposes a stateless ``cdef`` method that applies its symbol at one wave-vector, without forming the matrix; :math:`\hat\Gamma(\mathbf 0) = \mathbf 0` by definition; the discrete operators compute the wave-vectors on the fly. The NumPy example was replaced by a Cython sketch for conductivity, whose signature is explicitly left open.
- *E7* is marked as not adopted: the array API and the functional interface lose their purpose once Cython is kept and JAX is no longer a goal; the in-place semantics of ``apply(x, y)`` is kept. The section is kept, so that the numbering of the evolutions does not change.

The other parts of the roadmap were *not* modified, and are now inconsistent with these decisions:

- *Goals*: "new physics … implementable in pure Python, without compilation" and "the architecture should not rule [GPU] out";
- friction point 2, which presents the per-frequency design as a problem;
- *E2*: the recommendation of ``scipy.fft`` goes "through the array namespace, see E7"; with Cython kept, the choice between the current FFTW wrapper, pyFFTW and ``scipy.fft`` must be reconsidered;
- *E5* (NumPy/JAX arguments), *E8* (vectorized constitutive functions, ``vmap``), *E10* (JAX solvers), *E11* ("once Cython is gone", pure Python package) and *E12* (independence from the backend);
- the milestones 0.3 (pure NumPy/SciPy core), 0.4 (removal of Cython and FFTW) and 0.6 (JAX backend), and two of the *Decisions required from the author* (default backend, acceptable slowdown of the pure Python implementation);
- the introduction, already discussed in *TO DISCUSS • Status of the roadmap*.

The documentation builds without warning; ``docs/`` was not modified.

Follow-up: milestones
~~~~~~~~~~~~~~~~~~~~~

The milestones of the :doc:`roadmap` were then discussed. The author approved Claude's three recommendations:

- the new core is developed *alongside* the current code (e.g. in a new subpackage), rather than by refactoring it in place, so that the current code remains an executable reference until the switch;
- the FFT moves to ``scipy.fft`` as early as milestone 0.3: it is called from Python once per field, the loop over frequencies remaining in Cython, which removes the link to FFTW (hence ``setup.cfg``) and eases the production of wheels; the loss of the fine tuning of FFTW (``FFTW_MEASURE``, wisdom) is to be measured by the benchmark;
- the dimension-generic code (*E5*) is postponed after the switch; the ``apply`` method of the physics takes the dimension as a parameter from the start, so that *E5* does not change it.

The *Milestones* section was rewritten accordingly: 0.2 is marked as done; 0.3 is the new core in Cython (*E4*, *E6*, ``scipy.fft``, elasticity with the three existing schemes, benchmark with no regression as acceptance criterion); 0.4 is the switch (deletion of the current API and of the link to FFTW, wheels built with ``cibuildwheel``, publication on PyPI); 0.5 is the second physics, in Cython, together with *E5*, *E8* and the reference schemes; 0.6 becomes a finite strain prototype, with hand-written tangent operators, the JAX backend being dropped; 1.0 is unchanged. The closing paragraph now explains the three ordering choices above.

Follow-up: local operators
~~~~~~~~~~~~~~~~~~~~~~~~~~

The author then raised the question of the *local* part of the linear operator, e.g. :math:`(\mathbf C - \mathbf C_0)^{-1} \cdot \boldsymbol\tau`. In the current code, it is what makes Janus particularly heavy: ``BlockDiagonalOperator2D/3D`` takes an array of local operators (``AbstractOperator[:, :]``), i.e. one Cython object per cell, with one indirect call per cell, and a microstructure described cell by cell rather than by its phases.

The situation differs from that of the Green operator. For the latter, vectorization required data to be *created* (the grid of wave-vectors, temporaries), which led to keeping Cython. For the local operator, the data already exist as fields (phase map, polarization), so that this objection does not apply. Moreover, the interface between both parts only consists of fields: the operator passed to the Krylov solver combines them in Python, with one call per field for each part, so that both parts need not be written with the same tool.

The author approved the following proposal:

- a local operator is a *function of fields*, ``tau ↦ eta``, acting on arrays of shape ``(*grid_shape, n)`` (*E8*);
- the reference implementation is written in pure NumPy, for linear laws with few phases: a phase map and one (precomputed) matrix per phase, with a Python loop over the *phases*, not over the cells. No Cython, no object per cell, and temporaries limited to the size of each phase;
- Numba is an *optional* dependency, not a foundation, for the cases where NumPy is not suitable: many phases (e.g. polycrystals with thousands of grains, where storing one matrix per cell costs :math:`n^2` reals per cell, i.e. six times a field in 3D), and hyperelastic laws at finite strain (milestone 0.6), whose stress and tangent operator are naturally evaluated cell by cell. A ``@njit`` loop over the cells, reading the matrix of each cell from a table indexed by the phase, creates no temporary;
- Cython is thus kept for the Green operator only, where it is required; for local operators, the goal of the task *The road to v1.0* (prototyping without compiling) is recovered;
- ``BlockDiagonalOperator2D/3D`` would disappear at milestone 0.4; the representation of hyperelastic laws is to be decided in milestone 0.6.

The :doc:`roadmap` was *not* modified: *E8* and milestones 0.4 and 0.5 are to be updated as part of the task *Make the roadmap consistent with the decisions of 2026-09-25* (above).

Follow-up: scope of the constitutive laws
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Claude had mentioned nonlinear constitutive laws several times during the discussion, including laws with internal variables. The author stated that such laws are not among the goals of Janus. Finite strain is of interest, but only for **hyperelasticity**: the stress and the tangent operator derive from a strain energy density, and depend on the current deformation gradient only. There are no internal variables, hence no history to store and no local integration of evolution laws (plasticity, damage, viscosity… are out of scope).

The problem remains nonlinear (material and geometric nonlinearities), and still calls for a Newton–Krylov loop, with hand-written tangent operators.

Milestone 0.6, which had been rewritten earlier in the day, was corrected accordingly: "nonlinear constitutive laws" became "hyperelastic constitutive laws (no internal variables)". The other mentions of nonlinear laws in the :doc:`roadmap` (*Goals*, *E8*, *E10*) are left to the task *Make the roadmap consistent with the decisions of 2026-09-25* (above).

Still inconsistent with the decisions: the *Goals*, friction point 2, the recommendation of *E2* (which now contradicts the milestones: it refers to the array namespace of *E7*, and does not mention that the link to FFTW disappears), *E5*, *E8*, *E10*, *E11*, *E12*, the two decisions *Default backend* and *Acceptable slowdown* of *Decisions required from the author*, and the introduction. The documentation builds without warning; ``docs/`` was not modified.


2026-09-26 • Status of the roadmap
==================================

The introduction of :doc:`roadmap` (``sphinx/roadmap.rst``, line 9) stated that "none of the evolutions described below has been implemented yet", which was no longer true: E1 is done, and so is the continuous integration part of E11. The author sees no point in rewriting the roadmap in the past tense, since the history is kept in the present page, in ``README.rst`` and in git. Claude suggested correcting only this sentence.

**Decision** (2026-09-26, as part of the task *Make the roadmap consistent with the decisions of 2026-09-25*). Only the last sentence of the introduction is replaced: it now states that the roadmap was revised on 2026-09-25 and 2026-09-26 after the decision to keep Cython, that the revised sections carry a note, and that milestone 0.2 (E1, and the continuous integration part of E11) is done, the other evolutions having not been implemented yet. ``docs/`` was not regenerated.


2026-09-23 • Continuous integration
===================================

Set up continuous integration on GitHub Actions, with the current Cython code, as announced in section *E11* and in milestone 0.2 of the :doc:`roadmap`. Its purpose is to build Janus from a clean clone, on Linux and Windows, at each push and once a week, so that the Cython implementation remains a reliable reference during the rewrite of milestone 0.3.

Proceed in the order below, one step at a time. Each step is a separate commit, made by the author: Claude does not commit, nor push. Report on each step in its own sub-paragraph of `Claude's report`, and wait for validation before starting the next one.

Out of scope: building and deploying the documentation to GitHub Pages, packaging and publication on PyPI (milestone 0.4), macOS.

2026-09-23 • Step 1 — Make ``setup.cfg`` optional
-------------------------------------------------

``setup.py``, ``environment.yml``, ``sphinx/installation.rst``, ``CLAUDE.md``.

When ``setup.cfg`` has no ``[fftw]`` section, ``setup.py`` falls back to defaults: ``libraries = fftw3`` on all platforms and, on Windows within a conda environment, ``include_dirs``/``library_dirs`` set to ``sys.prefix\Library\include``/``sys.prefix\Library\lib``. An existing ``[fftw]`` section keeps precedence over these defaults, so that the precompiled DLLs from fftw.org remain usable. The workflow of step 2 then reduces to ``pip install``, with no configuration file to generate.

Update the instructions accordingly: the header comment of ``environment.yml``, ``sphinx/installation.rst`` and the *Build* section of ``CLAUDE.md`` state that ``setup.cfg`` is only needed outside conda, or to override the defaults.

Check on Linux: move ``setup.cfg`` aside, remove the compiled artifacts (``git clean -Xfd janus/``), then ``pip install --no-build-isolation -e .`` and run the tests. Windows cannot be checked by Claude on this machine: the author checks it, or step 2 does.

2026-09-23 • Step 2 — GitHub Actions workflow
---------------------------------------------

``.github/workflows/tests.yml`` (new file).

- Triggers: ``push``, ``pull_request``, a weekly ``schedule`` (to catch breakage caused by new versions of the dependencies, which ``environment.yml`` does not pin), and ``workflow_dispatch``.
- Matrix: ``ubuntu-latest`` and ``windows-latest``.
- Steps: ``actions/checkout``; ``conda-incubator/setup-miniconda`` with ``environment.yml`` and the ``janus`` environment activated (``shell: bash -el {0}``); ``pip install --no-build-isolation -e .``; ``python -m pytest tests``; ``python -m sphinx -b doctest sphinx sphinx/_build/doctest``.

The workflow only runs once pushed to GitHub, which the author does. Two difficulties are expected, to be reported rather than worked around silently:

- the ``defaults`` channel of Anaconda may require the acceptance of its terms of service, which blocks non-interactive runs. If so, propose switching ``environment.yml`` to ``conda-forge`` (after checking that Python 3.14 and ``fftw`` are available there for both platforms), as a separate commit;
- on Windows, ``setuptools`` must find the MSVC compiler of the runner.

Acceptance criterion: both jobs are green, with the same test count as locally (3043 passed, 111 skipped on Linux), and no failing doctest.

2026-09-23 • Step 3 — Badge
---------------------------

``README.rst``.

Add the status badge of the workflow at the top of ``README.rst``, next to the documentation badge.

Claude's report
---------------

Step 1 — Make ``setup.cfg`` optional
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``setup.py``: new function ``fftw_config()``, which replaces the direct call to ``update_from_config({}, "fftw")`` in ``extensions_and_packages()``. If ``setup.cfg`` provides at least one entry in its ``[fftw]`` section, these entries are used as before, and the defaults are ignored altogether (they are not merged, so that a hand-written configuration behaves exactly as it did). Otherwise, ``libraries = fftw3`` and, on Windows within a conda environment (detected by the ``conda-meta`` directory of ``sys.prefix``), ``include_dirs``/``library_dirs`` are set to ``sys.prefix\Library\include``/``lib``. The message printed by ``python setup.py clean`` no longer says that ``setup.cfg`` is required.
- Why nothing more is needed on Linux: the Python interpreter of the ``janus`` environment compiles with ``-isystem $PREFIX/include`` and links with ``-L$PREFIX/lib`` and ``-rpath $PREFIX/lib`` (checked with ``sysconfig``), and the compiled ``_serial_fft`` module does load the ``libfftw3.so.3`` of the environment, not the one of the system (checked with ``ldd``). The same holds for macOS in principle, but was not checked.
- ``environment.yml`` (header comment), ``sphinx/installation.rst`` (sections *Configuration* and *Compilation with Miniconda and Visual Studio*, step 3) and ``CLAUDE.md`` (*Build*) now state that ``setup.cfg`` is not needed within the ``janus`` environment, and when it must still be written.

Check on Linux: with ``setup.cfg`` moved aside and the compiled artifacts removed (``git clean -Xfd janus/``), ``pip install --no-build-isolation -e .`` succeeds, and the tests give 3043 passed, 111 skipped; the doctests (``python -m sphinx -b doctest sphinx sphinx/_build/doctest``) give 0 failures. ``setup.cfg`` was then put back. The Windows branch of ``fftw_config()`` was not run: it will be checked by the author, or by the Windows job of step 2.

Step 2 — GitHub Actions workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

New file ``.github/workflows/tests.yml``, as specified. A few choices beyond the specification:

- ``fail-fast: false``: a failure on one platform does not cancel the job on the other, so that each run tells whether the problem is specific to one platform;
- ``concurrency`` with ``cancel-in-progress``: a new push to a branch cancels the run still in progress on that branch;
- ``permissions: contents: read``: the workflow only needs to read the repository;
- a *Show environment* step (``conda list``), so that the versions of the dependencies are recorded in the log of each run. This is what makes a failure of the weekly run diagnosable: comparing with the log of the last successful run shows which package changed;
- no ``setup.cfg`` is written: the build relies on the defaults of step 1.

The weekly run (Mondays, 04:00 UTC) only starts once the workflow is on ``master``: GitHub runs scheduled workflows on the default branch only.

What was checked: the file is valid YAML, and the commands of the workflow were run by hand on a fresh clone of the branch (``git clone``, no ``setup.cfg``, no compiled artifacts), in the ``janus`` environment: build OK, 3043 passed, 111 skipped, 0 failing doctest. The workflow itself could not be run: it will run when the author pushes the branch. The two expected difficulties (terms of service of the ``defaults`` channel, MSVC on Windows) can only be observed there.

Step 2 (continued) — Switch to ``conda-forge``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the author's request, the first of the two difficulties was removed before the first run, rather than waited for.

- ``environment.yml``: the only channel is now ``conda-forge``. ``fftw`` is pinned to its ``nompi_*`` builds: conda-forge also publishes builds of FFTW linked against MPI (OpenMPI, MPICH), which Janus no longer needs.
- ``.github/workflows/tests.yml``: ``setup-miniconda`` installs Miniforge (``miniforge-version: latest``), whose only channel is ``conda-forge``, and ``conda-remove-defaults: true`` makes sure that the ``defaults`` channel is never queried. There are no terms of service to accept.
- ``sphinx/installation.rst``: a note states that the packages come from ``conda-forge`` (which works with Miniconda as well as Miniforge), and that an environment created from the former ``environment.yml`` should be recreated rather than updated, so as not to mix channels.

What was checked:

- both platforms resolve on ``conda-forge`` (``conda create --dry-run`` with ``CONDA_SUBDIR`` set to ``linux-64`` and ``win-64``): Python 3.14.7 (the regular build, not the free-threaded one), Cython 3.3.0, NumPy 2.5.3, FFTW 3.3.11 ``nompi``;
- the Windows package of FFTW on ``conda-forge`` provides ``Library/include/fftw3.h`` and ``Library/lib/fftw3.lib`` (checked by downloading the package): the defaults of ``fftw_config()`` (step 1) apply unchanged;
- on Linux, end to end: a fresh environment created from the new ``environment.yml`` (in a temporary prefix), a fresh clone without ``setup.cfg``, ``python setup.py build_ext --inplace``, then 3043 passed, 111 skipped, and 0 failing doctest. The compiled module loads the FFTW of the new environment (``ldd``). The temporary environment and clone were deleted.

The author's own ``janus`` environment was not modified: it still comes from the ``defaults`` channel, and should be recreated as explained above.

After the push by the author, both jobs of the workflow are green on GitHub. This also validates the two points that could not be checked on this machine: the Windows branch of ``fftw_config()`` (step 1), and the detection of the MSVC compiler of the runner by ``setuptools``.

Step 3 — Badge
~~~~~~~~~~~~~~

``README.rst``: the status badge of the workflow is placed before the documentation badge, on the same line. Both are now written as substitutions (``|tests| |docs|``), since two consecutive ``image`` directives would be rendered one below the other. The rendering was checked with ``docutils`` (no warning, both images in a single paragraph).

The badge reports the status of ``master`` (``?branch=master``), which is the reference for a visitor, not the status of the working branches. It will show *no status* until ``continuous_integration`` is merged into ``master``; the weekly run will also start then.


2026-09-23 • Removal of the vestiges of distributed memory
==========================================================

Now that the parallel code has been removed (see the branch ``MPI-ectomy``), the classes of ``janus/fft/serial/`` and ``janus/green.pyx`` still carry the attributes that used to locate a local slab within a domain-decomposed grid. They are now redundant: ``offset0`` is always 0, ``idispl`` and ``odispl`` are always 0, ``global_shape0 == shape0``, and ``global_ishape``/``global_oshape`` are equal to ``ishape``/``oshape``. Remove them, as announced in section *E1* of the :doc:`roadmap`.

Proceed one attribute at a time, in the order below, so that each step is a separate commit. Each step leaves the library in a working state: rebuild the extension modules (``python setup.py build_ext --inplace``) and run the tests before moving on. The order is not arbitrary — steps 2 and 4 remove the *readers* of the attributes that steps 3 and 5 delete.

Report on each step in its own sub-paragraph of `Claude's report`, and wait for validation before starting the next one.

2026-09-23 • Step 1 — ``idispl`` and ``odispl``
-----------------------------------------------

``janus/fft/serial/_serial_fft.pxd``, ``_serial_fft.pyx``.

Dead code: both are computed in ``__cinit__`` from ``offset0``, and read nowhere else in the code base. Pure deletion, no other file affected.

2026-09-23 • Step 2 — ``offset0`` of the discrete Green operators
-----------------------------------------------------------------

``janus/green.pyx``.

Remove the ``offset0`` attribute of ``DiscreteGreenOperator2D`` and ``3D``, which is read from ``transform.offset0`` (or set to 0 when there is no transform). Three consequences:

- in ``set_frequency``, the bounds check on ``b0`` becomes ``0 <= b0 < shape0``, like the checks on ``b1`` and ``b2``; the error message becomes the same as theirs, which leaves the text produced unchanged, since ``offset0`` was interpolated as the lower bound and was 0;
- in the ``c_apply`` of ``DiscreteGreenOperator2D``/``3D``, ``b[0] = i0 + self.offset0`` becomes ``b[0] = i0``;
- in the ``c_apply`` of ``TruncatedGreenOperator2D``/``3D``, ``b0 = i0 + self.offset0`` becomes ``b0 = i0``.

This step must come before step 3, which removes the attribute that is read here.

2026-09-23 • Step 3 — ``offset0`` of the FFT objects
----------------------------------------------------

``janus/fft/serial/_serial_fft.pxd``, ``_serial_fft.pyx``, ``examples/square_inclusion.py``.

Once step 2 is done, the attribute is read nowhere in the library. Remove it, together with the ``offset0`` parameter of the constructors of ``_RealFFT2D`` and ``_RealFFT3D``. These constructors are called only by ``create_real_2D``/``create_real_3D``, which are ``cdef`` functions: the public entry point ``janus.fft.serial.create_real(shape, flags)`` is unaffected.

``examples/square_inclusion.py`` reads ``transform.offset0``. The example is already broken (it depends on ``mpi4py`` and ``petsc4py``) and carries a ``TODO`` header describing its repair: state there that the attribute no longer exists, but do not attempt the repair, which is a separate task.

2026-09-23 • Step 4 — ``global_shape0``
---------------------------------------

``janus/green.pyx``, ``CLAUDE.md``.

``global_shape0`` is equal to ``shape0``, which ``init_shapes`` sets just above its last use. Replace it with ``self.shape0`` in the computation of ``s0`` (``DiscreteGreenOperator2D``/``3D``) and in the wave-vectors of ``TruncatedGreenOperator2D``/``3D`` and ``FilteredGreenOperator2D``/``3D``, then remove the attribute. ``FiniteDifferences2D``/``3D`` does not use it.

The constructors also read ``transform.global_ishape``, to check the shape of the transform: switch that check to ``transform.ishape``. Step 5 removes the attribute, so this step must come first.

In ``CLAUDE.md``, delete the paragraph that presents ``offset0``/``global_shape0`` as a pending simplification.

2026-09-23 • Step 5 — ``global_ishape``, ``global_oshape`` and ``n0_loc``
-------------------------------------------------------------------------

``janus/fft/serial/_serial_fft.pxd``, ``_serial_fft.pyx``, ``sphinx/fft_tutorial.rst``, ``CLAUDE.md``.

Once step 4 is done, these are read only by the documentation. Remove the two attributes and the ``n0_loc`` parameter of the constructors (which then take the shape alone), and drop the word *global* from the docstring of ``create_real``.

In ``sphinx/fft_tutorial.rst``, the four attributes documented for a serial transform are reduced to ``ishape`` and ``oshape``: remove the two doctests on ``global_ishape``/``global_oshape``, and the two sentences explaining that local and global shapes coincide in the serial case. In ``CLAUDE.md``, the description of the FFT layer no longer mentions the vestigial attributes.

Whatever the step, the behaviour of the library must not change: the tests (3043 passed, 111 skipped) are the acceptance criterion, and no reference data in ``tests/data`` is to be regenerated.

Claude's report
---------------

Step 1 — ``idispl`` and ``odispl``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both attributes were removed, as a pure deletion. Two files were modified.

- ``janus/fft/serial/_serial_fft.pxd``: in the declarations of ``_RealFFT2D`` and ``_RealFFT3D``, the ``readonly ptrdiff_t`` line becomes ``isize, osize, offset0``.
- ``janus/fft/serial/_serial_fft.pyx``: the two assignments of ``__cinit__`` were deleted in each class (``self.idispl``/``self.odispl``, computed from ``self.offset0``).

Before deleting, a search for ``idispl`` and ``odispl`` over the whole repository (excluding ``docs/`` and the build artifacts) confirmed that the only remaining occurrences were these four lines and the description of the task in the present file: the attributes were written in ``__cinit__`` and read nowhere — neither in ``janus/``, nor in ``tests/``, ``examples/`` or ``sphinx/``. No other file was affected.

The extension modules were rebuilt (``python setup.py build_ext --inplace``; all five ``.pyx`` files were re-cythonized, not only the modified one — the ``.c`` files present in the checkout were stale) and the tests were run: **3043 passed, 111 skipped**, i.e. the expected baseline. No reference data was touched.

The changes are not committed, as usual.

Step 2 — ``offset0`` of the discrete Green operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The attribute was removed from ``DiscreteGreenOperator2D`` and ``DiscreteGreenOperator3D``; only ``janus/green.pyx`` was modified. The three announced consequences were applied, in both classes:

- the declaration ``cdef int offset0`` and its two assignments in ``__cinit__`` (``transform.offset0`` when a transform is given, ``0`` otherwise) are gone. The attribute was declared ``cdef``, not ``cdef readonly``: it was never visible from Python, so this is not a breaking change for the public API;
- in ``set_frequency``, the bounds check on ``b0`` is now ``if (b0 < 0) or (b0 >= self.shape0)``, and raises the same message as the checks on ``b1`` and ``b2``;
- in ``c_apply``, ``b[0] = i0 + self.offset0`` becomes ``b[0] = i0`` (``DiscreteGreenOperator2D``/``3D``), and ``b0 = i0 + self.offset0`` becomes ``b0 = i0`` (``TruncatedGreenOperator2D``/``3D``).

``global_shape0`` was left untouched, as it belongs to step 4; so was the check on ``transform.global_ishape``.

After rebuilding, the tests give the expected baseline: **3043 passed, 111 skipped**, with the reference data unchanged.

The out-of-bounds branch of ``set_frequency`` turned out *not* to be covered by the test suite (``tests/test_greenop.py`` only exercises the continuum operators), so the invariance of the error message was checked by hand, on a ``TruncatedGreenOperator2D`` of shape ``(8, 8)``, built both with and without a transform. In both cases, ``set_frequency`` raises ``index must be >= 0 and < 8 (was -1)`` (resp. ``(was 8)``) for an out-of-range ``b0``, which is the text produced before the change, and the message is now identical to the one raised for an out-of-range ``b1``. A valid index is still accepted.

The changes are not committed.

Step 3 — ``offset0`` of the FFT objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three files were modified.

- ``janus/fft/serial/_serial_fft.pxd``: in ``_RealFFT2D`` and ``_RealFFT3D``, the ``readonly ptrdiff_t`` line is reduced to ``isize, osize``. Unlike the ``offset0`` of the Green operators (step 2), this one *was* ``readonly``, hence visible from Python: its removal is a breaking change for any code that read ``transform.offset0``. This is the intended outcome, announced in section *E1* of the :doc:`roadmap`.
- ``janus/fft/serial/_serial_fft.pyx``: the ``offset0`` parameter was dropped from the ``__cinit__`` of both classes, together with the assignment ``self.offset0 = offset0``. The two call sites were updated accordingly: ``_RealFFT2D(n0, n1, n0, 0)`` becomes ``_RealFFT2D(n0, n1, n0)``, and ``_RealFFT3D(n0, n1, n2, n0, 0)`` becomes ``_RealFFT3D(n0, n1, n2, n0)``. A search confirmed that these two lines, in the ``cdef`` functions ``create_real_2D``/``create_real_3D``, are the only places where the constructors are called; the public entry point ``janus.fft.serial.create_real(shape, flags)`` is unchanged, and so is the ``n0_loc`` parameter, which belongs to step 5.
- ``examples/square_inclusion.py``: the ``TODO`` header now states that ``transform.offset0`` no longer exists at all, and that the two references to it in the body of the example (``self.offset0 = transform.offset0`` and the slice ``ops[:imax - self.offset0, :imax]``) are therefore dead. The repair itself was not attempted, as agreed: the example still imports ``mpi4py``, ``petsc4py`` and ``janus.fft.parallel``, and does not run.

After ``offset0`` was removed, no occurrence of it remains anywhere under ``janus/``.

The extension modules were rebuilt and the tests run: **3043 passed, 111 skipped**, the expected baseline, with the reference data unchanged. ``tests/test_fft.py`` builds its transforms through the public ``create_real`` and exercises ``r2c``/``c2r`` round trips, so the constructors are covered; the attribute itself was read by no test.

Step 4 — ``global_shape0``
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``global_shape0`` disappears from ``DiscreteGreenOperator2D`` and ``3D``. In ``janus/green.pyx``:

- the declaration ``cdef int global_shape0`` and its two assignments in ``__cinit__`` (``transform.global_ishape[0]``, or ``shape[0]`` without a transform) were deleted;
- the twelve readers now use ``self.shape0``: the computation of ``s0``, the wave-vector of ``TruncatedGreenOperator2D``/``3D`` (``if 2 * b0 > self.shape0`` and the corresponding ``self.k[0]``, in both ``c_set_frequency`` and ``c_apply``), and the first component of the wave-vectors of ``FilteredGreenOperator2D``/``3D``. ``FiniteDifferences2D``/``3D`` was indeed not concerned;
- the check on the shape of the transform now reads ``transform.ishape``, in the condition and in the error message, which frees step 5;
- the comment above ``s0`` in ``DiscreteGreenOperator2D`` described ``n[i]`` as the size of the *global* grid; it now reads like its 3D counterpart, which never mentioned the global grid.

The substitution is legitimate because ``init_shapes``, called just above the computation of ``s0``, sets ``shape0`` to exactly the value that ``global_shape0`` received: ``transform.ishape0`` (equal to ``transform.global_ishape[0]`` in the serial case) or ``shape[0]``.

Both replacements are covered by the test suite. The mismatched-transform branch is exercised by ``tests/test_discrete_green_operator.py``, whose parameters for ``test_init_invalid_params`` include a grid of shape ``(9, 9)`` with a transform of shape ``(8, 9)`` and ``(9, 8)``; and the wave-vectors are precisely what the reference data of ``tests/data`` check. Tests: **3043 passed, 111 skipped**, reference data unchanged.

In ``CLAUDE.md``, the paragraph presenting the ``offset0``/``global_shape0`` pair as a pending simplification was deleted.

One residue is left on purpose: in the *FFT layer* paragraph of ``CLAUDE.md``, the serial transforms are still described as exposing ``ishape``/``oshape`` "alongside the vestigial ``global_ishape``/``global_oshape`` and ``offset0``". Since step 3, ``offset0`` no longer exists; that sentence is rewritten by step 5, which removes the last two attributes as well.

Step 5 — ``global_ishape``, ``global_oshape`` and ``n0_loc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last vestiges are gone. Four files were modified.

- ``janus/fft/serial/_serial_fft.pxd``: in both classes, the ``readonly tuple`` line is reduced to ``ishape, oshape``.
- ``janus/fft/serial/_serial_fft.pyx``: the four assignments of ``global_ishape``/``global_oshape`` were deleted, and the constructors now take the shape alone — ``__cinit__(self, n0, n1)`` and ``__cinit__(self, n0, n1, n2)``. Wherever ``n0_loc`` was used (``ishape0`` and ``oshape0``), ``n0`` is used instead; the call sites in ``create_real_2D``/``create_real_3D`` lose their repeated argument (``_RealFFT2D(n0, n1, n0)`` becomes ``_RealFFT2D(n0, n1)``). In the docstring of ``create_real``, ``shape`` is now "the shape of the input data".
- ``sphinx/fft_tutorial.rst``: the list of attributes is reduced to ``ishape`` and ``oshape``, described without the *local*/*global* emphasis, which no longer opposes anything; the two doctests on ``global_ishape``/``global_oshape`` and the two sentences on the coincidence of local and global shapes were removed.
- ``CLAUDE.md``: the *FFT layer* paragraph now says that the transforms expose ``ishape``/``oshape`` (and ``isize``/``osize``), with no mention of the vestigial attributes. This also settles the residue reported in step 4.

After rebuilding, the tests give **3043 passed, 111 skipped**, with the reference data unchanged. The doctests of the documentation were run as well (``python -m sphinx -b doctest``, into a temporary directory): ``fft_tutorial`` passes its 30 tests, which covers the rewritten section.

Two mentions were deliberately left alone.

- ``README.rst`` documents the renaming of the FFT attributes in its *History of major changes*, under the date 2015-02-23 (``shape`` → ``global_ishape``, and the addition of ``global_oshape``). This is a historical record of what happened in 2015, not a description of the current API: rewriting it would falsify the history. The removal is already recorded in the same file, in the entry of 2026-09-15 on the MPI-ectomy.
- ``sphinx/roadmap.rst`` lists these attributes twice, as *friction point 6* and in the description of milestone 0.2, i.e. as the work to be done. They now *are* done; the roadmap will have to be updated when the milestone is closed, which is beyond the scope of this task.

Unrelated observation, found while running the doctests: ``sphinx/operators.rst`` (line 81) has one failing doctest, ``np.sqrt(np.sum((yy - y)**2))``, which expects ``0.0`` but gets ``np.float64(0.0)``. This is the NumPy 2 scalar representation, has nothing to do with the present task, and does not affect the HTML build (the ``doctest`` builder is not run by ``make ghpages``).

Suggestions
-----------

On the interactions:

- the form of this task is worth reusing for the rest of the roadmap. Splitting it into numbered steps, each with the list of files it touches, the consequences to expect, the reason for its rank in the order, and a global acceptance criterion (the test count), left no room for interpretation: each step was a mechanical edit followed by a check. The validation between steps also kept the reviewing effort proportionate to what had just changed;
- one thing the steps could state explicitly: whether Claude should commit. The introduction says "so that each step is a separate commit", while the standing rule of *How to update this page* is never to commit. The commits were made by the author, which is consistent, but a word in the task would remove the ambiguity.

Follow-ups for the code base, in decreasing order of usefulness:

- ``sphinx/roadmap.rst`` still lists ``offset0``, ``global_shape0``, ``global_ishape``, ``global_oshape`` and ``n0_loc`` as work to be done, in *friction point 6* and in milestone 0.2. Both passages could now be written in the past tense, when the milestone is closed;
- ``examples/square_inclusion.py`` is the only example of the symmetric formulation, and it no longer runs. Its ``TODO`` header now describes its repair completely, including the two dead references to ``offset0``: the rewrite in serial form would make a well-defined task;
- the failing doctest of ``sphinx/operators.rst`` (see above) could be fixed, either by expecting ``np.float64(0.0)``, or by wrapping the result in ``float(...)``, which keeps the output stable across NumPy versions;
- the out-of-bounds branch of ``set_frequency`` in the discrete Green operators is not covered by any test (this was found in step 2, and checked by hand). A short test on the three indices would protect the bounds checks, which are the only remaining place where the shape of the grid is interpreted.


2026-09-15 • Planning the MPI-ectomy
====================================

I am ready to start working on the removal of the MPI dependency.

I will work in a dedicated branch: show me how to proceed.

Then, draft a plan for this task.

Claude's report
---------------

Should a branch be created?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, for three reasons.

- The task spans several commits and introduces breaking changes to the public API (``offset0``, ``global_ishape``… disappear). ``master`` should remain usable in the meantime.
- ``master`` is the published branch: GitHub Pages serves ``master:/docs``. On a branch, ``sphinx/`` can be modified freely.
- The repository has no tags at all. Before deleting anything, the last commit that includes MPI should be tagged, so that it remains available as a reference (this is also part of milestone 0.2 of :doc:`roadmap`).

Procedure
~~~~~~~~~

At the time of writing, ``master`` has no local changes and is up to date with ``origin/master``. From the root of the project:

1. Tag the last commit with MPI (annotated tag), and push the tag::

     git tag -a last-mpi -m "Last version with the MPI-enabled parallel FFT"
     git push origin last-mpi

   The name ``last-mpi`` is only a suggestion; a version number (e.g. ``v0.1``) would also do.

2. Create the branch, switch to it, and (optionally) push it to GitHub, as a backup and to allow a pull request later::

     git switch -c remove-mpi
     git push -u origin remove-mpi

3. Commit on the branch as the work progresses.

4. Once the work is done, merge the branch into ``master``, either through a pull request on GitHub, or locally::

     git switch master
     git merge --no-ff remove-mpi

   Regenerate ``docs/`` only after the merge.

Draft plan
~~~~~~~~~~

The work proceeds from the outside in, so that the serial tests pass after each step. Before starting, record a baseline with ``python -m pytest tests`` (last recorded result: 3043 passed, 111 skipped). Each step could be one commit.

1. **Delete the MPI code** (no change for the serial code).

   - Delete ``janus/fft/parallel/``, ``tests/parallel/``, ``sphinx/parallel_fft_tutorial.py`` and ``examples/fftw_python_benchmark_mpi.py``.
   - ``examples/square_inclusion.py`` uses MPI and PETSc: rewrite it for serial use, or delete it.
   - ``setup.py``: remove ``mpicc_show()`` and ``extensions_and_packages_with_mpi()`` (lines 75–128), and the ``[fftw_mpi]`` configuration.
   - ``pytest.ini``: remove ``parallel`` from ``norecursedirs``.
   - ``sphinx/api.rst``: remove the ``janus.fft.parallel`` section (this also removes the two warnings of the documentation build).

2. **Simplify the serial FFT** (breaking change). In ``janus/fft/serial/_serial_fft.pyx`` and ``_serial_fft.pxd``, remove ``n0_loc``, ``offset0``, ``idispl``, ``odispl``, ``global_ishape`` and ``global_oshape`` from ``_RealFFT2D`` and ``_RealFFT3D``.

3. **Simplify** ``janus/green.pyx``. Remove ``global_shape0`` and ``offset0`` from the discrete Green operators, and use ``shape0`` instead (around lines 92–155, 199, 253–321, 371, 436–550, 616 and 738), together with the checks of ``transform.global_ishape``. ``tests/test_operators.py`` also contains a few matching names, to be checked (some may be unrelated uses of the word "parallel"). Recompile with ``python setup.py build_ext --inplace``. The reference data (``tests/data/*.npz``) must *not* be modified: they are precisely what detects a mistake in this step.

4. **Update the documentation**: ``README.rst`` and ``sphinx/introduction.rst`` ("fully parallelized"), ``sphinx/installation.rst`` and ``sphinx/fft_tutorial.rst`` (both mention MPI extensively), ``CLAUDE.md``, and the comment on ``mpi4py`` in ``environment.yml``.

What was checked
~~~~~~~~~~~~~~~~

The plan is based on a search for ``mpi``, ``parallel``, ``offset0``, ``global_`` and ``n0_loc`` in the sources (304 occurrences in 26 files, excluding ``docs/`` and the build artifacts), and on the current state of the repository (``git branch -a``, ``git tag``, ``git status``). No command of the procedure was run: no tag and no branch were created, and no code was modified.

Follow-up: tag and branch created by the author
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The author then created the tag ``Farewell_MPI`` (on commit ``161dc75``) and the branch ``MPI-ectomy``, both pushed to GitHub, and added a corresponding entry to the *History of major changes* of ``README.rst``. At the author's request, Claude suggested language improvements to this entry (in the console only; ``README.rst`` was not modified by Claude).

The author validated this report on 2026-09-15, and the task was marked as done.


2026-09-11 • The road to v1.0
=============================

The goal of this task is to write a roadmap to v1.0 of `Janus`

Context
-------

`Janus` was initially built to target both prototyping and large simulations.

Large simulations has now become obsolete, because new developments in the community will soon offer excellent alternatives. So `Janus` should really evolve into a more flexible library, easily maintanable and extensible.

The architecture should reduce as much friction as possible at installation and allow for rapid prototyping of new ideas. More precisely

- implementation of new discretizations
- implementation of new physics (e.g. Darcy, finite strain solid mechanics, etc.)

should be possible without having to compile with Cython, which is not possible at the time being I think, even for experimental (sub-optimal) code.

I have listed below a few evolutions I have in mind. Analyse each of them in terms of gains, losses, consequences on the code, implementations problems, …

With the above context in mind, try to come up with supplementary evolutions that you think might improve the code's flexibility, and go through the same analysis for each of your suggestions. At this stage, breaking changes in the code would be allowed.

Report briefly in this file, and write the detailed analysis in a new chapter of these docs (file called `roadmap.rst`); add the corresponding entry in the `index.rst` file.

Possible evolutions
-------------------

Removal of the MPI dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`MPI` is no longer useful if we do not target large simulations.

The gain would be considerable

- installation
- the structure of some objects: no need to encode the global shape and the local shape of an operator

Interfacing with FFTW
~~~~~~~~~~~~~~~~~~~~~

At the time of creation of `Janus`, interfacing with `FFTW` was done “manually” through a hand-crafted `Cython` file. Nowedays, `pyFFTW` seems to be the *de facto* module (with presumably a robust installation procedure).

Replacing `Cython` with other (dynamic) tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compilation required by `Cython` adds some friction. I am under the impression that tools such as `Numba`, `JaX` are mature tools that could (partly) replace `Cython`. I am ready to pay a little price in terms of speed if it is justified by a significant gain in flexibility.

Claude's report
---------------

Before starting, Claude asked four questions; the author answered that GPU execution and automatic differentiation are *nice to have*, that any dependency distributed as binary wheels is acceptable, that the analysis should not be backed by benchmarks, and that the roadmap should include ordered milestones.

The detailed analysis is in the new chapter :doc:`roadmap` (file ``sphinx/roadmap.rst``, added to ``sphinx/index.rst`` after the API reference). It is based on a complete reading of the code base, and on the current state of the candidate dependencies on PyPI. In brief:

- **Friction points.** Seven were identified. The most important one: all the methods that do the actual work are ``cdef`` methods, so that Python subclasses are *silently ignored* inside the library. For instance, a Python local operator used in a ``BlockDiagonalOperator2D`` leaves the output untouched; this was checked with the current build. The other ones: operators are scalar and stateful (one call per frequency, impossible to transpose to Python); everything is duplicated for 2D and 3D; the filtered discretization is hard-coded for elasticity; the data model is restricted (Mandel–Voigt vectors, ``float64``, real symbols); local and global shapes leak from MPI into the serial code; the build chain is fragile.
- **Evolutions proposed by the author.** Remove MPI: yes, first (cheap, and it reduces the code to be ported). FFTW: rather than making pyFFTW a hard dependency, use ``scipy.fft`` by default, with pyFFTW as an optional accelerator through the ``scipy.fft`` backend mechanism. Cython: replace it entirely, first by a pure NumPy/SciPy implementation written in functional style, then by an optional JAX backend (fusion, GPU, automatic differentiation); Numba is not recommended as a foundation, because it does not solve the extensibility problem.
- **Additional evolutions (E4 to E12).** Vectorized, stateless operators (the Green operator as a function of an array of wave-vectors); dimension-generic code; separation of discretizations (weighted modified wave-vectors) and physics; array API and functional interface; local operators as fields (phase maps, constitutive functions, autodiff tangents); generic tensor representations (vectors, full tensors for finite strain); thin solver adapters; pure Python packaging and continuous integration; testing strategy based on the reference data and on mathematical properties.
- **Milestones.** 0.2 clean-up (MPI removal, continuous integration); 0.3 new core alongside the Cython code, validated against the reference data and benchmarked; 0.4 switch (Cython and FFTW removed, pure Python packaging); 0.5 extensibility (second physics in pure Python); 0.6 backends (JAX, autodiff, finite strain prototype); 1.0 stabilization.
- **Decisions required from the author** are listed at the end of the chapter. In particular, the name ``janus`` is already taken on PyPI (by an unrelated package), so a distribution name must be chosen before publication.

The speed estimates of the chapter are not backed by benchmarks, as agreed; a benchmark is part of milestone 0.3. No code was modified. The documentation builds without new warnings.

2026-09-11 • Updating the online docs
=====================================

Report on the procedure for compiling the sphinx docs locally and updating the Github pages accordingly (the project lives in the following repo: https://github.com/sbrisard/janus while the github pages are: https://sbrisard.github.io/janus/).

Claude's report
---------------

Current setup
~~~~~~~~~~~~~

- The HTML docs are versioned in the ``docs/`` directory of the ``master`` branch (since 2018), with a ``docs/.nojekyll`` file, so that GitHub does not process the files with Jekyll (which would ignore the ``_static``, ``_sources``… directories). There is no ``gh-pages`` branch, and no GitHub Actions workflow. GitHub Pages is therefore most likely configured as *Deploy from a branch*, ``master``, ``/docs``. This could not be checked directly (the corresponding GitHub API requires authentication); it can be checked in *Settings → Pages* of the repository.
- The online docs are exactly the content of ``docs/``: they were last updated on 2019-08-20 (commit ``811359f``), with Sphinx 2.1.2. In particular, they do not contain the present page, nor the new installation procedure.
- ``sphinx/make.bat html`` builds the docs *directly* into ``../docs`` (the doctrees go to ``sphinx/_build/doctrees``; both ``sphinx/_build`` and ``docs/.buildinfo`` are git-ignored). By contrast, ``sphinx/Makefile`` (Linux, macOS) builds into ``sphinx/_build/html``, and the files must then be copied to ``docs/``.

Procedure
~~~~~~~~~

1. Activate the ``janus`` environment, where `Janus` must be compiled and installed (see *Installation on a windows machine*): ``autodoc`` imports the compiled modules.

2. Empty the ``docs/`` directory (from the root of the project, in PowerShell). ``make.bat html`` does not remove files that Sphinx no longer produces; without this step, obsolete files would accumulate in ``docs/``. Since the ``sphinx.ext.githubpages`` extension was added (see follow-up below), ``docs/.nojekyll`` is recreated by the build, and the whole directory can be emptied::

     Get-ChildItem -Force docs | Remove-Item -Recurse -Force

3. Build the docs, from the ``sphinx/`` directory. ``make.bat clean`` deletes the cached doctrees, which forces a complete rebuild::

     cd sphinx
     .\make.bat clean
     .\make.bat html

   Equivalently, from the root of the project: ``python -m sphinx -b html -E -d sphinx/_build/doctrees sphinx docs``.

4. Check the result: open ``docs/index.html`` in a browser, and review the changes with ``git status docs``.

5. Commit the ``docs/`` directory, and push to ``master``::

     git add -A docs
     git commit -m "Update HTML docs"
     git push

   GitHub Pages then redeploys the site automatically (usually within a few minutes). The deployment can be followed in the *Actions* tab of the repository (*pages build and deployment*).

What was checked
~~~~~~~~~~~~~~~~

Steps 2 and 3 were not run as such on ``docs/``, and step 5 was not run at all (no commit, no push). The following equivalent operations were performed instead.

- A complete build (``python -m sphinx -b html -E``) in an empty temporary directory succeeds (2 warnings, both related to ``janus.fft.parallel``). Compared to the current ``docs/``, it adds 8 files (``claude.html``, ``_sources/claude.rst.txt``, 2 files in ``_downloads/``, 4 JavaScript/SVG files in ``_static/``) and makes 4 files obsolete (``jquery*.js`` and ``underscore*.js`` in ``_static/``), which step 2 removes.
- Mirroring this build into a copy of ``docs/`` with ``robocopy <build> docs /MIR /XF .buildinfo .nojekyll`` (an alternative to steps 2 and 3) gives the same result, and keeps ``.nojekyll``. Note that ``robocopy`` returns non-zero exit codes on success (codes 1 to 7), which may confuse scripts.
- ``make.bat html`` was also run as is, and wrote into ``docs/``: this is how Claude found out that it targets ``../docs``. Since this task only asked for a report, the ``docs/`` directory was then restored to its committed state (``git restore docs``, removal of the new files, of ``docs/.buildinfo`` and of ``sphinx/_build``). ``docs/`` is currently unchanged.

Points to be aware of
~~~~~~~~~~~~~~~~~~~~~

- The API of ``janus.fft.parallel`` is absent from the docs built under Windows (the parallel extension is not compiled). This was already the case for the current online docs.
- The present page (``claude.rst``) will be published. It contained an absolute local path, in the report of *Installation on a windows machine* (replaced since, see follow-up below).
- The generated HTML files are versioned: each update of the docs adds all modified HTML files to the history of the repository.

Follow-up: suggestions 2 and 4 applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the author's request, suggestions 2 and 4 below were applied.

- ``sphinx.ext.githubpages`` was added to the ``extensions`` of ``sphinx/conf.py``. A complete build in a temporary directory succeeds (still 2 warnings, both related to ``janus.fft.parallel``), and now produces a ``.nojekyll`` file in the output directory. Step 2 of the above procedure was updated accordingly: the whole ``docs/`` directory can now be emptied before the build. The extension would also create a ``CNAME`` file if ``html_baseurl`` pointed to a custom domain, which is not the case here.
- In the report of *Installation on a windows machine*, the absolute local paths of the ``setup.cfg`` example were replaced by ``C:\path\to\miniconda3\envs\janus\Library\…``. No other absolute local path remains in ``sphinx/``.

Follow-up: inconsistency between ``Makefile`` and ``make.bat``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both files were generated by ``sphinx-quickstart`` on 2013-09-17 (commit ``e2b3e81``), for Linux/macOS (``Makefile``) and Windows (``make.bat``) respectively. Originally, the ``html`` target of both files built the docs into ``_build/html`` (i.e. ``sphinx/_build/html`` today). When the HTML docs were moved to ``docs/`` to be served by GitHub Pages (commit ``f0f32ee``, 2018-07-13, which also renamed the source directory ``docs/`` → ``sphinx/``), the ``html`` target of ``make.bat`` was changed to build into ``../docs``, but ``Makefile`` was not modified. As a consequence:

- under Windows, ``make.bat html`` updates ``docs/`` directly (and the changes show up in ``git status``);
- under Linux/macOS, ``make html`` builds into ``sphinx/_build/html``, which is git-ignored: ``docs/`` is left untouched, and nothing shows up in ``git status``. The files must then be copied to ``docs/`` by hand (without ``.buildinfo``), which is easily forgotten.

In both files, all other targets (``dirhtml``, ``latex``, ``epub``…) still build into ``_build/``, and ``clean`` only empties ``_build/`` (the doctrees), never ``docs/``. Both files also contain leftovers from another project: the ``qthelp`` and ``devhelp`` targets refer to ``HashTri`` (e.g. ``HashTri.qhcp``) instead of ``Janus``. Finally, both use the old ``sphinx-quickstart`` template, which lists every builder explicitly; current versions of ``sphinx-quickstart`` generate much shorter files, based on ``sphinx-build -M`` ("make mode").

Possible fixes, from the smallest to the largest change:

a. Change the ``html`` target of ``Makefile`` to build into ``../docs``, as ``make.bat`` does (two lines). The ``HashTri`` leftovers can be fixed at the same time.
b. Replace both files with those generated by the current ``sphinx-quickstart`` (make mode), and add a dedicated target (e.g. ``ghpages``) that empties ``docs/`` and builds into it. This must be done twice (``make`` and batch syntax).
c. Remove both files, and document a single command, identical on all platforms, run from the root of the project::

     python -m sphinx -b html -E -d sphinx/_build/doctrees sphinx docs

   Emptying ``docs/`` beforehand is still platform-dependent (``rm -rf docs/*`` vs. PowerShell); a small Python script (e.g. ``scripts/build_docs.py``) could perform both steps on all platforms.

Claude would recommend c: the docs are only built as HTML, for GitHub Pages, so that the many other targets of the ``make`` files are not needed, and a single command (or script) avoids maintaining two files in two syntaxes. None of these fixes was applied.

Follow-up: options b and c applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The author chose to combine b and c: the ``make`` files are updated according to b (the author wants to build the PDF version of the docs from time to time), but the procedure that is *documented* is c. This follow-up supersedes steps 2 and 3 of the above procedure.

Changes:

- ``scripts/empty_docs.py`` (new): empties ``docs/``, except ``docs/.nojekyll``. The ``docs/`` directory is located relative to the script (at the root of the project), not relative to the current directory, so that the script cannot empty another directory by mistake.
- ``sphinx/Makefile`` and ``sphinx/make.bat`` were replaced by the files generated by the current ``sphinx-quickstart`` (Sphinx 9.1, "make mode": all targets are forwarded to ``sphinx-build -M``), with one additional target, ``ghpages``, which runs ``scripts/empty_docs.py``, then builds the HTML docs from scratch into ``../docs``. This fixes the inconsistency between both files, as well as the ``HashTri`` leftovers. Note the change of behavior: ``make.bat html`` now builds into ``sphinx/_build/html``, like ``make html``; it no longer writes into ``docs/``. ``make.bat`` was saved with CRLF line endings (git stores it with LF, as before).
- ``sphinx/installation.rst``: new section *Build the documentation*, which documents the two commands of option c (run from the root of the project)::

    python scripts/empty_docs.py
    python -m sphinx -b html -E -d sphinx/_build/doctrees sphinx docs

  followed by the commit and push of ``docs/``. The ``make`` files are not mentioned.
- ``CLAUDE.md``: the note on ``docs/`` in *Repository layout notes* now gives the same commands (and mentions the ``ghpages`` target), and says that ``docs/`` should only be regenerated on request, since it is the published site.

Tests (all performed on a copy of ``sphinx/``, ``scripts/`` and ``docs/`` in a temporary directory; the actual ``docs/`` directory was not modified):

- The documented commands (option c): ``docs/`` is emptied, ``.nojekyll`` is kept, the obsolete files (``jquery*.js``, ``underscore*.js``) are removed, and the build succeeds (2 warnings, related to ``janus.fft.parallel``).
- ``make.bat`` (help, including the description of the ``ghpages`` target), ``make.bat ghpages`` (same result as the documented commands) and ``make.bat html`` (builds into ``_build/html``) work as expected.
- ``sphinx/Makefile`` was not tested: ``make`` is not available on this machine.
- ``make.bat latexpdf`` (TeX Live 2026 is installed on this machine) **fails**, for a reason unrelated to the ``make`` files: with the default LaTeX engine (``pdflatex``), and the ``latex_elements`` of ``sphinx/conf.py`` (which empty ``inputenc`` and ``fontenc``), LaTeX stops on a Unicode character (``δ``, U+03B4). With ``-D latex_engine=xelatex``, the PDF is produced, but ``latexmk`` then fails on the index, because Sphinx uses ``xindy`` with ``xelatex``, and ``xindy`` fails on this machine. With ``-D latex_engine=xelatex -D latex_use_xindy=0`` (``makeindex`` is then used), the build succeeds: ``sphinx/_build/latex/janus.pdf``, 66 pages, no missing character. ``sphinx/conf.py`` was not modified (see suggestions).
- Note that ``make.bat`` returns exit code 0 even when the build fails (this is also the case of the template generated by ``sphinx-quickstart``): the output must be read.

Follow-up: PDF settings added to ``conf.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the author's request, ``latex_engine = 'xelatex'`` and ``latex_use_xindy = False`` (with a comment explaining the latter) were added to ``sphinx/conf.py``, just before ``latex_elements``, which was left unchanged. On a copy of ``sphinx/`` in a temporary directory, ``make.bat latexpdf`` (without any ``-D`` option) now succeeds: ``_build/latex/janus.pdf``, 66 pages, no missing character. The HTML build is unaffected (2 warnings, related to ``janus.fft.parallel``).

Follow-up: ``latex_elements`` removed from ``conf.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the author's request, the ``latex_elements`` overrides were reviewed. The dictionary set ``preamble``, ``inputenc``, ``fontenc`` and ``utf8extra`` to empty strings. The last one was added in 2016 (commit ``18071f2``, *Remove* ``\DeclareUnicodeCharacter``), before Sphinx offered the ``latex_engine`` option (Sphinx 1.5): these overrides most likely allowed the generated ``.tex`` file to be compiled with ``xelatex`` by hand, by removing the ``pdflatex``-specific parts of the preamble.

Now that ``latex_engine = 'xelatex'`` is set, Sphinx adapts these entries itself, and the overrides are either redundant or counterproductive:

- ``preamble`` and ``inputenc``: the defaults are already empty strings (with ``xelatex``, for the latter);
- ``fontenc``: with ``xelatex``, the default loads ``fontspec`` explicitly (``\usepackage{fontspec}`` and ``\defaultfontfeatures…``); the override removed these lines (``fontspec`` was still loaded indirectly, by ``polyglossia``);
- ``utf8extra``: with ``xelatex``, the default makes the non-breaking space (U+00A0) behave as ``~``; the override removed this.

The whole ``latex_elements`` dictionary was therefore removed from ``sphinx/conf.py``. On a copy of ``sphinx/`` in a temporary directory, ``make.bat latexpdf`` succeeds: 66 pages, no missing character. The only differences in the preamble of the generated ``janus.tex`` are the three lines listed above (``fontspec``, ``\defaultfontfeatures`` and non-breaking space), which are now present.

Suggestions
-----------

- *Applied (options b and c), see above.* Make ``sphinx/Makefile`` consistent with ``sphinx/make.bat`` (build into ``../docs``), or replace both by a single documented command.
- *Applied, see above.* Add the ``sphinx.ext.githubpages`` extension to ``sphinx/conf.py``: it creates ``.nojekyll`` in the output directory automatically, so that ``docs/`` can be emptied without special care.
- In the longer term, a GitHub Actions workflow could build the docs on each push and deploy them to GitHub Pages. The generated HTML files would then no longer be versioned, and building under Linux (with MPI) would include the API of ``janus.fft.parallel``. This requires compiling `Janus` in the workflow.
- *Applied, see above.* Replace the absolute local path in the report of *Installation on a windows machine* by a generic one (``C:\path\to\miniconda3\…``) before publishing.
- *Applied, see above.* To build the PDF version of the docs, add ``latex_engine = 'xelatex'`` and ``latex_use_xindy = False`` to ``sphinx/conf.py`` (tested with TeX Live 2026, see above). The ``latex_elements`` overrides (empty ``inputenc``, ``fontenc`` and ``utf8extra``) could then be reviewed: they seem to date from an earlier attempt at supporting Unicode characters.


2026-09-11 • Automatic creation of environments
===============================================

Following your suggestion, add an ``environment.yml`` file, to create the `conda` environment in one command.

This environment should also include the packages that are required to compile the Sphinx docs.

Claude's report
---------------

Claude created the file ``environment.yml`` at the root of the project. It defines the ``janus`` environment (channel ``defaults``, as the existing environment), with the following packages: ``python=3.14``, ``pip``, ``setuptools``, ``cython``, ``fftw`` (build), ``numpy`` (run), ``pytest`` (tests) and ``sphinx`` (documentation). The environment is created with::

  conda env create -f environment.yml
  conda activate janus

and an existing environment is synchronized with the file with::

  conda env update -n janus -f environment.yml

Only the Python version is pinned; the other packages take the latest version available in the channel.

The documentation only requires ``sphinx``: all extensions listed in ``sphinx/conf.py`` are bundled with Sphinx (including ``sphinx.ext.napoleon``), the default theme (``alabaster``) is installed as a dependency of Sphinx, and the tutorials only include static figures and source files (``literalinclude``), which are not executed. Note however that ``autodoc`` imports `Janus`, which must therefore be compiled and installed first.

`Janus` itself is not installed by ``environment.yml``: the build requires ``setup.cfg`` and the ``--no-build-isolation`` flag, neither of which can be expressed in the ``pip`` section of an environment file. The comments at the top of the file recall the remaining steps, and warn against adding ``mpi4py`` on Windows.

The file was checked as follows.

- ``conda env create -f environment.yml -n janus-envcheck --dry-run``: the environment resolves (Sphinx 9.1.0).
- ``conda env update -n janus -f environment.yml``: this added Sphinx to the existing ``janus`` environment, and upgraded ``cython`` (3.2.5 → 3.2.9) and ``fftw`` (3.3.10 → 3.3.11). The serial tests still pass (3043 passed, 111 skipped).
- ``python -m sphinx -b html -E sphinx <temporary directory>``: the build succeeds, with 3 warnings. The ``_static`` directory listed in ``html_static_path`` does not exist (1 warning), and ``janus.fft.parallel`` cannot be imported by ``autodoc``, since the parallel extension is not built (2 warnings). The documentation of this module is therefore missing from the API page. The ``docs/`` directory was not modified.

Follow-up: first two suggestions applied
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the author's request, the first two suggestions below were applied.

- ``sphinx/requirements.txt`` was deleted. In ``sphinx/conf.py``, the test on ``sphinx.__version__`` was removed, and ``sphinx.ext.napoleon`` was added directly to the list of ``extensions``.
- ``html_static_path`` was removed from ``sphinx/conf.py``, rather than creating an empty ``sphinx/_static`` directory (an empty directory cannot be tracked by git; besides, ``_static`` is listed in ``sphinx/.gitignore``).

The documentation was built again: the build succeeds, with 2 warnings (both related to ``janus.fft.parallel``, see above). The docstrings are still processed by ``napoleon`` (the *Parameters* and *Returns* sections are rendered in the API page).

Follow-up: proposed text for ``sphinx/installation.rst``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following sub-section is proposed as a replacement for the current *Compilation with Anaconda and Visual Studio* sub-section of ``sphinx/installation.rst``. It has not been applied. It summarizes the procedure of the task *Installation on a windows machine*, which was only tested under Windows (hence its location in the Windows section)::

  Compilation with Miniconda and Visual Studio (recommended)
  ----------------------------------------------------------

  This procedure was tested with Miniconda, Python 3.14 and Visual Studio Build Tools 2026.

  1. Install the `Visual Studio Build Tools`_, with the *Desktop development with C++* workload. The compiler is located automatically by ``setuptools``: there is no need to use a *Developer Command Prompt*.

  2. From the root of the project, create and activate the ``janus`` environment. It provides all the dependencies of Janus (including FFTW), as well as the packages that are required to run the tests and build the documentation::

       conda env create -f environment.yml
       conda activate janus

     To synchronize an existing environment with ``environment.yml``, use ``conda env update -n janus -f environment.yml`` instead.

  3. Create the ``setup.cfg`` file. FFTW is installed in the ``Library`` subdirectory of the environment, whose path is printed by ``echo %CONDA_PREFIX%``. Environment variables are not expanded in ``setup.cfg``: this path must be written in full::

       [fftw]
       include_dirs = C:\path\to\miniconda3\envs\janus\Library\include
       library_dirs = C:\path\to\miniconda3\envs\janus\Library\lib
       libraries = fftw3

     Note that, unlike the precompiled binaries downloaded from fftw.org, the library provided by conda is called ``fftw3`` (without the ``lib`` prefix).

  4. Install Janus in development (editable) mode::

       pip install --no-build-isolation -e .

     The ``--no-build-isolation`` flag ensures that the versions of Cython and setuptools installed in the environment are used for the build (otherwise, pip downloads the latest versions from PyPI).

     After modifying a ``*.pyx`` or ``*.pxd`` file, recompile the extension modules in place with ``python setup.py build_ext --inplace``.

  .. warning:: Do not install ``mpi4py`` in this environment: ``setup.py`` would then try to build the parallel version of Janus, which is not tested under Windows, and requires ``mpicc``.

  .. _Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

If this text is adopted, the note on the ``lib`` prefix in ``CLAUDE.md`` should be updated accordingly.

Follow-up: proposed text applied, ``CLAUDE.md`` corrected
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The author approved the above text, which now replaces the *Compilation with Anaconda and Visual Studio* sub-section of ``sphinx/installation.rst``.

The *Build* section of ``CLAUDE.md`` was corrected as follows.

- The FFTW library name on Windows depends on where FFTW comes from: ``fftw3`` with `conda`'s ``fftw`` package (recommended), ``libfftw3-3`` with the precompiled DLLs from fftw.org. ``CLAUDE.md`` previously stated that the ``lib`` prefix must always be kept.
- ``environment.yml`` and the editable install (``pip install --no-build-isolation -e .``) are mentioned, instead of ``python setup.py install --user``.
- ``python setup.py clean`` does not remove the compiled files any more (commit ``40eee5e``): ``git clean -Xfd janus/`` is mentioned instead.
- If ``mpi4py`` is installed but ``mpicc`` cannot be found, the build most likely fails (``CLAUDE.md`` stated that the parallel extension was then silently skipped).

The documentation was built again, without new warnings.

Note that the introduction of the Windows section of ``sphinx/installation.rst`` still reads "You must first download and install the precompiled binaries of FFTW for Windows". This is not required by the new (recommended) procedure, where FFTW is provided by `conda`. At the author's request, this sentence was removed, together with the ``setup.cfg`` example and the note on the ``lib`` prefix that it introduced (and the now unused link to the precompiled binaries). The Windows section now only covers the `conda` procedure (and the unfinished MinGW/MSYS procedure).

Suggestions
-----------

- *Applied, see above.* ``sphinx/requirements.txt`` only lists ``sphinxcontrib-napoleon``, which has been bundled with Sphinx (as ``sphinx.ext.napoleon``) since version 1.3. This file, as well as the corresponding test on ``sphinx.__version__`` in ``sphinx/conf.py``, could be removed.
- *Applied, see above.* Remove ``html_static_path`` from ``sphinx/conf.py`` (or create the ``sphinx/_static`` directory) to get rid of the corresponding warning.
- *Applied, see above.* Once the installation procedure is stabilized, mention ``environment.yml`` in ``sphinx/installation.rst``.


2026-09-11 • Installation on a windows machine (`Miniconda` environment)
========================================================================

Following the update of the installation procedure, I am not sure how to install `Janus` on a windows machine. Use the `conda` environment `janus` to install the local version of `janus` in development mode. Add all required packages. Write down the procedure in your report. In particular, identify the fragile steps of the procedure (if any).

Claude's report
---------------

What was done
~~~~~~~~~~~~~

`Janus` was already installed in the `janus` environment, in editable mode. To check the procedure from scratch, Claude uninstalled it (``pip uninstall janus``), removed all build artifacts (``git clean -Xfd janus/``, plus the ``build/`` and ``Janus.egg-info/`` directories at the root), then reinstalled it with the procedure below. The build succeeded, and the serial tests pass (3043 passed, 111 skipped; all skipped tests are 3D, fourth-rank cubic tensors, which are not implemented yet).

No package had to be added: the environment already contained everything required (``python`` 3.14, ``cython``, ``setuptools``, ``pip``, ``numpy``, ``fftw``, ``pytest``, all from the ``pkgs/main`` channel). The parallel version was not installed (``mpi4py`` is absent, see below).

Procedure
~~~~~~~~~

1. Install the Microsoft C/C++ compiler: *Visual Studio Build Tools*, with the *Desktop development with C++* workload. It is not provided by `conda`, but `setuptools` locates it automatically (no need for a *Developer Command Prompt*). *Visual Studio Build Tools 2026* was used here.

2. Create and activate the environment (the creation command was checked with ``conda create --dry-run``)::

     conda create -n janus python=3.14 cython fftw numpy pytest setuptools pip
     conda activate janus

3. Create the ``setup.cfg`` file at the root of the project. FFTW is provided by `conda`, in ``%CONDA_PREFIX%\Library``. Environment variables are not expanded in ``setup.cfg``, so the absolute path must be written in full (``echo %CONDA_PREFIX%`` prints it). Note that the library name is ``fftw3`` (the `conda` package provides ``fftw3.lib`` and ``fftw3.dll``)::

     [fftw]
     include_dirs = C:\path\to\miniconda3\envs\janus\Library\include
     library_dirs = C:\path\to\miniconda3\envs\janus\Library\lib
     libraries = fftw3

4. Install `Janus` in development (editable) mode, from the root of the project::

     pip install --no-build-isolation -e .

   The extension modules (``*.pyd``) are compiled in place, next to the ``*.pyx`` sources. Changes to the ``*.py`` files are picked up immediately.

5. Run the tests::

     python -m pytest tests

6. After editing a ``*.pyx`` or ``*.pxd`` file, recompile the modified extensions in place (only the modified modules are recompiled)::

     python setup.py build_ext --inplace

   Re-running ``pip install --no-build-isolation -e .`` also works.

Fragile steps
~~~~~~~~~~~~~

- **The** ``setup.cfg`` **file.** It is git-ignored, written by hand, and contains absolute paths that depend on the machine and on the name of the environment. It must be rewritten on each new machine, or if the environment is renamed.
- **The FFTW library name.** With the `conda` package, the library name is ``fftw3``. The Windows section of ``sphinx/installation.rst`` (and ``CLAUDE.md``) state that the ``lib`` prefix must be kept (``libfftw3-3``): this is true only for the precompiled DLLs downloaded from fftw.org.
- **Build isolation.** Plain ``pip install -e .`` also works, but pip then downloads the latest `setuptools` and `Cython` from PyPI (84.0.0 and 3.3.0 at the time of writing), instead of using the versions installed by `conda` (83.0.0 and 3.2.5). This requires network access, and the versions used for the build are not pinned. Hence the ``--no-build-isolation`` flag.
- **mpi4py.** It must *not* be installed in the environment. ``setup.py`` tries to build the parallel extension as soon as ``mpi4py`` can be imported, and only catches ``ImportError``. On Windows, there is usually no ``mpicc``, and the build would most likely fail. This was inferred from reading ``setup.py``, not tested.
- **distutils.** ``setup.py`` imports ``distutils``, which was removed from the standard library in Python 3.12. It currently works only because `setuptools` provides a replacement ``distutils`` module; a future version of `setuptools` might drop it. ``setup.py`` also raises a ``SyntaxWarning`` (invalid escape sequence ``"\S"``, line 86).
- **The** ``packages`` **list in** ``setup.py``. It omits ``janus.material``, ``janus.material.elastic`` and ``janus.material.elastic.linear``, and `setuptools` issues several "Package would be ignored" warnings. This is harmless for an editable install (the whole ``janus/`` directory is mapped), but a regular (non-editable) install would probably lack these subpackages. Not tested.
- **Stale compiled files.** The ``*.c`` and ``*.pyd`` files live in the source tree. They must be rebuilt after editing ``*.pyx``/``*.pxd`` files, or after upgrading Python. ``python setup.py clean`` does not remove them; use ``git clean -Xfd janus/``.

Some points turned out *not* to be fragile. The FFTW DLL (``fftw3.dll``) is found at import time even when the environment is not activated (checked with ``conda`` directories removed from ``PATH``): `conda`'s Python adds ``Library\bin`` to the DLL search path. The compiler emits many ``C4244`` warnings (conversions from ``Py_ssize_t``/``ptrdiff_t`` to ``int``), which do not prevent the build.

Suggestions
-----------

- Update the Windows section of ``sphinx/installation.rst`` with the above procedure, and fix the note on the ``lib`` prefix in ``CLAUDE.md``.
- Add an ``environment.yml`` file, to create the environment in one command.
- In ``setup.py``, replace the ``distutils`` imports with their `setuptools` counterparts, fix the ``"\S"`` escape sequence (raw string), complete the ``packages`` list, and make the build of the parallel extension skip gracefully when ``mpicc`` cannot be found.
- In ``setup.py``, default the FFTW paths to ``sys.prefix\Library`` when running in a `conda` environment on Windows, so that ``setup.cfg`` becomes optional.

2026-09-11 • How to update this page
====================================

@Claude: this paragraph describe how we are to interact for the development of `Janus`. Take note of these instructions so that I don't have to repeat myself at each new session.

Each time you are invoked in a console, you will read the present file and find the paragraphs that are marked `TODO`.

These are your instructions for the current session: each paragraph is an elementary task. As far as possible, each task will be given its own section with a short, unique title.

After completion of a task, report on what you did in a sub-paragraph called `Claude's report` of the `claude.rst` file.

Once I have validated your changes, I will ask you to mark the task as done. This means you will replace the `TODO` flag with the date in yyyy-mm-dd format. You will then move to the next task. For tasks that cannot be completed in one session, use an `IN PROGRESS` flag, so that the state of the task is visible in the present file.

Questions that are still open, on which I have not made a decision yet, are recorded in sections marked `TO DISCUSS`. These are not tasks: you will not act on them, but raise them with me when they become relevant. Once a decision is made, the section is either turned into a `TODO` task, or marked with the date of the decision, together with a short statement of that decision.

Unless otherwise stated, you will never commit your changes.

Before you mark the present task as done, you can collect in a sub-paragraph any suggestions that might improve our interactions.

For these instructions to apply on any machine, add the following pointer in `CLAUDE.md`

  at session start, read `sphinx/claude.rst` and process the `TODO` sections according to the `How to update this page` section


Claude's report
---------------

The pointer above was added to `CLAUDE.md`, in a new `Session workflow` section at the top of the file. Since `CLAUDE.md` is loaded at the start of every session, the present file is now the single source of truth for our workflow, on any machine.

The copy of these instructions that had been saved in Claude Code's local memory (under `~/.claude/projects/`, on the author's machine only) was deleted: it duplicated the present file and was already out of date (it did not require the author's validation before marking a task as done).

The author validated these changes on 2026-09-11, and the task was marked as done.
