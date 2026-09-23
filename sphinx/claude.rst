*********************************
Claude's contributions to `Janus`
*********************************

As of august 2026, `Janus` is revived by the author with the help of `Claude Code`. This page will collect all interactions between the author and Claude.


TODO • Removal of the vestiges of distributed memory
====================================================

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

TODO • Step 4 — ``global_shape0``
---------------------------------

``janus/green.pyx``, ``CLAUDE.md``.

``global_shape0`` is equal to ``shape0``, which ``init_shapes`` sets just above its last use. Replace it with ``self.shape0`` in the computation of ``s0`` (``DiscreteGreenOperator2D``/``3D``) and in the wave-vectors of ``TruncatedGreenOperator2D``/``3D`` and ``FilteredGreenOperator2D``/``3D``, then remove the attribute. ``FiniteDifferences2D``/``3D`` does not use it.

The constructors also read ``transform.global_ishape``, to check the shape of the transform: switch that check to ``transform.ishape``. Step 5 removes the attribute, so this step must come first.

In ``CLAUDE.md``, delete the paragraph that presents ``offset0``/``global_shape0`` as a pending simplification.

TODO • Step 5 — ``global_ishape``, ``global_oshape`` and ``n0_loc``
-------------------------------------------------------------------

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

Unless otherwise stated, you will never commit your changes.

Before you mark the present task as done, you can collect in a sub-paragraph any suggestions that might improve our interactions.

For these instructions to apply on any machine, add the following pointer in `CLAUDE.md`

  at session start, read `sphinx/claude.rst` and process the `TODO` sections according to the `How to update this page` section


Claude's report
---------------

The pointer above was added to `CLAUDE.md`, in a new `Session workflow` section at the top of the file. Since `CLAUDE.md` is loaded at the start of every session, the present file is now the single source of truth for our workflow, on any machine.

The copy of these instructions that had been saved in Claude Code's local memory (under `~/.claude/projects/`, on the author's machine only) was deleted: it duplicated the present file and was already out of date (it did not require the author's validation before marking a task as done).

The author validated these changes on 2026-09-11, and the task was marked as done.
