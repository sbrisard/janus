# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Janus is a Python/Cython library for discretizing the Lippmann–Schwinger equation with periodic
boundary conditions (a matrix-free FFT-based method for homogenization of heterogeneous materials,
following Moulinec & Suquet). Performance-critical code is written in Cython; MPI (via `mpi4py`) is
used for the parallel (distributed FFT) variant. The library builds operators for iterative
(matrix-free) linear solvers — it does not itself implement the solvers; those come from
`scipy.sparse.linalg` or `petsc4py`.

## Build

The project is built with `setuptools` + `Cython`, compiling `.pyx`/`.pxd` sources into extension
modules (`.pyd`/`.so`). It links against FFTW3 (serial) and, optionally, an MPI-enabled FFTW3 build
(parallel).

Before building, `setup.cfg` must exist at the repo root with an `[fftw]` section (and `[fftw_mpi]`
if building the MPI-enabled extensions), e.g.:

```ini
[fftw]
include_dirs = C:\path\to\fftw\include
library_dirs = C:\path\to\fftw\lib
libraries = fftw3
```

On Windows, the FFTW library name must keep the `lib` prefix (e.g. `libfftw3-3`, not `fftw3`).

Build in place / install:

```
python setup.py build_ext --inplace   # compile extensions next to sources
python setup.py install --user
```

Clean compiled artifacts (`.c`, `.so`, `.pyd`, `.pyo`, `__pycache__`) under `janus/`:

```
python setup.py clean
```

The MPI/parallel extension (`janus.fft.parallel._parallel_fft`) is only built if `mpi4py` is
importable and `mpicc` can be located (`setup.py` probes `mpicc --show`/`--showme`); if `mpi4py`
is missing, `setup.py` silently skips the parallel package.

## Tests

Serial tests:

```
python -m pytest tests
```

Single test file / test:

```
python -m pytest tests/test_operators.py
python -m pytest tests/test_operators.py::TestAbstractOperator::test_init_sizes
```

Parallel tests (require the MPI-enabled build) must be run through `mpiexec`, and are excluded from
plain `pytest` runs — `pytest.ini` sets `norecursedirs = parallel data` so `tests/parallel` and
`tests/data` (reference `.npz` fixtures) are skipped by default:

```
mpiexec -np 3 pytest tests/parallel
```

Use an odd process count for parallel runs — it's more likely to reveal bugs in the data
distribution logic than an even count.

## Architecture

The core abstraction is the **operator hierarchy** in `janus/operators.pyx`, all implemented as
Cython `cdef class`:

- `AbstractOperator` — maps a flat vector of size `isize` to one of size `osize` via `c_apply`
  (Cython-level, no bounds checking) / `apply` (Python-facing, validates shapes). Subclassing in
  pure Python is supported via `init_sizes()`.
- `AbstractLinearOperator` — adds `c_to_memoryview`/`to_memoryview` to export the operator's matrix.
- `AbstractStructuredOperator2D` / `3D` — operators applied grid-wise: input/output are `(shape0,
  shape1, ishape2)` / `(..., oshape2)` arrays (one small vector per grid cell), not flat vectors.
- `BlockDiagonalOperator2D/3D`, `BlockDiagonalLinearOperator2D/3D` — apply a per-cell local operator/
  matrix independently at each grid point.
- `FourthRankIsotropicTensor2D/3D`, `FourthRankCubicTensor2D` — closed-form linear operators
  representing fourth-rank tensors in Mandel–Voigt notation (see `janus/mandelvoigt.py` for the
  index conventions and the `√2` scaling of off-diagonal components).

**Green operators** (`janus/green.pyx`) extend `AbstractLinearOperator` with a `set_frequency(k)`
step: a Green operator is evaluated in Fourier space at a wave-vector `k`, then applied like any
linear operator. Concrete continuum Green operators (e.g.
`janus/material/elastic/linear/isotropic.pyx: _GreenOperatorForStrains2D/3D`) implement
`c_set_frequency` and `c_apply` for a specific constitutive law.

`DiscreteGreenOperator2D/3D` wraps a continuum Green operator plus an FFT `transform` to act as a
structured operator over a real-space grid: `c_apply` performs, per grid component, an r2c FFT, a
frequency-wise `c_apply_by_freq` (the continuum Green operator evaluated at each discrete
wave-vector `b`), then a c2r inverse FFT. Variants differ in how continuous wave-vectors are derived
from the discrete grid index `b` and in whether the tensor is filtered:
- `TruncatedGreenOperator2D/3D` — Green operator evaluated exactly at the Fourier grid frequencies.
- `FilteredGreenOperator2D/3D` — a weighted average of the Green operator over 4 (2D) / 8 (3D)
  nearby frequencies, used to mitigate spurious oscillations (Willot's filtering scheme). The
  `c_set_frequency` bodies here are generated boilerplate — see `scripts/gencode.py` for the
  code-generation template if these need to be regenerated for a different tensor size.
- `FiniteDifferences2D/3D` (`willot2015`) — Green operator evaluated using a finite-difference
  discretization of the gradient instead of the exact Fourier symbol.

When an FFT `transform` is supplied and it exposes an `offset0`/local shape smaller than the global
shape, these classes operate on the local slab of a domain-decomposed grid (this is what enables the
parallel/MPI variant — see `janus/fft/parallel/`).

**FFT layer** (`janus/fft/`): `janus/fft/serial/_serial_fft.pyx` wraps FFTW3 real-to-complex/
complex-to-real transforms (`_RealFFT2D`/`3D`), exposing `ishape`/`oshape` (local) vs
`global_ishape`/`global_oshape` and an `offset0` for the local slab's position in the global array.
`janus/fft/parallel/_parallel_fft.pyx` is the MPI-enabled counterpart (`fftw3-mpi`), sharing the
same shape/offset attribute naming so `DiscreteGreenOperator*` code is agnostic to serial vs.
parallel transforms. `janus/fft/__init__.py` exposes the FFTW planner flag constants
(`FFTW_ESTIMATE`, `FFTW_MEASURE`, etc.).

**Materials** (`janus/material/`) mirror the mechanical constitutive-law hierarchy; currently only
`janus/material/elastic/linear/isotropic.pyx` is implemented, producing a Green operator via
`IsotropicLinearElasticMaterial.green_operator()`.

`janus/utils/checkarray.pyx` centralizes memoryview shape validation/allocation
(`check_shape_1d/3d/4d`, `create_or_check_shape_1d/2d/3d/4d`) used throughout `operators.pyx` and
`green.pyx` to validate `apply()`-style calls and allocate output buffers on demand.

### Public API shape conventions

- 2D structured data: `(shape0, shape1, vector_size)`; 3D: `(shape0, shape1, shape2, vector_size)`.
- Symmetric tensors are stored in reduced Mandel–Voigt form (`sym = dim*(dim+1)/2`, i.e. 3
  components in 2D, 6 in 3D) rather than full tensor form; see `janus/mandelvoigt.py`.
- `.pyx` files declare the public/`cimport`-able Cython API; matching `.pxd` files declare `cdef`
  members and methods used across modules (e.g. `green.pxd` exports `AbstractGreenOperator` for
  `isotropic.pyx` to extend via `cimport`).

## Repository layout notes

- `docs/` is generated Sphinx HTML output (published at http://sbrisard.github.io/janus/); the
  actual documentation sources live in `sphinx/` (`.rst` files, `conf.py`). Edit `sphinx/`, not
  `docs/`.
- `tests/data/*.npz` are reference/golden arrays used by the Green operator tests — don't regenerate
  these casually; `scripts/npy2npz.py`/`raw2npz.py`/`convert.py` are the conversion utilities that
  originally produced them.
- Generated/compiled files (`*.c`, `*.pyd`, `*.so`, `build/`) are gitignored local build artifacts
  present in this checkout, not sources — always edit the `.pyx`/`.pxd` files and rebuild.
