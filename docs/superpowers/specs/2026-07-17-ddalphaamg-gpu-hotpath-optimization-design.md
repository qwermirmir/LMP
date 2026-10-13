# DD-alphaAMG GPU Hot-Path Optimization Design

## Goal

Reduce DD-alphaAMG V-cycle time without changing the public solver API, solver
parameters, `ComplexF64` arithmetic, or the mathematical `boundary8` SAP and
coarse-GMRES algorithms.

## Scope

This change has two targets:

1. Replace each full-lattice `D * delta` residual update inside GPU
   `boundary8` SAP with an update restricted to the active blocks and their
   one-site Wilson halo.
2. Replace per-vector GPU reductions and updates in coarse Arnoldi with batched
   GPU matrix-vector operations.

Mixed precision, setup algorithm changes, interpolation changes, tolerance
changes, and CPU-path optimization are out of scope.

## SAP Residual Update

For one `boundary8` step, the correction `delta` is zero outside that step's
active blocks. Since Wilson `D` is nearest-neighbor, `D * delta` can be nonzero
only on the active block sites and their nearest-neighbor halo.

The GPU cache will store a compact affected-site list for every schedule step.
A dedicated kernel will evaluate the same Wilson operator convention used by
`_amg_mul_dirac!` only at those sites and directly perform

```text
res(site) <- res(site) - (D * delta)(site).
```

The existing full-lattice implementation remains as a fallback for unsupported
operators, layouts, and schedules. The optimized result must agree with the
full-lattice update to relative error at most `1e-12`.

## Batched Coarse Arnoldi

On GPU, the Arnoldi basis will additionally be maintained as a packed matrix
whose columns are basis vectors. One Arnoldi orthogonalization step will use:

```text
h <- V' * w
w <- w - V * h
```

through GPU BLAS operations. A second batched pass will reorthogonalize `w` to
retain the stability of the current modified Gram-Schmidt implementation. Only
the small coefficient vector is transferred to the host for the existing
Hessenberg and Givens logic.

CPU coarse GMRES and unsupported GPU storage retain the current scalar MGS
path. The optimized coarse solution must agree with the current path to
relative error at most `1e-12` on deterministic regression problems and reach
the same true-residual tolerance in solver tests.

## Cache And Fallbacks

- Schedule site lists are created once in `_ensure_gpu_cache!` and reused.
- Packed Arnoldi storage is resized only when coarse dimensions or restart
  length change.
- Existing setup objects remain valid; caches are rebuilt lazily.
- Optimized kernels are selected only for the fused GPU Wilson layout already
  recognized by `_amg_can_fuse_wilson`.
- No environment variables or hidden runtime switches are introduced.

## Verification

Implementation follows test-first development:

1. Compare sparse SAP residual updates with full `D * delta` on every
   `boundary8` step.
2. Compare one batched Arnoldi orthogonalization step with scalar MGS, including
   orthogonality and norm checks.
3. Compare complete coarse solves and one V-cycle on deterministic inputs.
4. Run existing odd-even Schur, `boundary8`, and GPU SAP regression tests.
5. Run a short benchmark after printing all settings. Confirm lower
   full-lattice fine-operator counts and report wall time without changing the
   requested scientific configuration.

## Success Criteria

- Numerical comparison errors are at most `1e-12` where direct operation-level
  equivalence is required.
- Final solver checks reach the same requested true-residual tolerance.
- A `boundary8` SAP sweep no longer performs a full-lattice Wilson application
  after every schedule step on the supported GPU path.
- Coarse Arnoldi no longer performs one host-synchronizing GPU dot and one GPU
  vector update per existing basis vector.
- Existing CPU behavior and public APIs remain unchanged.
