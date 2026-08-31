# GD-lambda-AMG DDalphaAMG Isolation Design

## Goal

Restore the tracked DDalphaAMG implementation and its existing setup-mode test
to the repository `HEAD`, while preserving the current GD-lambda-AMG solver's
numerical behavior in GD-owned code. No GD experiment may require a modification
to `src/DDalphaAMG.jl`.

## Non-negotiable boundaries

- `src/DDalphaAMG.jl` must be byte-for-byte identical to `HEAD` after the work.
- `test/ddalphaamg_setup_modes.jl` must be byte-for-byte identical to `HEAD`.
- The clean external C repository at `external/DDalphaAMG` remains untouched.
- Pre-existing July changes in `external/DDalphaAMG_nosse` remain untouched.
- Existing result directories and CSV/PNG outputs remain untouched.
- The GD experiment retains its kappa, configuration IDs, tolerances, vector
  counts, checkpoints, random seeds, setup stages, adaptive-cycle count, and
  solver limits.

## Architecture

Create `src/GDlambdaAMGDDalphaAdapter.jl` as the only integration boundary
between GD-lambda-AMG and the unchanged DDalphaAMG implementation. The adapter
may consume DDalphaAMG data structures and operations, but it must neither
redefine nor monkey-patch any DDalphaAMG method.

`src/GDlambdaAMG.jl` remains responsible for the generalized Davidson method,
harmonic Ritz extraction, shifted operator, correction equation, locking, and
auditing. It delegates only GD-specific AMG setup and coarse-grid work to the
new adapter.

The experiment driver calls the GD adapter instead of requesting experiment-only
outputs or behavior from `setup_DDalphaAMG`.

## Adaptive setup data flow

The adapter accepts either no prototypes or exactly 24 caller-provided
prototypes.

- With no prototypes, it creates the 24 deterministic random prototypes using
  the existing seeded DDalphaAMG random-field operation.
- With supplied prototypes, it clones all 24 fields so model-owned vectors are
  never mutated.
- Both sources then enter the same GD-owned setup pipeline:
  1. inverse-SAP stages 1, 2, and 3;
  2. six adaptive V-cycle updates;
  3. two-pass global orthogonalization and normalization before each update;
  4. the same chiral interpolation, Galerkin coarse operator, coarse tolerance,
     restart, cycle limit, SAP post-smoother, and block-MR settings.
- The adapter returns both the final unsplit refined prototypes and the final
  DDalphaAMG setup.
- The final setup is built by the original `setup_DDalphaAMG` using the already
  refined prototypes with external smoothing disabled. The original function
  alone performs the final chiral split, block orthonormalization, and coarse
  operator construction.

This preserves the current experiment's only intended difference: the source
of the initial 24 prototypes.

## GD-owned coarse solve

The adapter owns a GD-specific restarted coarse GMRES implementation for the
shifted coarse operator. It preserves the current solver behavior:

- odd-even Schur reduction when supported;
- scalar Arnoldi for the shifted correction;
- Arnoldi breakdown determined independently of the requested coarse tolerance;
- direct recomputation of the true residual before declaring convergence;
- reconstruction and verification of the full coarse solution;
- coarse application accounting used by the result CSV.

The implementation uses distinct `gd_` names and is called only by
`shifted_coarse_correction!`. Ordinary DDalphaAMG V-cycles and solves continue
to use the original `DDalphaAMG.coarse_gmres` unchanged.

## Failure handling

- A missing, non-finite, or rank-deficient prototype fails before hierarchy
  construction and identifies the failing column.
- The adapter rejects a prototype count different from the requested count.
- A singular projected GMRES system or non-finite residual terminates the GD
  case with a diagnostic error rather than modifying DDalphaAMG behavior.
- Reaching a configured GD expansion cap remains a recorded nonconvergence and
  does not abort the remaining configurations.
- Existing output-overwrite protection remains unchanged.

## Test strategy

Characterization tests first capture the behavior currently supplied by the
modified DDalphaAMG code. After restoring the DDalphaAMG files, those tests must
fail until the behavior is available through the GD adapter.

The completed implementation must pass all of the following:

1. `git diff --exit-code HEAD -- src/DDalphaAMG.jl`.
2. `git diff --exit-code HEAD -- test/ddalphaamg_setup_modes.jl`.
3. Existing DDalphaAMG CPU setup-mode and solver tests.
4. GD adapter tests showing that seeded random prototypes and the same supplied
   prototypes produce identical refined outputs.
5. GD adapter tests for two-pass global orthonormalization, rank rejection,
   shifted coarse residual verification, and odd-even Schur verification.
6. Existing generalized Davidson, harmonic Ritz, locking, and eigenpair-audit
   tests.
7. Experiment source tests proving it uses the GD adapter and does not rely on
   experiment-only DDalphaAMG keywords.
8. A GPU smoke run using one configuration and one expansion, with the exact
   production setup settings and without overwriting previous results.

## Acceptance criteria

- Both tracked DDalphaAMG files have zero diff against `HEAD`.
- No GD-only symbol or branch exists in `src/DDalphaAMG.jl`.
- Standard and learned cases follow the same GD-owned adaptive setup pipeline.
- The driver still emits the same summary, history, audit, application plot,
  and wall-time plot schemas.
- CPU tests and the GPU smoke run complete without unexpected errors.
- No existing experiment process, result, or unrelated working-tree change is
  stopped, deleted, restored, or overwritten.
