# Shifted GD-lambda-AMG V-cycle Correction Fix

## Objective

Replace the coarse-only correction preconditioner used by the current
GD-lambda-AMG experiment with a complete, matrix-free shifted two-level AMG
preconditioner for `A(theta) = D - theta*Gamma5`.

The fix must preserve completed experiment outputs, preserve the generalized
Davidson and harmonic Ritz contracts, and establish with tests that the inner
correction reduces its directly recomputed residual before any new scientific
run is launched.

## Confirmed defect

The current `gd_correction_solver` supplies FGMRES with
`shifted_coarse_correction!` as its right preconditioner. That operation only
restricts, solves the Galerkin coarse system, and prolongates. It omits the
fine-grid smoother that is part of a DD-alphaAMG cycle. Therefore the completed
experiment measured coarse-only preconditioned GD, not the intended shifted
DD-alphaAMG-preconditioned GD.

The following existing algebra remains unchanged:

- `Q = Gamma5*D`;
- `r = Q*u - theta*u`;
- `(D - theta*Gamma5)t = Gamma5*r`;
- harmonic Ritz extraction nearest zero;
- thick restart dimensions `m_min=30` and `m_max=50`.

## Selected design

Add a dedicated shifted two-level V-cycle rather than passing the shifted
operator through the existing Wilson-specific `vcycle` blindly.

For a fine right-hand side `b`, the preconditioner computes:

1. zero initial correction `x`;
2. a small fixed number of generic fine-grid GMRES smoothing steps on
   `A(theta)*x=b`;
3. the explicit fine residual `rho=b-A(theta)*x`;
4. the existing shifted Galerkin coarse correction
   `P*(P' A(theta) P)^(-1)*P'*rho`;
5. correction accumulation into `x`;
6. a small fixed number of generic post-smoothing steps on the recomputed
   residual, followed by accumulation into `x`.

Generic GMRES smoothing is selected instead of directly reusing the optimized
SAP kernels because `A(theta)` is an operator wrapper, while several SAP GPU
paths assume Wilson-operator internals and fused stencil access. This design is
slower than a future shifted SAP kernel but is the safer correctness baseline.
It matches the paper's allowance for GMRES smoothing of indefinite shifted
systems and avoids changing low-level CUDA kernels during this fix.

The smoother iteration counts are explicit required experiment settings. They
will not be inferred or changed during a run.

## Interfaces and accounting

Add `shifted_amg_vcycle!` to `LMP/src/GDlambdaAMG.jl`. It accepts caller-owned
output, fine RHS, shifted operator, and an existing `DDalphaAMG_Setup`, plus
explicit smoothing and coarse-solve parameters.

It returns diagnostics containing:

- directly recomputed input and output residual norms;
- pre- and post-smoothing applications;
- coarse applications;
- total fine shifted-operator applications.

`gd_correction_solver` will use this complete cycle as the FGMRES right
preconditioner. Outer `Q` applications, inner shifted fine applications,
smoother applications, and coarse applications remain separately reported so
unlike units are not combined silently.

## Error handling

- Reject negative smoothing counts and invalid tolerances.
- Reject non-finite residuals immediately with the affected stage named.
- Never silently fall back to coarse-only correction.
- Preserve explicit cap and rank-loss states in generalized Davidson.
- Refuse to overwrite all prior result directories.

## Test-first verification

Before production changes, add tests that fail against the coarse-only code:

1. a deterministic dense shifted two-level fixture proving pre- and
   post-smoothers are invoked and counted;
2. direct verification of
   `norm(b-A(theta)*x)/norm(b)` before and after one complete cycle;
3. a comparison demonstrating the full cycle is distinct from coarse-only
   correction;
4. a small Hermitian GD integration fixture using the new correction path;
5. source/integration coverage proving the experiment wrapper calls the full
   shifted cycle and exposes explicit smoother settings.

After CPU tests pass, run the existing GD, coarse-grid, and DDalphaAMG
regression tests. Then run one GPU1 configuration-401 smoke test in a new
directory. Acceptance requires finite correction histories, recorded smoother
work, and materially decreasing correction residual. A full 401--410 rerun is
not launched automatically; its parameters and launch require a separate
explicit decision after inspecting the smoke result.

## Non-goals

- Modifying completed experiment data.
- Changing kappa, configuration IDs, outer tolerance, subspace sizes, or vector
  counts.
- Implementing a new optimized shifted SAP CUDA kernel.
- Claiming paper-equivalent performance before the corrected smoke test.
- Dynamically updating interpolation with converged eigenvectors.
