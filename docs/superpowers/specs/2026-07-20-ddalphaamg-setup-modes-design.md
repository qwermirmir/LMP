# DD-alphaAMG Setup Modes Design

## Goal

Preserve the validated paper-style DD-alphaAMG implementation while making the
2026-07-14 direct near-null algorithm reproducible from the current codebase by
changing one explicit setup mode.

## Public API

`setup_DDalphaAMG` accepts:

```julia
setup_mode::Symbol=:inverse_sap
```

The supported values are:

- `:inverse_sap`: the current validated paper-style inverse-SAP adaptive setup.
- `:direct_near_null`: the algorithm used to produce the 2026-07-14 comparison
  results stored under
  `results/logs/lmp_nv12_24_48_64_vs_vcycle_nv12_24_48_64_sap3_mr4_nvcycle6_tol1e10_maxiter200_k157_k1586_cfg450_20260714`.

The selected mode is stored in `DDalphaAMG_Setup`. Existing positional setup
constructors remain source-compatible and default to `:inverse_sap`.

`solve_DDalphaAMG` reads the setup mode automatically. Mode-dependent keyword
arguments may still be supplied explicitly; explicit values override mode
defaults.

## Inverse-SAP Mode

`:inverse_sap` retains the behavior of commit `c6602cf` without numerical or
default-setting changes. It remains the default so existing calls continue to
use the validated implementation.

Its setup behavior includes staged inverse smoothing, the current adaptive
V-cycle update, boundary-eight SAP by default, and odd-even preconditioning.

## Direct Near-Null Mode

`:direct_near_null` reproduces the algorithm in commit `ff3d1bf`.

For random prototype vectors, phase 1 performs homogeneous smoothing:

```math
v_j \leftarrow S_{\mathrm{SAP}}^{(3)} v_j
```

where the smoother is invoked on `D v = 0`. For external LMP vectors, the same
homogeneous smoothing is applied when `smooth_external_vectors=true`.

Each adaptive setup iteration constructs the chiral prolongator and coarse
operator from the current vectors, then performs:

```math
r_j = v_j-Dv_j,
```

```math
e_{c,j}\approx D_c^{-1}Rr_j,
```

```math
v_j\leftarrow
\operatorname{normalize}\left(v_j + S\left(Pe_{c,j},r_j\right)\right).
```

This is the exact update order used in `ff3d1bf`: prolongation initializes the
correction, SAP or MR post-smoothing refines that correction, and the correction
is added to the previous vector.

Direct near-null setup uses the historical internal defaults unless explicitly
overridden:

- two-color SAP schedule;
- no SAP block odd-even preconditioning;
- no coarse odd-even preconditioning;
- adaptive coarse GMRES tolerance `1e-2`;
- adaptive coarse GMRES maximum cycles `50`;
- adaptive coarse GMRES restart length `20`;
- one adaptive post-SAP sweep with four block-local MR steps.

## Solve Defaults

When solving a setup in `:direct_near_null` mode and no corresponding override
is supplied, `solve_DDalphaAMG` uses the settings from the plotted experiment:

- outer FGMRES restart `50`;
- fine smoother `:mr`;
- pre-smoothing steps `4`;
- post-smoothing steps `4`;
- coarse tolerance `1e-1`;
- coarse maximum cycles `50`;
- coarse restart length `20`;
- restart-boundary scalar coarse GMRES, matching `ff3d1bf` rather than the
  newer incremental-Givens implementation;
- coarse odd-even disabled;
- SAP odd-even disabled;
- two-color SAP schedule.

The convergence tolerance and outer maximum cycles remain ordinary experiment
parameters. The 2026-07-14 comparison explicitly used `tol=1e-10` and
`maxiter=200`.

For `:inverse_sap`, all current solve defaults remain unchanged.

The modes can therefore be switched without checking out another revision:

```julia
setup = setup_DDalphaAMG(D, x, 4, Nv;
    setup_mode=:direct_near_null,
    # remaining experiment settings
)
```

Changing only `setup_mode` to `:inverse_sap` selects the validated paper-style
path. The selected mode is carried by `setup` into `solve_DDalphaAMG`.

## LMP Initial Guess

The setup mode does not implicitly create a deflated initial guess. The LMP
comparison continues to request it explicitly:

```julia
x0 = deflated_initial_guess(D, b, setup.test_vectors)
solve_DDalphaAMG(D, b, setup; x0=x0, tol=1e-10, maxiter=200)
```

This avoids changing non-LMP solves and makes the comparison protocol visible.

## Validation

Tests must establish all of the following:

1. Omitting `setup_mode` is identical to `setup_mode=:inverse_sap`.
2. Invalid modes fail before setup work begins.
3. Existing positional `DDalphaAMG_Setup` construction remains compatible.
4. Direct near-null external smoothing matches explicit homogeneous SAP/MR.
5. A direct near-null adaptive update matches the `ff3d1bf` operation order on
   a deterministic small lattice.
6. Direct near-null solve defaults resolve to the 2026-07-14 settings, while
   explicit keyword overrides take precedence.
7. The inverse-SAP test suite remains unchanged and passes.
8. A GPU smoke test confirms both modes build a stencil and complete one
   V-cycle without host-only fallback in the mode dispatch.

The historical large `cfg450` experiment is a final reproduction run, not a
unit test. Its gauge, kappa values, checkpoints, vector counts, seeds, and
solver settings must be printed before execution and must not be changed by the
mode implementation.

## Non-Goals

- Reclassifying direct near-null setup as the paper implementation.
- Changing the numerical behavior or defaults of `:inverse_sap`.
- Automatically rerunning or deleting existing experiment results.
- Combining individual internals from the two modes into undocumented hybrids.
