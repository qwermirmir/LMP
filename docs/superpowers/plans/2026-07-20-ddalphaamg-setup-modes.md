# DD-alphaAMG Setup Modes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit switch between the validated inverse-SAP setup and the historical direct near-null setup without changing the default numerical path.

**Architecture:** Store `setup_mode` in `DDalphaAMG_Setup`, dispatch setup smoothing and adaptive updates through mode-specific helpers, and resolve omitted solve options from the stored mode. Keep all coarse operator, V-cycle, and FGMRES kernels shared.

**Tech Stack:** Julia 1.11, CUDA.jl, LatticeDiracOperators.jl, Julia `Test`, Git historical source at `ff3d1bf`.

## Global Constraints

- `setup_mode=:inverse_sap` remains the default and preserves commit `c6602cf` behavior.
- `setup_mode=:direct_near_null` reproduces the setup/update order from `ff3d1bf`.
- Existing positional `DDalphaAMG_Setup` calls remain valid and select `:inverse_sap`.
- Explicit solve keywords override mode defaults.
- Do not modify or remove existing experiment outputs or unrelated untracked tests.
- Do not start the large cfg450 reproduction run without first printing its complete settings.

---

### Task 1: Mode Metadata And Default Resolution

**Files:**
- Modify: `src/DDalphaAMG.jl`
- Create: `test/ddalphaamg_setup_modes.jl`

**Interfaces:**
- Produces: `setup.setup_mode::Symbol`, `_amg_validate_setup_mode(::Symbol)`, `_amg_solve_defaults(::Symbol)`.
- Preserves: the existing 22-argument positional `DDalphaAMG_Setup{T}(...)` construction pattern.

- [ ] **Step 1: Write failing metadata/default tests**

Create tests that construct a minimal setup with the existing positional arguments, assert its mode is `:inverse_sap`, validate both supported symbols, reject an unknown symbol, and assert the exact solve default tuples:

```julia
@test DDalphaAMG._amg_validate_setup_mode(:inverse_sap) == :inverse_sap
@test DDalphaAMG._amg_validate_setup_mode(:direct_near_null) == :direct_near_null
@test_throws ArgumentError DDalphaAMG._amg_validate_setup_mode(:unknown)

@test DDalphaAMG._amg_solve_defaults(:inverse_sap) == (
    restart=25, smoother=:sap, pre_smooth=0, post_smooth=2,
    coarse_tol=5e-2, coarse_maxiter=50, coarse_restart=30,
    coarse_odd_even=true, sap_odd_even=true, sap_schedule=:boundary8,
)
@test DDalphaAMG._amg_solve_defaults(:direct_near_null) == (
    restart=50, smoother=:mr, pre_smooth=4, post_smooth=4,
    coarse_tol=1e-1, coarse_maxiter=50, coarse_restart=20,
    coarse_odd_even=false, sap_odd_even=false, sap_schedule=:twocolor,
)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
julia --project=. test/ddalphaamg_setup_modes.jl
```

Expected: failure because the mode helpers and field do not exist.

- [ ] **Step 3: Add the field, compatibility constructor, and helpers**

Append `setup_mode::Symbol` to `DDalphaAMG_Setup`. Add an outer compatibility constructor matching the previous field list and appending `:inverse_sap`. Implement exact validation and named-tuple solve defaults.

- [ ] **Step 4: Run the test and verify GREEN**

Run the same command and expect all metadata/default assertions to pass.

### Task 2: Direct Near-Null Setup Algorithm

**Files:**
- Modify: `src/DDalphaAMG.jl`
- Modify: `test/ddalphaamg_setup_modes.jl`

**Interfaces:**
- Produces: `_amg_initial_direct_smooth!` and `_amg_direct_near_null_update!`.
- Consumes: existing `sap_smooth!`, `mr_smooth!`, `restrict_f2c!`, `coarse_gmres`, `prolongate_c2f!`, and `normalize_fermion!`.

- [ ] **Step 1: Write failing operation-order tests**

Use deterministic tiny CPU fields and counters/hooks already supported by the solver to establish:

```julia
@test direct_phase1 == explicit_homogeneous_smoothing
@test direct_update == normalize(v_old + smoothed_prolongated_correction)
```

The explicit reference sequence must be written in the test in the same order as `ff3d1bf`, including `r = v-Dv`, coarse solve `tol=1e-2`, `maxiter=50`, `restart=20`, one post-SAP sweep, and no odd-even path.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
julia --project=. test/ddalphaamg_setup_modes.jl
```

Expected: direct helper methods are undefined.

- [ ] **Step 3: Implement historical helpers**

Implement homogeneous setup smoothing:

```julia
zero_rhs = similar(test_vectors[1])
zero_wf!(zero_rhs)
for vector in test_vectors
    sap_smooth!(vector, D, zero_rhs, setup;
                niter=setup_sap_iters,
                block_solver=setup_block_solver,
                nblock_mr=setup_block_mr_steps,
                nblock_gmres=setup_block_gmres_steps,
                sap_schedule=:twocolor)
end
```

Implement the adaptive update in the historical order:

```julia
_amg_mul_dirac!(r_fine, D, vector)
axpby!(1, vector, -1, r_fine)
restrict_f2c!(r_coarse, cur_setup, r_fine)
e_coarse = coarse_gmres(cur_setup, D, r_coarse;
                        maxiter=50, restart=20, tol=1e-2,
                        coarse_odd_even=false)
prolongate_c2f!(correction, cur_setup, e_coarse)
sap_smooth!(correction, D, r_fine, cur_setup;
            niter=1, block_solver=:mr, nblock_mr=4,
            sap_schedule=:twocolor, odd_even=false)
axpby!(1, correction, 1, vector)
normalize_fermion!(vector)
```

Use caller-provided setup keyword values where the public API explicitly supplies them; mode defaults fill only omitted mode-dependent values.

- [ ] **Step 4: Dispatch `setup_DDalphaAMG` by mode**

Add `setup_mode::Symbol=:inverse_sap`, validate it before allocations, print it under `verbose=true`, preserve the existing inverse branch verbatim, and invoke the direct helpers only for `:direct_near_null`. Store the selected mode in temporary and final setup objects.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the focused test and require deterministic reference equality within its existing floating-point tolerance.

### Task 3: Mode-Aware Solve Defaults With Explicit Overrides

**Files:**
- Modify: `src/DDalphaAMG.jl`
- Modify: `test/ddalphaamg_setup_modes.jl`

**Interfaces:**
- Produces: `solve_DDalphaAMG` keyword resolution based on `setup.setup_mode`.
- Preserves: explicit values passed by existing experiment scripts.

- [ ] **Step 1: Write failing resolution tests**

Test a small pure resolver helper so no expensive solve is required:

```julia
resolved = DDalphaAMG._amg_resolve_solve_options(:direct_near_null)
@test resolved.restart == 50
@test resolved.smoother == :mr
@test resolved.coarse_odd_even == false

overridden = DDalphaAMG._amg_resolve_solve_options(
    :direct_near_null; restart=25, coarse_odd_even=true)
@test overridden.restart == 25
@test overridden.coarse_odd_even == true
@test overridden.coarse_restart == 20
```

- [ ] **Step 2: Run and verify RED**

Expected: `_amg_resolve_solve_options` is undefined.

- [ ] **Step 3: Implement resolver and nullable public defaults**

Change only mode-dependent public solve keywords to `Union{Nothing,T}=nothing`, resolve them once at entry, print the resolved settings when `verbose=true`, and forward concrete values to `fgmres`.

Keep `tol`, `maxiter`, `x0`, profiling, and `sap_block_mr_steps` independent of setup mode.

- [ ] **Step 4: Run and verify GREEN**

Run focused tests. Also parse representative existing calls to ensure concrete explicit keywords remain accepted.

### Task 4: Regression And GPU Verification

**Files:**
- Modify: `test/ddalphaamg_setup_modes.jl`
- Modify only if needed: `test/runtests.jl`

**Interfaces:**
- Verifies both setup modes without changing production behavior.

- [ ] **Step 1: Run syntax/load verification**

```bash
julia --project=. -e 'using LMP; using LMP.DDalphaAMG'
```

Expected: exit code 0.

- [ ] **Step 2: Run focused setup-mode tests**

```bash
julia --project=. test/ddalphaamg_setup_modes.jl
```

Expected: all tests pass.

- [ ] **Step 3: Run existing CPU regression tests**

Run the DD-alphaAMG tests already tracked by the repository test entrypoint and the existing external-vector smoothing test without adding unrelated untracked files to commits.

- [ ] **Step 4: Run GPU smoke tests**

Print CUDA device and exact tiny-lattice settings, then build one setup and one V-cycle for each mode. Assert GPU-backed test vectors and coarse stencil arrays remain `CuArray`.

- [ ] **Step 5: Compare default inverse mode against explicit inverse mode**

With identical deterministic seeds, assert setup vectors, stencil application, and one V-cycle match between omitted `setup_mode` and `setup_mode=:inverse_sap` to the established machine-precision tolerance.

- [ ] **Step 6: Inspect diff and commit implementation**

```bash
git diff --check
git diff -- src/DDalphaAMG.jl test/ddalphaamg_setup_modes.jl
git status --short
```

Commit only the implementation and its new focused test; leave pre-existing untracked files untouched.
