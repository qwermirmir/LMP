# GD-lambda-AMG DDalphaAMG Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore DDalphaAMG to repository `HEAD` while preserving the current GD-lambda-AMG adaptive setup and verified shifted coarse-solve behavior in GD-owned code.

**Architecture:** Add `GDlambdaAMGDDalphaAdapter.jl` as a namespaced adapter that consumes unchanged DDalphaAMG structures and operations. Move GD-only prototype refinement, true-residual coarse GMRES, and Schur verification into the adapter, then change `GDlambdaAMG.jl` and the experiment driver to call it.

**Tech Stack:** Julia 1.11, LMP, DDalphaAMG Julia module, CUDA.jl, Test, Git.

## Global Constraints

- `src/DDalphaAMG.jl` must be byte-for-byte identical to `HEAD`.
- `test/ddalphaamg_setup_modes.jl` must be byte-for-byte identical to `HEAD`.
- Do not modify either external DDalphaAMG repository.
- Do not change kappa, configuration IDs, tolerances, vector counts, checkpoints, random seeds, setup stages, adaptive cycles, or solver limits.
- Do not overwrite or delete existing experiment outputs.
- All GD-only functions use a `gd_` prefix and live outside `DDalphaAMG.jl`.

---

### Task 1: Lock the isolation boundary with failing tests

**Files:**
- Modify: `test/gdlambda_amg.jl`
- Test: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: current `LMP` module and source paths.
- Produces: failing expectations for `gd_global_orthonormalize!`, `gd_adaptive_amg_setup`, `gd_coarse_gmres`, and source isolation.

- [ ] **Step 1: Add source-boundary and API tests**

```julia
const GD_ADAPTER_SRC = normpath(joinpath(
    @__DIR__, "..", "src", "GDlambdaAMGDDalphaAdapter.jl"))

@testset "GD DDalpha adapter owns experiment-only behavior" begin
    @test isfile(GD_ADAPTER_SRC)
    @test isdefined(LMP, :gd_global_orthonormalize!)
    @test isdefined(LMP, :gd_adaptive_amg_setup)
    @test isdefined(LMP, :gd_coarse_gmres)
    ddalpha = read(joinpath(@__DIR__, "..", "src", "DDalphaAMG.jl"), String)
    @test !occursin("raw_test_vectors_out", ddalpha)
    @test !occursin("_amg_global_orthonormalize!", ddalpha)
    @test !occursin("coarse_odd_even_relative_residual", ddalpha)
end
```

- [ ] **Step 2: Run the test and verify RED**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: FAIL because the adapter file and `gd_` APIs do not exist and the current DDalphaAMG source still contains GD-only symbols.

- [ ] **Step 3: Commit only the failing test**

Run:

```bash
git add test/gdlambda_amg.jl
git commit -m "test: require GD DDalphaAMG isolation"
```

---

### Task 2: Add the GD-owned adapter primitives

**Files:**
- Create: `src/GDlambdaAMGDDalphaAdapter.jl`
- Modify: `src/LMP.jl`
- Modify: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: `DDalphaAMG.CoarseVector`, `DDalphaAMG.CoarseStencilOperator`, Wilson-field `inner_product`, and field `axpby!`.
- Produces: `gd_global_orthonormalize!(vectors; drop_tol=1e-14)` and `gd_coarse_odd_even_relative_residual(op, solution, rhs)`.

- [ ] **Step 1: Add behavioral tests for orthonormalization and Schur residual**

```julia
@testset "GD adapter globally orthonormalizes prototype fields" begin
    template, _ = paper_identity_cpu_problem()
    vectors = [similar(template) for _ in 1:3]
    foreach(LMP.DDalphaAMG.random_fermion!, vectors)
    axpby!(1, vectors[1], 1, vectors[2])
    LMP.gd_global_orthonormalize!(vectors)
    gram = [inner_product(vectors[i], vectors[j]) for i in 1:3, j in 1:3]
    @test gram ≈ Matrix{ComplexF64}(I, 3, 3) atol=1e-11
end
```

- [ ] **Step 2: Run the test and verify RED**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: FAIL with `UndefVarError` for the new GD adapter functions.

- [ ] **Step 3: Add and include the adapter**

`src/LMP.jl` includes the adapter after `DDalphaAMG.jl` and before `GDlambdaAMG.jl`:

```julia
include("DDalphaAMG.jl")
include("GDlambdaAMGDDalphaAdapter.jl")
include("CoarseCorrectionLoss.jl")
include("GDlambdaAMG.jl")
```

The adapter implements two-pass modified Gram-Schmidt with rank and finite-value checks, and computes the Schur residual through the unchanged DDalphaAMG odd-even helper operations.

- [ ] **Step 4: Run the focused test and verify GREEN**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: the new primitive tests pass; the source-isolation assertions remain RED until Task 5.

- [ ] **Step 5: Commit the adapter primitives**

Run:

```bash
git add src/GDlambdaAMGDDalphaAdapter.jl src/LMP.jl test/gdlambda_amg.jl
git commit -m "feat: add GD-owned DDalphaAMG adapter primitives"
```

---

### Task 3: Move verified shifted coarse GMRES into the adapter

**Files:**
- Modify: `src/GDlambdaAMGDDalphaAdapter.jl`
- Modify: `src/GDlambdaAMG.jl`
- Modify: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: unchanged DDalphaAMG coarse caches, stencil application, odd-even reduction, reconstruction, norms, and axpby operations.
- Produces: `gd_coarse_gmres(setup, D, rhs; maxiter, restart, tol, coarse_odd_even, arnoldi_mode, coarse_apply_counter)`.

- [ ] **Step 1: Change the shifted coarse test to require `gd_coarse_gmres`**

```julia
computed = try
    setup.coarse_stencil = shifted
    LMP.gd_coarse_gmres(setup, nothing, shifted_rhs;
        tol=1e-10, maxiter=10, restart=100,
        coarse_odd_even=false, arnoldi_mode=:scalar)
finally
    setup.coarse_stencil = original
end
```

- [ ] **Step 2: Run the test and verify RED**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: FAIL because `gd_coarse_gmres` has not yet been implemented.

- [ ] **Step 3: Implement GD-local restarted GMRES**

Implement the existing corrected algorithm under the `gd_coarse_gmres` name: preserve odd-even reduction, scalar/batched Arnoldi selection, `eps(Float64)` happy-breakdown criterion, direct true-residual verification, full-solution reconstruction, finite checks, and application accounting. All DDalphaAMG helper calls are explicitly qualified; no DDalphaAMG method is redefined.

- [ ] **Step 4: Switch `shifted_coarse_correction!` to GD-local functions**

```julia
coarse_solution = gd_coarse_gmres(
    setup, operator, coarse_rhs;
    tol=coarse_tol, maxiter=coarse_maxiter, restart=coarse_restart,
    coarse_odd_even=true, arnoldi_mode=:scalar,
    coarse_apply_counter=coarse_apply_counter)

schur_relative_residual = gd_coarse_odd_even_relative_residual(
    shifted_stencil, coarse_solution, coarse_rhs)
```

- [ ] **Step 5: Run GD tests and verify GREEN**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: shifted coarse solve and residual tests pass.

- [ ] **Step 6: Commit the coarse-solve isolation**

Run:

```bash
git add src/GDlambdaAMGDDalphaAdapter.jl src/GDlambdaAMG.jl test/gdlambda_amg.jl
git commit -m "refactor: isolate GD shifted coarse GMRES"
```

---

### Task 4: Move adaptive prototype refinement into the adapter

**Files:**
- Modify: `src/GDlambdaAMGDDalphaAdapter.jl`
- Modify: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: `D`, a Wilson-field template, block size `r`, prototype count `Nv`, optional `prototypes`, and the production setup keywords.
- Produces: `gd_adaptive_amg_setup(D, template, r, Nv; prototypes=nothing, ...) -> (; setup, prototypes)`.

- [ ] **Step 1: Add a seeded source-equivalence test**

The test creates a small CPU Wilson problem, seeds `Random`, obtains one result from `prototypes=nothing`, recreates the same random fields under the same seed, obtains the second result from `prototypes=supplied`, and asserts all returned refined fields differ by less than `1e-11`.

- [ ] **Step 2: Run the test and verify RED**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: FAIL because `gd_adaptive_amg_setup` does not exist.

- [ ] **Step 3: Implement the common GD setup pipeline**

Implement one path for both prototype sources: clone or generate fields, perform the requested inverse-SAP initial stages, globally orthonormalize before every adaptive cycle, build the temporary chiral hierarchy, apply the unchanged DDalphaAMG V-cycle update with identical keywords, and return unsplit refined fields. Build the final hierarchy with original `setup_DDalphaAMG(...; coarse_vectors=refined, smooth_external_vectors=false, smoother=:mr)` so the core function performs only final chiral splitting, block orthonormalization, and coarse construction.

- [ ] **Step 4: Run the equivalence and rank-rejection tests**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: seeded random and supplied paths agree within `1e-11`; dependent fields throw `ArgumentError` naming the column.

- [ ] **Step 5: Commit the adaptive setup adapter**

Run:

```bash
git add src/GDlambdaAMGDDalphaAdapter.jl test/gdlambda_amg.jl
git commit -m "feat: isolate GD adaptive AMG setup"
```

---

### Task 5: Restore DDalphaAMG and switch the experiment driver

**Files:**
- Restore: `src/DDalphaAMG.jl`
- Restore: `test/ddalphaamg_setup_modes.jl`
- Modify: `scripts/experiments/evaluate_gdlambda_amg_learned24_vs_random24.jl`
- Modify: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: `gd_adaptive_amg_setup(...).setup` and `.prototypes`.
- Produces: unchanged standard and learned experiment rows without `raw_test_vectors_out` or altered DDalphaAMG behavior.

- [ ] **Step 1: Switch both production and warm-up setup calls**

```julia
standard = gd_adaptive_amg_setup(
    D, template, GD_AMG_BLOCK_R, GD_AMG_NV;
    nsmooth=GD_AMG_NSMOOTH, nvcycle=GD_AMG_NVCYCLE,
    setup_smoother=:sap, setup_sap_iters=3,
    setup_block_solver=:mr, setup_block_mr_steps=4,
    vcycle_post_sap_iters=GD_POST_SMOOTH,
    use_chirality=true, verbose=false)

learned_refined = gd_adaptive_amg_setup(
    D, template, GD_AMG_BLOCK_R, GD_AMG_NV;
    prototypes=learned,
    nsmooth=GD_AMG_NSMOOTH, nvcycle=GD_AMG_NVCYCLE,
    setup_smoother=:sap, setup_sap_iters=3,
    setup_block_solver=:mr, setup_block_mr_steps=4,
    vcycle_post_sap_iters=GD_POST_SMOOTH,
    use_chirality=true, verbose=false)
```

Pass `standard.setup`/`standard.prototypes` and `learned_refined.setup`/`learned_refined.prototypes` to `gd_run_case!`.

- [ ] **Step 2: Restore only the two approved tracked DDalphaAMG files**

Run:

```bash
git restore --source=HEAD -- src/DDalphaAMG.jl test/ddalphaamg_setup_modes.jl
```

- [ ] **Step 3: Verify the isolation boundary**

Run:

```bash
git diff --exit-code HEAD -- src/DDalphaAMG.jl
git diff --exit-code HEAD -- test/ddalphaamg_setup_modes.jl
```

Expected: both commands exit 0 with no output.

- [ ] **Step 4: Run DDalphaAMG and GD tests**

Run:

```bash
julia --project=. test/ddalphaamg_setup_modes.jl
julia --project=. test/gdlambda_amg.jl
```

Expected: all tests pass, and the experiment source contains `gd_adaptive_amg_setup` but not `raw_test_vectors_out`.

- [ ] **Step 5: Commit the restoration and driver migration**

Run:

```bash
git add src/DDalphaAMG.jl test/ddalphaamg_setup_modes.jl \
  test/gdlambda_amg.jl
git commit -m "fix: restore DDalphaAMG and isolate GD integration"
```

The experiment driver is outside the nested `LMP` Git repository. Verify its
diff separately and leave it in the shared workspace; do not attempt to stage
it through the nested repository.

---

### Task 6: Full verification and GPU smoke

**Files:**
- Read: `src/DDalphaAMG.jl`
- Read: `src/GDlambdaAMGDDalphaAdapter.jl`
- Read: `src/GDlambdaAMG.jl`
- Read: `scripts/experiments/evaluate_gdlambda_amg_learned24_vs_random24.jl`
- Create: a new timestamped smoke result directory under `results/experiments/`

**Interfaces:**
- Consumes: completed isolated implementation.
- Produces: CPU verification output and a non-overwriting GPU smoke result.

- [ ] **Step 1: Run source and whitespace checks**

Run:

```bash
git diff --check
git diff --exit-code HEAD -- src/DDalphaAMG.jl test/ddalphaamg_setup_modes.jl
grep -nE 'raw_test_vectors_out|_amg_global_orthonormalize!|coarse_odd_even_relative_residual' src/DDalphaAMG.jl
```

Expected: no diff/check errors; grep produces no output.

- [ ] **Step 2: Run CPU regression tests**

Run:

```bash
julia --project=. test/ddalphaamg_setup_modes.jl
julia --project=. test/gdlambda_amg.jl
```

Expected: all tests pass.

- [ ] **Step 3: Run one-configuration, one-expansion GPU smoke**

Use GPU1 and a new timestamped result directory. Preserve production kappa, checkpoint, 24 prototypes, setup stages, six adaptive cycles, smoother settings, and solver tolerances; set only `GD_CONFIG_IDS=401` and `GD_MAX_EXPANSIONS=1` for the smoke scope.

Expected: exactly two summary rows, one standard and one learned, with finite diagnostics and no unexpected exception.

- [ ] **Step 4: Perform final code review**

Confirm no DDalphaAMG source/test diff, no experiment-only keyword dependency, no mutation of caller-owned model vectors, and no accidental changes outside the approved files.

- [ ] **Step 5: Report evidence**

Report exact test counts, the two zero-diff commands, smoke result path, and any remaining scientific limitations without claiming equivalence beyond the verified checks.
