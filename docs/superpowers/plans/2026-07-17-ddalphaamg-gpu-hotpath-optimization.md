# DD-alphaAMG GPU Hot-Path Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unnecessary full-lattice Wilson applications from GPU `boundary8` SAP and batch coarse Arnoldi GPU reductions while preserving `ComplexF64` numerical behavior.

**Architecture:** Add a compact per-step affected-site map to the existing AMG GPU cache and use a direct residual-update Wilson kernel on those sites. Add a packed GPU Arnoldi basis and two-pass BLAS orthogonalization helper, selected only for CuArray-backed coarse vectors, with existing paths retained as fallbacks.

**Tech Stack:** Julia 1.11, CUDA.jl, LinearAlgebra/cuBLAS, Gaugefields, LatticeDiracOperators, Julia `Test`.

## Global Constraints

- Preserve all public solver signatures and defaults.
- Preserve `ComplexF64` arithmetic.
- Direct operation comparisons must have relative error at most `1e-12`.
- Keep CPU and unsupported-operator fallbacks unchanged.
- Do not alter scientific run parameters or stop existing experiments.
- Print complete settings before any benchmark run.

---

### Task 1: Sparse SAP Residual-Update Contract

**Files:**
- Create: `test/test_sap_boundary8_sparse_residual_gpu.jl`
- Modify: `src/DDalphaAMG.jl`

**Interfaces:**
- Consumes: existing `_amg_mul_dirac!`, block schedule helpers, fused Wilson layout.
- Produces: `_amg_boundary8_residual_update!(res, delta, D, setup, step)::Bool`; returns `true` when the optimized path handled the update and `false` for fallback.

- [ ] **Step 1: Write the failing equivalence test**

For every `step in 0:7`, create a deterministic GPU correction supported only
on that step's blocks. Compute `expected = res - D*delta`, call the wished-for
helper on a copy, and assert:

```julia
handled = DDalphaAMG._amg_boundary8_residual_update!(actual, delta, D, setup, step)
@test handled
@test relative_wf_error(actual, expected) <= 1e-12
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
env -u CUDA_VISIBLE_DEVICES julia --project=. -e 'using CUDA; CUDA.device!(0); include("test/test_sap_boundary8_sparse_residual_gpu.jl")'
```

Expected: failure because `_amg_boundary8_residual_update!` is undefined.

- [ ] **Step 3: Add affected-site cache construction**

Extend `_ensure_gpu_cache!` with eight compact site-index vectors. A site is
included when its coarse block belongs to the selected step or any nearest
neighbor coarse block belongs to that step. Store device vectors under
`:boundary8_affected_sites` without changing existing cache keys.

- [ ] **Step 4: Add the direct residual-update kernel**

Implement a kernel with the interface:

```julia
function _amg_boundary8_residual_update_kernel!(
    res, delta, affected_sites, U1, U2, U3, U4,
    kappa::Float64, blockinfo, total_sites::Int32)
```

Each thread loads one affected site, evaluates the same onsite and eight Wilson
hops as the fused fine operator, including temporal antiperiodic factors, and
subtracts the result from `res`. No writes occur outside the affected list.

- [ ] **Step 5: Dispatch with a safe fallback**

Implement `_amg_boundary8_residual_update!` to require GPU fields,
`_amg_can_fuse_wilson`, and `step in 0:7`. Return `false` otherwise so the
caller can execute the existing full `Ddelta` path.

- [ ] **Step 6: Run the equivalence test and existing SAP tests**

Expected: all eight step comparisons pass at `1e-12`, followed by existing
schedule and odd-even GPU smoke tests passing.

---

### Task 2: Integrate Sparse Update Into GPU SAP

**Files:**
- Modify: `src/DDalphaAMG.jl`
- Create: `test/test_sap_boundary8_fine_apply_count.jl`

**Interfaces:**
- Consumes: `_amg_boundary8_residual_update!` from Task 1.
- Produces: optimized `_sap_smooth_gpu_mr_batched!` and `_sap_smooth_gpu_odd_even_mr_batched!` behavior.

- [ ] **Step 1: Write a failing fine-apply-count test**

Run one deterministic `boundary8` GPU SAP sweep with a `Ref(0)` counter. Assert
the optimized path performs the initial residual application but no per-step
full-lattice applications:

```julia
@test fine_apply_counter[] == 1
```

Also compare the final field against a test helper executing the old full
update path with relative error at most `1e-12`.

- [ ] **Step 2: Run the test and verify RED**

Expected: count is `9`, demonstrating one initial plus eight per-step full
applications.

- [ ] **Step 3: Replace per-step full applications conditionally**

In both GPU MR SAP implementations, use:

```julia
handled = sap_schedule == :boundary8 &&
          _amg_boundary8_residual_update!(res, delta, D, setup, step_i)
if !handled
    _amg_mul_dirac!(Ddelta, D, delta; fine_apply_counter=fine_apply_counter)
    # existing subtraction kernel
end
```

Do not change two-color or CPU behavior.

- [ ] **Step 4: Run count, equivalence, and SAP regression tests**

Expected: one full apply per SAP sweep, numerical equivalence at `1e-12`, and
all existing SAP tests pass.

---

### Task 3: Batched GPU Arnoldi Primitive

**Files:**
- Modify: `src/DDalphaAMG.jl`
- Create: `test/test_coarse_batched_arnoldi_gpu.jl`

**Interfaces:**
- Produces: `_amg_gpu_batched_orthogonalize!(w, basis_matrix, j, coeff_dev, coeff_host)::Float64`.
- Uses two BLAS passes and returns the final norm after updating `w` in place.

- [ ] **Step 1: Write the failing primitive test**

Construct deterministic orthonormal CuArray columns and a deterministic `w`.
Compare the helper with two-pass scalar MGS:

```julia
nrm = DDalphaAMG._amg_gpu_batched_orthogonalize!(
    w_gpu, basis_gpu, j, coeff_dev, coeff_host)
@test norm(Array(w_gpu) - w_reference) / norm(w_reference) <= 1e-12
@test norm(coeff_host[1:j] - h_reference) / norm(h_reference) <= 1e-12
@test abs(nrm - norm(w_reference)) / norm(w_reference) <= 1e-12
```

- [ ] **Step 2: Run the test and verify RED**

Expected: undefined helper failure.

- [ ] **Step 3: Implement packed basis storage and two-pass BLAS**

Use preallocated `CuArray{ComplexF64,2}` and coefficient buffers. For each pass:

```julia
mul!(view(coeff_dev, 1:j), adjoint(view(basis, :, 1:j)), w)
mul!(w, view(basis, :, 1:j), view(coeff_dev, 1:j), -1, 1)
```

Accumulate both coefficient vectors on device, copy the final length-`j`
coefficient vector to host once, and compute the final norm. Do not allocate in
the helper after cache initialization.

- [ ] **Step 4: Run the primitive test and verify GREEN**

Expected: vector, coefficient, norm, and orthogonality checks pass.

---

### Task 4: Integrate Batched Arnoldi Into Coarse GMRES

**Files:**
- Modify: `src/DDalphaAMG.jl`
- Create: `test/test_coarse_gmres_batched_gpu.jl`

**Interfaces:**
- Consumes: Task 3 packed basis and orthogonalization helper.
- Produces: GPU coarse GMRES with one coefficient transfer per Arnoldi step and existing scalar MGS fallback.

- [ ] **Step 1: Write a failing coarse-solve equivalence test**

Extract an internal `_coarse_gmres_impl` interface that accepts an
`arnoldi_mode::Symbol`; keep the public `coarse_gmres` signature unchanged and
have it select `:auto`. On a deterministic coarse stencil system, call the
internal implementation with `:scalar` and `:batched`, then assert solution and
true-residual agreement within `1e-12`.

- [ ] **Step 2: Run the test and verify RED**

Expected: unknown keyword or missing batched mode.

- [ ] **Step 3: Extend coarse GMRES cache**

Allocate packed basis, two device coefficient buffers, and one host coefficient
buffer whenever coarse size or restart changes. Preserve existing `_gmres_V`
objects for operator applications and fallback.

- [ ] **Step 4: Integrate batched orthogonalization**

After each basis vector is created, copy it into the corresponding packed
column. Replace the inner MGS loop only on CuArray-backed coarse vectors. Write
the accumulated host coefficients into `H[1:j,j]`; retain existing Givens and
backsolve code.

- [ ] **Step 5: Run coarse equivalence and odd-even Schur tests**

Expected: scalar and batched solves agree at `1e-12`; existing Schur tests pass.

---

### Task 5: End-To-End Verification And Benchmark

**Files:**
- Create: `test/test_ddalphaamg_gpu_optimized_vcycle.jl`
- Create: `results/diagnostics/ddalphaamg_gpu_hotpath_20260717/settings.txt`
- Create: `results/diagnostics/ddalphaamg_gpu_hotpath_20260717/benchmark.log`

**Interfaces:**
- Verifies Tasks 1-4 together without changing production APIs.

- [ ] **Step 1: Add deterministic V-cycle equivalence test**

Compare an optimized V-cycle with a test-local reference assembled from the
existing full residual update and scalar internal coarse-GMRES mode, using the
same setup and RHS. Require operation-level relative error at most `1e-12`.

- [ ] **Step 2: Run all focused tests**

Run sparse residual, SAP count, batched Arnoldi, coarse GMRES, Schur, schedule,
and GPU smoke tests on an available GPU. Expected: all pass.

- [ ] **Step 3: Print and archive benchmark settings**

Before execution, print gauge ID/path, kappa, lattice, setup source, `Nv`, SAP
schedule and iterations, coarse settings, outer settings, precision, GPU, and
commit/worktree status. Save the same text to `settings.txt`.

- [ ] **Step 4: Run a short before/after benchmark**

Use the existing deterministic notebook-equivalent configuration without
changing its scientific parameters. Limit only the diagnostic outer-step count
equally for both paths. Record wall time, fine/coarse application counts,
profile breakdown, and true residual trajectory.

- [ ] **Step 5: Verify success criteria and report residual risk**

Confirm supported `boundary8` SAP no longer counts per-step full fine applies,
batched coarse GMRES reaches the same true-residual tolerance, and no CPU or
unsupported fallback regressed. Report measured speedup without extrapolation.
