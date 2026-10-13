# Paper-Faithful Shifted GD-lambda-AMG V-cycle Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the coarse-only GD correction preconditioner with the paper's shifted coarse correction plus four-step GMRES post-smoother and verify real correction residual reduction.

**Architecture:** Keep harmonic GD and the left-gamma5 correction equation unchanged. Add a focused shifted two-level preconditioner in `GDlambdaAMG.jl`, use it as the nonstationary right preconditioner inside the existing FGMRES, and expose stage-specific application/residual diagnostics in the experiment wrapper.

**Tech Stack:** Julia 1.11, LinearAlgebra, LMP Wilson fermion fields, CUDA, DDalphaAMG matrix-free coarse solve, Julia `Test`.

## Global Constraints

- Preserve all completed result directories and never overwrite them.
- Do not stop or alter any running experiment.
- Keep kappa `0.157`, configurations `401:410`, 24 vectors, outer tolerance `1e-8`, `m_min=30`, and `m_max=50` unchanged.
- Use correction equation `(D-theta*Gamma5)t=Gamma5*r`.
- Use exactly four GMRES post-smoothing steps, inner relative tolerance `1e-1` with at most five iterations, and coarse relative tolerance `5e-1`.
- Do not silently fall back to coarse-only correction.

---

### Task 1: RED tests for the missing post-smoother

**Files:**
- Modify: `test/gdlambda_amg.jl`
- Test: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: existing `flexible_gmres!` and vector primitives.
- Produces: behavioral contract for `shifted_amg_vcycle!(out, rhs, operator, coarse_correction!; post_smooth_steps=4)` and its diagnostics.

- [ ] Add a deterministic dense fixture with diagonal `A`, an intentionally incomplete coarse correction callback, and a counted operator callback. Assert that the new cycle performs four post-smoothing Krylov steps, returns finite directly recomputed residuals, and improves over coarse-only correction.
- [ ] Add validation tests rejecting `post_smooth_steps != 4`, nonpositive coarse tolerance, and non-finite stage residuals.
- [ ] Run `julia --project=. test/gdlambda_amg.jl` from `/home/choi/LMP/LMP` and verify failure because `shifted_amg_vcycle!` is undefined.
- [ ] Commit only the RED test with message `test: expose missing shifted AMG post-smoother`.

### Task 2: Implement the paper-default shifted two-level cycle

**Files:**
- Modify: `src/GDlambdaAMG.jl`
- Modify: `src/LMP.jl`
- Test: `test/gdlambda_amg.jl`

**Interfaces:**
- Produces `shifted_amg_vcycle!(out, rhs, apply_A!, coarse_correction!; post_smooth_steps::Int=4, drop_tol::Float64=1e-14)`.
- Returns `(; initial_relative_residual, coarse_relative_residual, final_relative_residual, post_iterations, post_applications)`.

- [ ] Implement input validation and zero-output initialization.
- [ ] Apply the supplied shifted coarse correction to `rhs`, recompute `rhs-A*out`, and reject non-finite residuals.
- [ ] Solve the residual equation from zero using exactly four unrestarted GMRES steps with identity right preconditioning, accumulate the delta into `out`, and recompute the final residual directly.
- [ ] Export the new function from `src/LMP.jl`.
- [ ] Run `julia --project=. test/gdlambda_amg.jl` and verify all GD tests pass.
- [ ] Run `julia --project=. test/ddalphaamg_setup_modes.jl` and verify existing setup behavior remains unchanged.
- [ ] Commit implementation and tests with message `fix: restore shifted AMG GMRES post-smoothing`.

### Task 3: Route the experiment correction solver through the full cycle

**Files:**
- Modify: `/home/choi/LMP/scripts/experiments/evaluate_gdlambda_amg_learned24_vs_random24.jl`
- Modify: `test/gdlambda_amg.jl`

**Interfaces:**
- Consumes: `shifted_amg_vcycle!`, `shifted_coarse_correction!`, `flexible_gmres!`.
- Produces: FGMRES right preconditioner containing shifted coarse correction and four-step GMRES post-smoothing.

- [ ] Add RED source/integration assertions proving the wrapper calls `shifted_amg_vcycle!`, fixes post smoothing to four, and records smoother applications and per-cycle residuals.
- [ ] Run the focused test and verify it fails against the current coarse-only wrapper.
- [ ] Replace `apply_M!` in `gd_correction_solver` with the full shifted cycle. Keep outer FGMRES tolerance `1e-1`, maximum five, and coarse tolerance `5e-1` supplied by the existing required environment values, while asserting those values equal the paper defaults for the paper-faithful runner.
- [ ] Extend history and summary schemas with post-smoother applications, worst/last preconditioner residual ratio, and explicit `preconditioner=shifted_coarse_plus_gmres4` metadata.
- [ ] Ensure all output paths still refuse overwrite.
- [ ] Run the focused test and verify it passes.
- [ ] Commit with message `fix: use full shifted AMG cycle in GD correction`.

### Task 4: CPU regression and numerical identity verification

**Files:**
- Test: `test/gdlambda_amg.jl`
- Test: `test/ddalphaamg_setup_modes.jl`
- Test: `test/coarse_correction_loss.jl` when present.

**Interfaces:**
- Verifies: shifted operator identity, correction residual definition, counters, GD restart, and coarse solve compatibility.

- [ ] Run `julia --project=. test/gdlambda_amg.jl` and record the complete pass count.
- [ ] Run `julia --project=. test/ddalphaamg_setup_modes.jl` and record the complete pass count.
- [ ] Locate and run the existing coarse-correction regression entry point; if no standalone file exists, run its containing test suite and record the exact command.
- [ ] Run `git diff --check` and inspect `git diff -- src/GDlambdaAMG.jl src/LMP.jl test/gdlambda_amg.jl` plus the external wrapper diff.

### Task 5: GPU1 configuration-401 smoke verification

**Files:**
- Create results only under: `/home/choi/LMP/results/experiments/gdlambda_amg_paper_vcycle_smoke_cfg401_gpu1_20260824`

**Interfaces:**
- Consumes: corrected wrapper and the same checkpoint/setup/configuration used by the failed experiment.
- Produces: non-overwriting smoke summary/history/settings and logs.

- [ ] Confirm GPU1 is available without touching another process; if occupied, report and wait rather than moving the run.
- [ ] Launch only configuration 401 with CUDA device 1 and paper defaults: outer `1e-8`, inner `1e-1`/5, coarse `5e-1`, post smoother 4, `m_min=30`, `m_max=50`.
- [ ] Preserve process PID, command, settings, and log in the new result directory.
- [ ] After completion, verify that every preconditioner call records four post-smoothing steps, all residuals are finite, and the correction residual ratio is materially below one.
- [ ] Compare the corrected outer residual trajectory with the preserved coarse-only config-401 trajectory; do not launch configurations 402--410 without a separate explicit user decision.

### Task 6: Final verification report

**Files:**
- Verify all modified files and new smoke artifacts.

**Interfaces:**
- Produces: evidence-backed statement of whether the implementation defect is fixed and whether the corrected solver improves convergence.

- [ ] Run fresh focused CPU regressions and inspect exit codes immediately before reporting.
- [ ] Confirm no prior result files changed using timestamps and paths.
- [ ] Report code changes, exact test counts, smoke residual ratios, outer residual trajectory, and remaining differences from full multi-eigenpair GD-lambda-AMG (locking and dynamic interpolation updates).
