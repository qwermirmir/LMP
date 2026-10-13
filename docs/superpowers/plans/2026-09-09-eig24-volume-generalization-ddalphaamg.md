# eig24 Volume-Generalization DD-alphaAMG Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare corrected-Adam BlockScalar eig24, corrected-Adam PT-replacement eig24, and V-cycle Nv=24 on beta-6.2 `24^3 x 48` config 0 at kappa 0.1527 and 0.1535.

**Architecture:** Create one standalone Julia experiment by preserving the established DD-alphaAMG setup/solve/count path and reusing the validated non-fermion checkpoint filter. Build each learned architecture exactly once per case, generate vectors with cache eviction, run all cases sequentially, and write incremental logs/CSV plus a two-panel plot.

**Tech Stack:** Julia 1.11.9, CUDA.jl, JLD2, Gaugefields, LatticeDiracOperators, LMP, Plots, Test, tmux.

## Global Constraints

- Gauge is config 0 from `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/raw_gauge`.
- Lattice is `24 x 24 x 24 x 48`; kappas are exactly 0.1527 and 0.1535.
- Compare exactly BlockScalar eig24, PT-replacement eig24, and V-cycle Nv=24.
- Use random fermion seed 20260831 identically for both learned models.
- Use `block_r=6` for the transferred BlockScalar model.
- Preserve the approved DD-alphaAMG and solver settings, including tolerance `1e-10` and maxiter 200.
- Warm all three complete paths at kappa 0.1527 and discard warm-up measurements.
- Refuse to overwrite outputs, impose no runtime timeout, and never alter a running case.
- Preserve all partial rows and logs after failure.

---

### Task 1: Define executable source contracts before implementation

**Files:**
- Create: `/home/choi/LMP/test/eig24_volume_generalization_ddalphaamg_script.jl`
- Create later: `/home/choi/LMP/scripts/experiments/compare_eig24_blockscalar_vs_plus1pt_vs_vcycle_24cube48time.jl`

**Interfaces:**
- Consumes: the approved design constants and existing LMP functions.
- Produces: static tests that fail while the evaluator is absent and later verify exact scientific settings and reuse points.

- [ ] **Step 1: Add a test that requires the evaluator and fixed settings**

The test must assert the evaluator exists and contains exact gauge/checkpoint
paths, lattice constants, `KAPPA_LIST=[0.1527,0.1535]`, `NV=24`,
`BLOCK_SCALAR_R=6`, random seed 20260831, the three case identifiers, solver
settings, per-case warm-up, non-overwrite guard, derived-checkpoint validation,
cache eviction, six-row validation, and a library-only main guard.

- [ ] **Step 2: Run the test and require one missing-evaluator failure**

```bash
julia --project=/home/choi/LMP/LMP \
  /home/choi/LMP/test/eig24_volume_generalization_ddalphaamg_script.jl
```

Expected: failure only because the evaluator file does not exist.

### Task 2: Implement the standalone evaluator from known-success paths

**Files:**
- Create: `/home/choi/LMP/scripts/experiments/compare_eig24_blockscalar_vs_plus1pt_vs_vcycle_24cube48time.jl`
- Test: `/home/choi/LMP/test/eig24_volume_generalization_ddalphaamg_script.jl`

**Interfaces:**
- Consumes: the two final corrected-Adam checkpoints, config 0, existing layer constructors, `setup_DDalphaAMG`, `solve_DDalphaAMG`, and cache-clear helpers.
- Produces: `main_eig24_volume_generalization_ddalphaamg()` and a non-overwriting result directory.

- [ ] **Step 1: Implement fixed configuration and checkpoint filtering**

Define exact constants from the design. Copy the validated
`is_fermion_checkpoint_key`, `create_nonfermion_checkpoint`, and
`validate_nonfermion_checkpoint` behavior. Require both source checkpoints to
contain `opt_t=4000`, and write source/derived SHA-256 provenance.

- [ ] **Step 2: Implement exact architecture construction and vector inference**

Use `LMP.build_lmp_32pt_layers(...; max_len=5, num_eig=24, block_r=6,
init_gpu=false)` for BlockScalar. Build the PT replacement with the established
38-layer layout and no BlockScalar. Load derived checkpoints through
`load_checkpoint!`, compare loaded non-fermion weights, randomize the single
fermion layer with seed 20260831, and forward through existing layers while
clearing consumed caches.

- [ ] **Step 3: Copy the successful DD-alphaAMG setup and solve calls**

Use `setup_mode=:direct_near_null`, learned MR smoothing 16, SAP setup 3,
block-MR 4, V-cycle setup 6, post-SAP 1, solve MR 4/4, tolerance `1e-10`,
maxiter 200, restart 50, and the approved coarse settings. Recreate the seeded
normalized RHS before every case and compute relative residual independently.

- [ ] **Step 4: Implement incremental output, plotting, and validation**

Write one CSV row and one case log immediately after each measured case. Plot
three distinct method markers in two log-log panels and save PNG/PDF. Require
six unique rows, finite positive counts/times, valid paths, and matching random
field digests for the two learned methods.

- [ ] **Step 5: Run source tests and parse checks**

```bash
julia --project=/home/choi/LMP/LMP \
  /home/choi/LMP/test/eig24_volume_generalization_ddalphaamg_script.jl
julia --project=/home/choi/LMP/LMP -e \
  'Meta.parseall(read(ARGS[1], String)); println("parse=PASS")' \
  /home/choi/LMP/scripts/experiments/compare_eig24_blockscalar_vs_plus1pt_vs_vcycle_24cube48time.jl
```

Expected: every test passes and the evaluator parses without executing.

### Task 3: Run a non-mutating preflight and launch production

**Files:**
- Consume: evaluator from Task 2.
- Produce: result directory, launcher log, CSV, derived checkpoints, and case logs.

**Interfaces:**
- Consumes: one free GPU and absent output/session names.
- Produces: a persistent tmux run on GPU 0.

- [ ] **Step 1: Verify source artifacts and GPU availability**

Require both final checkpoints, `opt_t=4000`, the exact config-0 raw gauge,
absent result root, absent launcher log, absent tmux session, and no compute
process on GPU 0. Do not disturb GPU 1.

- [ ] **Step 2: Launch in tmux without a timeout**

Run the evaluator with `CUDA_VISIBLE_DEVICES=0` in session
`eig24_volume_ddamg_cfg0_gpu0_20260909`. Redirect stdout/stderr to a dedicated
launcher log outside the not-yet-created result root.

- [ ] **Step 3: Validate live startup**

Require the Julia process and tmux session to remain alive, GPU 0 allocation,
the exact settings record, both derived checkpoints, identical random-field
digests, and successful entry into the first warm-up. Do not stop or replace
the run if it is slow.

### Task 4: Validate natural completion

**Files:**
- Inspect: result CSV, settings, provenance, logs, PNG, PDF, and validation report.

**Interfaces:**
- Consumes: a naturally completed production run.
- Produces: verified six-point solver comparison ready for interpretation.

- [ ] **Step 1: Require natural process termination and exactly six rows**

Require no matching process/session and each `(kappa,method)` key exactly once.

- [ ] **Step 2: Independently verify measurements and artifacts**

Require positive fine/coarse counts, finite setup/solve times and residuals,
correct convergence labels, exact gauge/checkpoint hashes, paired field digest,
readable PNG/PDF, and all internal validation entries marked PASS.

- [ ] **Step 3: Report individual results without changing failed cases**

Report fine/coarse counts, solve time, and residual for all six points. If a
case did not converge, report it as measured and retain its row; do not rerun
with changed settings unless the user explicitly requests a new experiment.
