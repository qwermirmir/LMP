# Ordered corner33 LMP Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the approved ordered corner33 path variant, verify its three-layer Ritz gradient, and start a 4,000-step matched-seed training run on GPU1.

**Architecture:** Minimally extend the existing `_default_lmp_paths` factory and reuse `train_lmp.jl` via a dedicated launcher. Keep diagnostics separate from production code. Gate the full run on exact path, parameter-count, checkpoint-mismatch, CPU centered-difference, and GPU checkpoint round-trip checks.

**Tech Stack:** Julia 1.11, LMP, JLD2, CUDA, Gaugefields, LatticeDiracOperators.

## Global Constraints

- Path order is identity, eight signed one-hop paths, then four sign combinations for each plane `μ < ν`.
- Keep 33 path slots, 32 PT layers, 24 output vectors, BlockScalar `r=2`, `κ=0.158`, batch 1, Adam learning rate 0.001, and 4,000 steps.
- Use `INIT_SEED=20260918`, configuration-selection seed 42, the established 320 train/20 test split, and Ritz-only loss.
- Do not change existing path variants, DDalphaAMG, or any running experiment.
- Use distinct smoke/full output paths and refuse overwrites.

---

### Task 1: Exact ordered corner33 variant

**Files:**
- Modify: `src/LMP.jl:135`
- Create: `test/corner33_ordered_paths.jl`

**Interfaces:**
- Consumes: `LMP._default_lmp_paths(max_len; path_variant)`.
- Produces: `path_variant=:corner33_ordered` with exactly 33 unique `Paths`.

- [ ] **Step 1: Write the failing path/checkpoint test.** The expected list is:

```julia
expected = vcat(
    [[0]],
    [[d] for d in (1, 2, 3, 4, -1, -2, -3, -4)],
    [[sμ * μ, sν * ν] for (μ, ν) in ((1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
                       for (sμ, sν) in ((1,1), (1,-1), (-1,1), (-1,-1))],
)
```

The test must verify exact order, 33 unique paths, 1089 PT weights for `Out=33`, saved path metadata, and mismatch rejection against short33.

- [ ] **Step 2: Run RED.** Execute `julia --project=. --startup-file=no test/corner33_ordered_paths.jl`; expect `Unknown path variant: corner33_ordered`.
- [ ] **Step 3: Implement the minimal factory branch:** identity, signed one-hop loop, then plane/sign loops matching the expected order.
- [ ] **Step 4: Run GREEN and regressions.** Execute the new test plus `test/short33_paths.jl` and `test/long_only2468_paths.jl`; all must pass.

### Task 2: Three-layer CPU gradient diagnostic

**Files:**
- Create: `scripts/diagnostics/check_corner33_three_layer_gradient.jl`

**Interfaces:**
- Consumes: `:short_repeat4` and `:corner33_ordered` path lists.
- Produces: per-layer centered-difference versus analytical Ritz gradient records and exits nonzero when tolerances fail.

- [ ] **Step 1: Reuse the already validated `4^4` CPU pattern:** fixed seed, identical input fields and initial weights, PT stack `33→33→33→2`, and `Ritz_Layer`.
- [ ] **Step 2: Keep the CPU union-buffer adapter local to the diagnostic process; do not modify runtime training code.**
- [ ] **Step 3: For each of three PT layers, check real and imaginary centered differences at representative one-hop and corner slots with `h=1e-4`; compare the imaginary derivative with `-imag(gradient)` according to the code's complex-gradient convention.**
- [ ] **Step 4: Require maximum relative error ≤ `1e-5` when derivative magnitude is ≥ `1e-8`, print a compact summary, and run the script to completion.

### Task 3: Smoke/full launcher

**Files:**
- Create: `scripts/training/run_train_lmp_corner33_ordered_k158_24eig_r2_4000_20260923.jl`

**Interfaces:**
- Consumes: `train_lmp.jl` and `:corner33_ordered`.
- Produces: separate smoke and full checkpoint directories.

- [ ] **Step 1: Copy the known-good short33 launcher and change only the smoke variable, output paths, path variant, checkpoint naming, and pass label.** Use:
  - `CORNER33_SMOKE`
  - `/data1/choi/LMP_corner33_ordered_k158_24eig_r2_smoke_20260923`
  - `/data1/choi/LMP_corner33_ordered_k158_24eig_r2_4000_20260923`
  - prefix `(16x32)32PTlayer33cornerpath(kappa=0.158)`
  - suffix `(corner33_ordered)(maxlen2)(xavier_init)(24eig)(320config)(blockscalar_r=2)(adamfix)`
- [ ] **Step 2: Add smoke assertions for 34 weight arrays, finite and changed weights, optimizer step 2, exact PT path metadata on layers 3–18 and 20–35, and loss equality after reload.**
- [ ] **Step 3: Parse the launcher with `Meta.parseall`.**
- [ ] **Step 4: Confirm GPU1 and both output targets are available, then run two-step smoke on GPU1. Require `CORNER33_SMOKE_PASS` and exit code 0.

### Task 4: Start and verify full training

**Files:**
- Runtime: `/data1/choi/LMP_corner33_ordered_k158_24eig_r2_4000_20260923/`
- Runtime: `results/run/train_corner33_ordered_20260923_gpu1.launch.log`

**Interfaces:**
- Consumes: verified launcher and GPU1.
- Produces: live 4,000-step experiment with checkpoints at 0/1000/2000/3000/4000 and final.

- [ ] **Step 1: Recheck GPU1, process list, target nonexistence, and tmux-session nonexistence without stopping other work.**
- [ ] **Step 2: Start the full launcher in `tmux` session `train_corner33_ordered_20260923_gpu1`, redirecting stdout/stderr to the dedicated launch log.**
- [ ] **Step 3: Verify the process is alive, the metadata records the exact corner path list and requested settings, the initial checkpoint exists, and at least one training step completes.**
- [ ] **Step 4: Report current step and all log/checkpoint paths; do not claim the run is complete until step 4000 and final save are present.**
