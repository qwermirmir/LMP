# Eig1 Volume-Generalization Ritz-Loss Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare configuration-wise eig1 Ritz test loss for the archived BlockScalar and no-BlockScalar-plus-PT models on all 30 beta 6.2, `24^3 x 48` gauges using one identical random fermion field.

**Architecture:** Add one experiment script that filters dimension-bound fermion keys into result-local checkpoints, loads all retained parameters with the existing loader, and evaluates the existing Ritz layer through the existing cache-clearing forward path. Run a one-gauge smoke for each model before the full 30-gauge sequential evaluation and paired-point plot.

**Tech Stack:** Julia 1.11.9, LMP, Gaugefields, LatticeDiracOperators, CUDA, JLD2, TimerOutputs, SHA, Plots.

## Global Constraints

- Compare only the two archived eig1 16,000-step checkpoints named in the design.
- Use gauges `config_0.gbin` through `config_29.gbin` from `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/raw_gauge`.
- Use lattice `24 x 24 x 24 x 48`, beta `6.2`, Wilson kappa `0.1535`, and CUDA blocks `(4,4,4,4)`.
- Use CPU and CUDA random seed `20260831`; reuse an identical random fermion payload across models and configurations.
- Use `block_r=6` for the BlockScalar model and require exact `(512,1)` checkpoint/target equality.
- Report only Ritz test loss as the scientific metric; do not run DD-alphaAMG in this experiment.
- Plot all 30 paired individual points without mean, standard deviation, or error bars.
- Do not modify core LMP code or archived checkpoints.
- Never change conditions after launch or delete partial intentional output without explicit user instruction.

---

### Task 1: Checkpoint Filtering and Source Contracts

**Files:**
- Create: `/home/choi/LMP/scripts/experiments/evaluate_eig1_volume_generalization_ritz_loss.jl`
- Create: `/home/choi/LMP/test/eig1_volume_generalization_ritz_loss_script.jl`

**Interfaces:**
- Produces `is_fermion_checkpoint_key(::AbstractString)::Bool`.
- Produces `create_nonfermion_checkpoint(source::String, destination::String)::Vector{String}`.
- Produces `validate_nonfermion_checkpoint(source::String, derived::String, removed::Vector{String})::Nothing`.

- [ ] **Step 1: Write failing filtering tests**

The test sets `ENV["EIG1_VOLUME_LOSS_LIBRARY_ONLY"]="1"`, includes the experiment script, and asserts:

```julia
@test is_fermion_checkpoint_key("layer_1_fermion_x_1")
@test is_fermion_checkpoint_key("layer_1_fermion_m_1")
@test is_fermion_checkpoint_key("layer_1_fermion_v_1")
@test is_fermion_checkpoint_key("layer_1_fermion_blocks_1")
@test !is_fermion_checkpoint_key("layer_2_W")
```

In a temporary JLD2 source containing those keys, `layer_2_W`, and `opt_t`, require the derived file to omit exactly the four fermion keys while retaining byte-equivalent values for the other keys. Require a second call to refuse overwrite.

- [ ] **Step 2: Run RED**

Run:

```bash
julia --project=/home/choi/LMP/LMP \
  /home/choi/LMP/test/eig1_volume_generalization_ritz_loss_script.jl
```

Expected: failure because the filtering functions and script do not exist.

- [ ] **Step 3: Implement minimal checkpoint filtering**

Use the exact key predicate:

```julia
is_fermion_checkpoint_key(key::AbstractString) =
    occursin(r"^layer_[0-9]+_fermion_(x|m|v|blocks)_[0-9]+$", key)
```

`create_nonfermion_checkpoint` must refuse an existing destination, iterate source keys with `JLD2.jldopen`, copy every non-fermion value unchanged, and return the sorted removed-key list. `validate_nonfermion_checkpoint` must require no fermion key, identical retained key sets, and `isequal` values for all retained keys.

- [ ] **Step 4: Add static source contracts**

Require the experiment source to contain exact checkpoint paths, lattice constants, `KAPPA=0.1535`, `RANDOM_SEED=20260831`, `BLOCK_R=6`, configuration IDs `0:29`, the new raw-gauge directory, `_checkpointed_forward_loss!`, and the library-only main guard.

- [ ] **Step 5: Run GREEN**

Require all filtering and static tests to pass and parse the experiment script with `Meta.parseall`.

### Task 2: Existing Architecture and Streaming-Loss Orchestration

**Files:**
- Modify: `/home/choi/LMP/scripts/experiments/evaluate_eig1_volume_generalization_ritz_loss.jl`
- Modify: `/home/choi/LMP/test/eig1_volume_generalization_ritz_loss_script.jl`

**Interfaces:**
- Produces `make_plus1pt_layers(U, x, D)` matching the archived 38-layer stack.
- Produces `make_model_layers(U, x, D, variant::Symbol)`.
- Produces `randomize_fermion_field!(layers, seed::Int)` and `fermion_digest(layers)::String`.
- Produces `streaming_ritz_loss!(layers, D)::Float64`.
- Produces `assert_loaded_checkpoint_weights!(layers, derived_path)::Nothing`.

- [ ] **Step 1: Write failing orchestration contracts**

Add source and lightweight unit assertions requiring both variants, a 38-layer architecture check, identical field-digest checks, exact BlockScalar shape `(512,1)`, retained-weight equality, and use of `_clear_forward_caches!` after streaming evaluation.

- [ ] **Step 2: Run RED**

Expected: failure because orchestration functions are absent.

- [ ] **Step 3: Reuse the archived eig1 constructors**

Copy only `make_plus1pt_layers` from `evaluate_eig1_absolute_residual_blockscalar_vs_plus1pt.jl`. Build the BlockScalar model through:

```julia
LMP.build_lmp_32pt_layers(
    U, x, D; max_len=5, num_eig=1, block_r=6, init_gpu=false,
)
```

Require BlockScalar count one versus zero and 38 layers for both variants.

- [ ] **Step 4: Reuse the random-field pattern**

Seed both `Random` and `CUDA`, apply `gauss_distribution_fermion!` to the only `fermionfield_Layer`, synchronize, and hash the resulting GPU payload copied to CPU. Never regenerate it between configurations.

- [ ] **Step 5: Wrap the existing streaming forward**

Allocate `TimerOutput`, `u::Vector{Any}`, one-element `temp`, and forward labels. Call:

```julia
LMP.Layer_sequence._checkpointed_forward_loss!(
    timer, labels, layers, D, u, temp, 1, depth, depth + 1,
)
```

Drop returned checkpoints, clear forward caches, synchronize, and require the real loss to be finite and positive.

- [ ] **Step 6: Verify every loaded weight**

For each retained `layer_N_W`, compare the checkpoint array with `Array(layer._W_gpu)` for Dense/PT layers or `layer.W` for BlockScalar. Fail on missing keys, wrong layer types, wrong shapes, or unequal values.

- [ ] **Step 7: Run GREEN**

Require the complete non-GPU tests and source contracts to pass.

### Task 3: Smoke Evaluation on Configuration 0

**Files:**
- Runtime output: `/home/choi/LMP/results/experiments/eig1_volume_generalization_ritz_loss_24cube48time_beta62_k1535_smoke_cfg0_20260908`

**Interfaces:**
- Consumes `EIG1_VOLUME_CONFIG_IDS=0` and the two archived checkpoints.
- Produces two successful loss rows, derived-checkpoint provenance, logs, plot, and validation report.

- [ ] **Step 1: Implement output and validation paths**

Write append-only CSV rows, `settings.txt`, derived-checkpoint provenance, paired per-model logs, a paired-point PNG, and `validation.txt`. Refuse any pre-existing result root.

- [ ] **Step 2: Add sequential model execution**

For each variant, initialize the same seeded field, build/load/verify the architecture, warm the streaming loss once on config 0, record config 0 after warm-up, verify the unchanged field digest, log CUDA memory, then release layers and reclaim CUDA memory before the next variant.

- [ ] **Step 3: Run preflight**

Require both checkpoints and all 30 gauges, verify hashes, inspect both GPUs without stopping other jobs, and select a GPU with sufficient free memory. If neither GPU is sufficiently free, wait without changing the experiment.

- [ ] **Step 4: Run smoke**

Run with a new result root and `EIG1_VOLUME_CONFIG_IDS=0`. Require two finite positive rows, one row per variant, matching random-field digests, exact loaded weights, no OOM, and a readable PNG.

### Task 4: Full 30-Configuration Production

**Files:**
- Runtime output: `/home/choi/LMP/results/experiments/eig1_volume_generalization_ritz_loss_24cube48time_beta62_k1535_20260908`

**Interfaces:**
- Consumes IDs 0 through 29 with the exact smoke-validated settings.
- Produces 60 successful rows and the publication-style paired-point PNG.

- [ ] **Step 1: Launch unchanged production settings**

Use a persistent tmux session. Preserve exact checkpoints, kappa, seed, streaming method, model order, and GPU. Redirect stdout/stderr to `launcher.log`.

- [ ] **Step 2: Monitor without interference**

Do not stop, replace, or change the run. Report completed row count, active model/configuration, GPU memory, and any explicit error.

- [ ] **Step 3: Verify completion**

Require exactly 60 successful rows and 30 paired IDs, unchanged hashes/digests, finite positive losses, exact weight verification for both variants, two derived checkpoints with provenance, no temporary files, a readable nonempty PNG, and every validation check marked PASS.
