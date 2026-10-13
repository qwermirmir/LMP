# Mixed-long 24-vector 16,000-step trajectory implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continue the exact mixed-long 24-vector step-4,000 model to step 16,000 while saving checkpoints every 100 steps.

**Architecture:** Correct the generic batch-one resume index so gauge selection uses absolute optimizer steps. Reuse `train_lmp.jl` through a dedicated, isolated launcher that restores the original step-4,000 weights and Adam state, gates the full run behind a one-step smoke test, and writes only to new directories.

**Tech Stack:** Julia 1.11, LMP, JLD2, CUDA, Python unittest for launcher source contracts.

## Global constraints

- Start only from the original mixed-long step-4,000 checkpoint with SHA-256 `c278e6b6eec51753cf9b04cce142979f2ed47dbe960cc1c75b972c42a20a0b2c` and `opt_t == 4000`.
- Preserve mixed-long paths, 32 PT layers, 24 vectors, BlockScalar `r=2`, κ=0.158, Ritz-only loss, Adam learning rate 0.001, batch size 1, 320 training configurations, 20 test configurations, and seed 42.
- Save steps 4,100 through 16,000 every 100 steps; evaluate test loss every 10 steps.
- Never overwrite, delete, stop, or replace an existing experiment.
- Do not use or merge the divergent regenerated 2,100→4,000 trajectory as the continuation start.

---

### Task 1: Absolute configuration indexing on resume

**Files:**
- Create: `test/training_resume_config_index.jl`
- Modify: `src/Layer_sequence.jl`

**Interfaces:**
- Produces: `Layer_sequence._training_config_index(epoch_offset, step, batch, batch_position, count)::Int`.
- Consumes: the existing `train(...; epoch_offset, batch)` loop.

- [ ] **Step 1: Write the failing test** in `test/training_resume_config_index.jl`:

```julia
using Test, LMP
const LS = LMP.Layer_sequence

@testset "absolute training configuration index" begin
    @test LS._training_config_index(0, 1, 1, 1, 320) == 1
    @test LS._training_config_index(4000, 1, 1, 1, 320) == 161
    @test LS._training_config_index(4000, 2, 1, 1, 320) == 162
    @test LS._training_config_index(4000, 1, 2, 1, 320) == 1
    @test LS._training_config_index(4000, 1, 2, 2, 320) == 2
end
```

- [ ] **Step 2: Verify RED.** Run `julia --project=. --startup-file=no test/training_resume_config_index.jl` from `/home/choi/LMP/LMP`. Expect `UndefVarError: _training_config_index not defined`.
- [ ] **Step 3: Add the minimal helper** before the gauge-path `train` method:

```julia
_training_config_index(epoch_offset::Int, step::Int, batch::Int,
                       batch_position::Int, count::Int) =
    mod1((epoch_offset + step - 1) * batch + batch_position, count)
```

Use it for the initial gauge, batch-one `current_k` and `next_k`, and the existing batch-greater-than-one sample mapping. The initial raw/JLD2 gauge must be `gauge_paths[initial_k]`, where `initial_k = _training_config_index(epoch_offset, 1, batch, 1, N)`.
- [ ] **Step 4: Verify GREEN.** Run the new test plus `test/long_only1248_paths.jl`, `test/long_only2468_paths.jl`, and `test/short33_paths.jl`; all must pass.
- [ ] **Step 5: Commit** only `src/Layer_sequence.jl` and `test/training_resume_config_index.jl` after reviewing the pre-existing `Layer_sequence.jl` diff.

### Task 2: Isolated 4,000→16,000 continuation launcher

**Files:**
- Create: `/home/choi/LMP/scripts/training/mixedlong_16000_trajectory_config.jl`
- Create: `/home/choi/LMP/scripts/training/test_mixedlong_16000_trajectory_config.jl`
- Create: `/home/choi/LMP/scripts/training/run_resume_lmp_mixedlong24_4000to16000_trajectory_20260930.jl`
- Create: `/home/choi/LMP/scripts/training/test_run_resume_lmp_mixedlong24_trajectory.py`

**Interfaces:**
- Produces: `mixedlong_16000_trajectory_config(smoke::Bool)` with the authoritative start, isolated target, total additional steps, and checkpoint cadence.
- Consumes: `scripts/training/train_lmp.jl` and Task 1's corrected resume semantics.

- [ ] **Step 1: Write failing configuration tests** asserting full `base_step=4000`, `additional_steps=12000`, `checkpoint_every=100`, `test_interval=10`, the original checkpoint suffix, and new full/smoke target directories. Add a Python source contract requiring SHA verification, smoke `Config: 161/320`, `opt_t == 4001`, PT-path checks, finite state checks, changed weights, and reload-loss equality.
- [ ] **Step 2: Verify RED.** Run the Julia and Python tests; expect missing config/launcher failures.
- [ ] **Step 3: Implement the config and launcher** by setting:

```julia
ENV["NUM_EIG"] = "24"
ENV["BLOCK_SCALAR_R"] = "2"
ENV["LMP_TAIL_VARIANT"] = "blockscalar"
ENV["LMP_PATH_VARIANT"] = "long"
ENV["TRAIN_KAPPA"] = "0.158"
ENV["TRAIN_LR"] = "0.001"
ENV["RITZ_LOSS_WEIGHT"] = "1.0"
ENV["COARSE_LOSS_WEIGHT"] = "0.0"
ENV["USE_ZERO_MODE_LOSS"] = "false"
ENV["RESUME_EXISTING_RUN"] = "0"
ENV["START_CKPT_PATH"] = CONFIG.start_checkpoint
ENV["BASE_STEP"] = "4000"
ENV["ADDITIONAL_STEPS"] = string(CONFIG.additional_steps)
ENV["CHECKPOINT_EVERY"] = string(CONFIG.checkpoint_every)
ENV["TEST_INTERVAL"] = string(CONFIG.test_interval)
ENV["BATCH_SIZE"] = "1"
```

Refuse pre-existing targets and reject a start-checkpoint hash other than the authoritative value before including `train_lmp.jl`.
- [ ] **Step 4: Verify GREEN.** Run both config/launcher tests and parse the launcher with `Meta.parseall`; all must pass.

### Task 3: One-step smoke continuation

**Files:**
- Runtime output: `/data1/choi/LMP_mixedlong24_k158_r2_4000to16000_trajectory_smoke_20260930/`

**Interfaces:**
- Consumes: Task 2 launcher with `TRAJECTORY_SMOKE=true`.
- Produces: a verified step-4,001 checkpoint and smoke log.

- [ ] **Step 1: Confirm** both GPUs, active processes, start hash, free target path, and at least 15 GB free storage without stopping any process.
- [ ] **Step 2: Run smoke** on one available GPU:

```bash
TRAJECTORY_SMOKE=true CUDA_VISIBLE_DEVICES=1 \
  julia --project=LMP --startup-file=no \
  scripts/training/run_resume_lmp_mixedlong24_4000to16000_trajectory_20260930.jl
```

- [ ] **Step 3: Require** `MIXEDLONG_16000_TRAJECTORY_SMOKE_PASS`, log step 4,001 with `Config: 161/320`, checkpoint `opt_t=4001`, exact mixed-long PT paths, finite state, at least one changed weight, and forward-loss reload equality.

### Task 4: Launch the full continuation

**Files:**
- Runtime output: `/data1/choi/LMP_mixedlong24_k158_r2_4000to16000_trajectory_ckpt100_20260930/`

**Interfaces:**
- Consumes: the smoke-verified launcher with `TRAJECTORY_SMOKE=false`.
- Produces: checkpoints 4,100:100:16,000 and a persistent training log.

- [ ] **Step 1: Recheck** GPU availability and verify the full target does not exist. Do not interrupt any active process.
- [ ] **Step 2: Start** the launcher in a uniquely named persistent tmux session with stdout/stderr redirected to a uniquely named run log.
- [ ] **Step 3: Verify launch** by confirming the process is alive, the metadata names the authoritative start checkpoint and `start_step=4000`, the first line is step 4,001 with configuration 161/320, and GPU utilization/memory rises on the selected device.
- [ ] **Step 4: Report** the session, GPU, output directory, training log, run log, start hash, observed first step, expected checkpoint count 120, estimated storage 13.4 GB, and estimated runtime about 60 hours. Do not claim training completion at launch.
