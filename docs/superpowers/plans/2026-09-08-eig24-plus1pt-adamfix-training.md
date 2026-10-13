# Corrected-Adam eig24 PT-Replacement Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start and validate one fresh 4000-step corrected-Adam `No BlockScalar + 1 PTLayer eig24` training run matched to the completed corrected-Adam `BlockScalar eig24` baseline.

**Architecture:** Reuse `scripts/training/train_lmp.jl` without changing its training loop. Add one environment-only launcher and one source-contract test in the established outer experiment tree, preflight the baseline checkpoint and gauge set, then launch on one free GPU in `tmux` and validate the live metadata and first losses.

**Tech Stack:** Julia 1.11.9, CUDA.jl, JLD2, LMP, `Test`, `tmux`, NVIDIA A100 80GB.

## Global Constraints

- Reuse `/data1/choi/LMP_path33_kappa0158_24eig_scratch_adamfix_20260902` unchanged.
- Train only `NUM_EIG=24`, `LMP_TAIL_VARIANT=plus1pt_no_blockscalar`.
- Use kappa 0.158, 320 seeded training gauges, test gauges 401:420, batch 1, learning rate 0.001, and exactly 4000 fresh steps.
- Save every 1000 steps and test every 10 steps.
- Use corrected Adam and require final `opt_t=4000`.
- Refuse to overwrite `/data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908`.
- Never stop, replace, resume, or alter the launched run without explicit user instruction.
- Preserve partial outputs and logs after any failure.
- The operational `scripts/` and `test/` trees are outside the inner Git repository; commit only this plan document and preserve the operational files in place.

---

### Task 1: Add the exact training launcher with a source-contract test

**Files:**
- Create: `/home/choi/LMP/test/train_lmp_k158_24eig_plus1pt_adamfix_config.jl`
- Create: `/home/choi/LMP/scripts/training/run_train_lmp_k158_24eig_no_blockscalar_plus1pt_scratch_4000_adamfix.jl`
- Reuse unchanged: `/home/choi/LMP/scripts/training/train_lmp.jl`

**Interfaces:**
- Consumes: environment values supported by `train_lmp.jl`.
- Produces: a launcher that refuses an existing target and includes `train_lmp.jl` with the approved fixed values.

- [ ] **Step 1: Write the failing source-contract test**

```julia
using Test

const ROOT = normpath(joinpath(@__DIR__, ".."))
const RUNNER = joinpath(
    ROOT, "scripts", "training",
    "run_train_lmp_k158_24eig_no_blockscalar_plus1pt_scratch_4000_adamfix.jl",
)

@testset "eig24 PT-replacement corrected-Adam launcher" begin
    @test isfile(RUNNER)
    if isfile(RUNNER)
        source = read(RUNNER, String)
        for required in (
            "NUM_EIG\"] = \"24\"",
            "LMP_TAIL_VARIANT\"] = \"plus1pt_no_blockscalar\"",
            "TRAIN_KAPPA\"] = \"0.158\"",
            "ADDITIONAL_STEPS\"] = \"4000\"",
            "CHECKPOINT_EVERY\"] = \"1000\"",
            "TEST_INTERVAL\"] = \"10\"",
            "BATCH_SIZE\"] = \"1\"",
            "ACTIVATION_CHECKPOINT_INTERVAL\"] = \"0\"",
            "RESUME_EXISTING_RUN\"] = \"0\"",
            "START_CKPT_PATH\"] = \"\"",
            "24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908",
            "include(joinpath(@__DIR__, \"train_lmp.jl\"))",
        )
            @test occursin(required, source)
        end
        @test occursin("ispath(TARGET_DIR)", source)
        @test occursin("refusing to overwrite", source)
    end
end
```

- [ ] **Step 2: Run the test and require the missing-runner failure**

Run:

```bash
julia --project=/home/choi/LMP/LMP \
  /home/choi/LMP/test/train_lmp_k158_24eig_plus1pt_adamfix_config.jl
```

Expected: one failed `isfile(RUNNER)` assertion because the launcher does not yet exist.

- [ ] **Step 3: Add the minimal environment-only launcher**

```julia
#!/usr/bin/env julia

const TARGET_DIR =
    "/data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908"
ispath(TARGET_DIR) && error("refusing to overwrite training target: $TARGET_DIR")

visible = strip(get(ENV, "CUDA_VISIBLE_DEVICES", "0"))
devices = filter(!isempty, strip.(split(visible, ',')))
length(devices) == 1 || error("exactly one CUDA device is required: $visible")
devices[1] in ("0", "1") || error("CUDA device must be 0 or 1: $(devices[1])")

ENV["CUDA_VISIBLE_DEVICES"] = devices[1]
ENV["ALLOW_CUDA_DEVICE1"] = "true"
ENV["NUM_EIG"] = "24"
ENV["LMP_TAIL_VARIANT"] = "plus1pt_no_blockscalar"
ENV["TRAIN_KAPPA"] = "0.158"
ENV["CKPT_DIR"] = TARGET_DIR
ENV["CKPT_PREFIX"] = "(16x32)33PTlayer33path(kappa=0.158)"
ENV["CKPT_SUFFIX"] =
    "(maxlen5)(xavier_init)(24eig)(320config)(no_blockscalar_plus1pt)"
ENV["BASE_CKPT_SUFFIX"] = ENV["CKPT_SUFFIX"]
ENV["RESUME_EXISTING_RUN"] = "0"
ENV["START_CKPT_PATH"] = ""
ENV["ADDITIONAL_STEPS"] = "4000"
ENV["CHECKPOINT_EVERY"] = "1000"
ENV["TEST_INTERVAL"] = "10"
ENV["BATCH_SIZE"] = "1"
ENV["ACTIVATION_CHECKPOINT_INTERVAL"] = "0"
ENV["LOG_PATH"] = joinpath(TARGET_DIR, "train_lmp_33path_24eig_plus1pt_adamfix.log")

include(joinpath(@__DIR__, "train_lmp.jl"))
```

- [ ] **Step 4: Run the source-contract test and parse check**

Run:

```bash
julia --project=/home/choi/LMP/LMP \
  /home/choi/LMP/test/train_lmp_k158_24eig_plus1pt_adamfix_config.jl
julia --project=/home/choi/LMP/LMP -e \
  'Meta.parseall(read(ARGS[1], String)); println("parse=PASS")' \
  /home/choi/LMP/scripts/training/run_train_lmp_k158_24eig_no_blockscalar_plus1pt_scratch_4000_adamfix.jl
```

Expected: the test passes and the parser prints `parse=PASS` without starting training.

### Task 2: Preflight and launch the training once

**Files:**
- Consume: the launcher from Task 1.
- Consume: the completed baseline final checkpoint.
- Produce: `/home/choi/LMP/results/run/train_eig24_plus1pt_adamfix_20260908.launch.log`
- Produce at runtime: the approved target directory and training log.

**Interfaces:**
- Consumes: a free physical GPU and the exact gauge files selected by seed 42.
- Produces: `tmux` session `train_eig24_plus1pt_adamfix_gpu0_20260908` and a Julia training process.

- [ ] **Step 1: Validate the baseline checkpoint and all gauges without allocating a GPU**

```bash
julia --project=/home/choi/LMP/LMP -e '
using JLD2, Random
baseline = "/data1/choi/LMP_path33_kappa0158_24eig_scratch_adamfix_20260902/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(24eig)(320config)(r=2)_final.jld2"
isfile(baseline) || error("missing baseline checkpoint")
JLD2.jldopen(baseline, "r") do f
    Int(f["opt_t"]) == 4000 || error("baseline opt_t is not 4000")
end
Random.seed!(42)
train_ids = randperm(400)[1:320]
train_paths = ["/data1/choi/heatbath_ensemble_beta6.0/raw_gauge/config_$(i).gbin" for i in train_ids]
test_paths = ["/data1/choi/heatbath_ensemble_beta6.0/raw_gauge/config_$(i).gbin" for i in 401:420]
all(isfile, vcat(train_paths, test_paths)) || error("one or more gauge files are missing")
println("baseline_opt_t=4000 gauges=340 preflight=PASS")
'
```

Expected: `baseline_opt_t=4000 gauges=340 preflight=PASS`.

- [ ] **Step 2: Require a free GPU, absent target, absent log, and absent session**

```bash
test ! -e /data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908
test ! -e /home/choi/LMP/results/run/train_eig24_plus1pt_adamfix_20260908.launch.log
! tmux has-session -t train_eig24_plus1pt_adamfix_gpu0_20260908 2>/dev/null
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | sed '/^[[:space:]]*$/d')"
```

Expected: exit code 0. If any check fails, do not launch or alter the existing state.

- [ ] **Step 3: Launch on GPU 0 in a persistent session**

```bash
tmux new-session -d -s train_eig24_plus1pt_adamfix_gpu0_20260908 \
  "cd /home/choi/LMP/scripts/training && exec env CUDA_VISIBLE_DEVICES=0 \
   julia --project=/home/choi/LMP/LMP \
   run_train_lmp_k158_24eig_no_blockscalar_plus1pt_scratch_4000_adamfix.jl \
   > /home/choi/LMP/results/run/train_eig24_plus1pt_adamfix_20260908.launch.log 2>&1"
```

Expected: the session remains alive, one Julia process appears on GPU 0, and the target directory is created.

### Task 3: Validate the live start and preserve the run

**Files:**
- Inspect: `/data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908/train_lmp_33path_24eig_plus1pt_adamfix.log`
- Inspect: `/home/choi/LMP/results/run/train_eig24_plus1pt_adamfix_20260908.launch.log`

**Interfaces:**
- Consumes: the live process from Task 2.
- Produces: evidence that the requested run, rather than another configuration, is training.

- [ ] **Step 1: Check the live process and GPU ownership**

```bash
tmux has-session -t train_eig24_plus1pt_adamfix_gpu0_20260908
pgrep -af run_train_lmp_k158_24eig_no_blockscalar_plus1pt_scratch_4000_adamfix.jl
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

Expected: one launcher Julia process using GPU 0; do not stop it.

- [ ] **Step 2: Validate metadata and the first finite losses**

Require the training log to contain:

```text
# epochs_requested,4000
# configs_per_epoch,320
# test_gauge_count,20
# learning_rate,0.001
# num_eig,24
# tail_variant,plus1pt_no_blockscalar
# resume_existing_run,false
# total_steps,4000
```

Then require a finite `Step: 1` loss and, after the run naturally reaches step 10, a finite `[Test] Step: 10` loss. Report these values without pausing or replacing the run.

- [ ] **Step 3: Leave the training running and report its identifiers**

Report the session, Julia PID, GPU, target directory, training log, first training loss, first test loss, and the observed baseline duration of about 19 hours 38 minutes as a non-binding estimate. Do not claim completion before the natural end and final validation.

### Task 4: Validate natural completion when it occurs

**Files:**
- Inspect: all target checkpoints and both logs.
- Create after completion: `/data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908/validation.txt`

**Interfaces:**
- Consumes: the naturally completed training run.
- Produces: a validated PT-replacement final checkpoint for the volume experiment.

- [ ] **Step 1: Require the process and session to have ended naturally**

Check that no matching Julia process and no matching `tmux` session remain. Do not terminate either to satisfy this check.

- [ ] **Step 2: Validate all five checkpoints and optimizer steps**

Require step checkpoints 1000, 2000, 3000, and 4000 plus the final checkpoint. Read each `opt_t`; require 1000, 2000, 3000, 4000, and 4000 respectively.

- [ ] **Step 3: Validate architecture and logs**

Load the final checkpoint into the approved `plus1pt_no_blockscalar`, `num_eig=24` stack. Require 38 layers, zero `BlockScalar_Layer` instances, one additional width-33 `PTLayer`, a width-24 output, finite training and test losses through step 4000, and no checkpoint-skip warning.

- [ ] **Step 4: Record checksums and the validation report**

Write SHA-256 values for the final checkpoint and logs into `validation.txt`, together with every required check and its result. Only an all-pass report authorizes the later `24^3 x 48` DD-alphaAMG evaluation.
