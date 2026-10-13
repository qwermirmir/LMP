# Ritz-loss versus AMG trajectory implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regenerate 100-step short/long checkpoints from step 2000 and evaluate all 40 checkpoints at two κ values for loss-versus-AMG plots.

**Architecture:** A single model-selecting resume launcher reuses `train_lmp.jl` without altering training logic. A resumable evaluation driver reuses the validated long/short AMG setup and writes one row after every solve. A plotting script reads only the completed CSV.

**Tech Stack:** Julia 1.11, LMP, CUDA, JLD2, Python/matplotlib.

## Global Constraints

- Generate steps 2100:100:4000 for `:short_repeat4` and `:long` from their original step-2000 checkpoints.
- Preserve optimizer state, fermion field, data ordering, κ=0.158, Nv=24, BlockScalar r=2, Adam 0.001, batch 1, and Ritz-only loss.
- Evaluate config 450 at κ=0.157 and 0.15860 with the exact established DD-αAMG settings and fixed paired RHS.
- Never overwrite original checkpoints or stop unrelated GPU work.

### Task 1: Resume launcher

**Files:**
- Create: `scripts/training/run_resume_lmp_loss_amg_trajectory_20260925.jl`

- [ ] Parse `TRAJECTORY_MODEL=short|long` and `TRAJECTORY_SMOKE=true|false`; reject other values.
- [ ] Map each model to its original 2000 checkpoint, exact path variant, original checkpoint prefix, and unique smoke/full output directory.
- [ ] Set `BASE_STEP=2000`, `START_CKPT_PATH`, `RESUME_EXISTING_RUN=0`, `SAVE_INITIAL_CHECKPOINT=false`, and use one additional step/checkpoint interval 1 for smoke or 2000 additional steps/checkpoint interval 100 for full.
- [ ] After smoke, verify optimizer step 2001, all 34 finite and changed weight arrays, exact saved path metadata, and forward loss equality after reload.
- [ ] Parse-test the launcher and run one smoke per model on GPU0.

### Task 2: Persistent checkpoint generation

**Files:**
- Runtime directories under `/data1/choi/LMP_loss_amg_trajectory_*_20260925/`
- Runtime logs under `results/run/`

- [ ] Confirm GPU0 and all targets/sessions are unused.
- [ ] Start a persistent GPU0 session that runs full short training and then full long training, stopping the chain if either command fails.
- [ ] Verify short metadata and first post-resume step 2001 before reporting launch status.
- [ ] After completion, verify exactly 20 selected checkpoints per model and compare regenerated steps 3000/4000 with originals.

### Task 3: Resumable 80-case AMG measurement

**Files:**
- Create: `scripts/experiments/evaluate_ritz_loss_amg_trajectory_cfg450.jl`

- [ ] Reuse the setup/solve routines and constants from `compare_long_short33_amg.jl`; change only the checkpoint sweep and recorded metrics.
- [ ] Parse mean test losses at steps 2100:100:4000 from each resumed training log.
- [ ] For each κ, step, and model, verify checkpoint/path/optimizer step, calculate config-450 Ritz and Gram metrics, then run fresh AMG setup and solve with fixed RHS.
- [ ] Append each result atomically and skip already-completed `(model,step,kappa)` keys on restart; record capped solves instead of terminating.
- [ ] Smoke one checkpoint/model at κ=0.157 before the full 80-case execution.

### Task 4: Figures and correlation table

**Files:**
- Create: `scripts/plotting/plot_ritz_loss_amg_trajectory.py`

- [ ] Validate 80 unique rows and paired gauge/RHS hashes.
- [ ] Produce κ-panel fine-versus-coarse plots with model style and shared loss color scale.
- [ ] Produce loss-versus-fine/coarse/total plots connected in step order.
- [ ] Write per-model/per-κ Spearman correlations and convergence counts to CSV.
