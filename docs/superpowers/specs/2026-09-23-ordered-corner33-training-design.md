# Ordered corner33 LMP training design

## Objective

Test whether local direction-changing gauge transports produce a DD-αAMG coarse space that is more useful than repeated one-hop paths while remaining trainable under the same Ritz objective. This is a path-geometry experiment: model width, output count, BlockScalar layer, optimizer, data, and training duration remain fixed.

## Path architecture

Add a new path variant named `:corner33_ordered` without changing `:long`, `:short_repeat4`, or `:long_only2468`. It contains exactly 33 unique path slots in this order:

1. Identity `[0]`.
2. The eight one-hop paths `[1]`, `[2]`, `[3]`, `[4]`, `[-1]`, `[-2]`, `[-3]`, `[-4]`.
3. For each unsigned plane `μ < ν` in `(1,2)`, `(1,3)`, `(1,4)`, `(2,3)`, `(2,4)`, `(3,4)`, the four ordered corners `[μ,ν]`, `[μ,-ν]`, `[-μ,ν]`, `[-μ,-ν]`.

The corner ordering is deliberately fixed as `μ` then `ν`; reverse-ordered paths are outside this experiment. All PT layers use this identical path list. The model remains 32 PT layers with 24 output vectors and BlockScalar `r=2`, so its trainable complex parameter count must equal the short33 and existing long33 models.

## Training

Reuse `scripts/training/train_lmp.jl` through a dedicated launcher. Train from scratch for 4,000 optimizer steps with `κ=0.158`, 24 vectors, BlockScalar `r=2`, batch size 1, Adam learning rate 0.001, the same 320 training configurations and 20 test configurations, and the Ritz-only objective used by the short33 experiment. Set `INIT_SEED=20260918` and preserve configuration-selection seed 42 so initialization and data order match short33.

Use separate non-overwriting directories:

- Smoke: `/data1/choi/LMP_corner33_ordered_k158_24eig_r2_smoke_20260923`
- Full: `/data1/choi/LMP_corner33_ordered_k158_24eig_r2_4000_20260923`

Run on GPU1 after verifying availability. Do not stop or replace any other run.

## Verification

Before full training:

1. Verify the exact 33-path sequence, uniqueness, parameter-count equality, and rejection of a short33 checkpoint/path mismatch.
2. On a CPU `4^4` lattice, build three PT layers ending in a Ritz loss and compare analytical gradients against centered finite differences for a one-hop slot and representative corner slots in every layer.
3. Run a two-step GPU smoke test and require finite, changed weights; optimizer step 2; exact saved PT path metadata; and checkpoint reload loss equality.

Only after all checks pass may the 4,000-step run start. Save checkpoints at steps 0, 1000, 2000, 3000, and 4000 plus the final checkpoint. The training log must record `path_variant=corner33_ordered`, the exact path list, seed, `κ`, vector count, and BlockScalar radius.

## Interpretation

The primary immediate outputs are train/test Ritz trajectories. Subsequent DD-αAMG comparison must use the established fixed gauges, RHS values, setup smoothing, chirality, deflated initial guess, and solver settings. A lower Ritz loss alone is not evidence of superior AMG performance.
