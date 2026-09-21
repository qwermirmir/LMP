# Long-only 33-path LMP training and comparison design

## Objective

Test whether an LMP restricted to longer gauge-transport paths produces a more useful DD-αAMG coarse space than the existing short-repeat model, despite a possibly higher Ritz training loss. The experiment measures both Ritz loss and downstream solver work; neither metric substitutes for the other.

## Architecture

Add a separate `:long_only2468` path variant without changing `:long` or `:short_repeat4`. Its 33 ordered paths are one identity `[0]`, followed by the eight signed directions `(1, 2, 3, 4, -1, -2, -3, -4)` at each length in `(2, 4, 6, 8)`. All nonidentity paths are straight repeated-direction paths. Keep the existing 32-PT-layer model, 24 output vectors, BlockScalar radius 2, and 33 path slots. The number of trainable complex parameters must equal the short-repeat and original long 33-path models.

The current `max_len=8` long-path generator does **not** produce this list (it omits length 6 and includes length 1). Construction must therefore select the explicit new variant. The checkpoint must store and verify all PT path arrays so a different variant cannot be loaded silently.

## Training

Reuse the existing `train_lmp.jl` implementation and its checkpoint machinery through a new, uniquely named launcher. Train from scratch for 4,000 optimizer steps with `κ=0.158`, 24 vectors, BlockScalar `r=2`, batch size 1, Adam learning rate 0.001, the same 320 training configurations and 20 test configurations, and the same Ritz-only objective as the September short-repeat run. Set `INIT_SEED=20260918` to match that run's initialization stream and preserve its configuration-selection seed of 42. Use a new output directory and checkpoint prefix; never overwrite either existing experiment.

This matched seed controls the initial random stream, but it does not turn this into an ablation against the older long model, which was trained without an explicit matching initialization seed. Any comparison to that model is observational.

## Evaluation

First validate the final checkpoint by checking optimizer step 4,000, all PT path arrays, weight shapes, finite weights, and a load/reload forward-loss round trip on the same gauge. Compare 4,000-step training and 401–420 mean test Ritz losses with the existing short and long logs.

Then evaluate DD-αAMG on the same four measured conditions as the existing long/short experiment: configurations 401 and 450, with `κ=(0.157, 0.15775)` for 401 and `κ=(0.157, 0.15860)` for 450. Preserve the exact gauge, RHS, 24-vector setup, smoothing, chirality, deflated initial guess, solve tolerance, restart, and coarse solver settings from `compare_long_short33_amg.jl`; only the learned model and its path-dependent checkpoint may differ. Perform an untimed warm-up before timing. Record fine and coarse operator applications, verified residual, convergence, setup/inference/solve times, seeds, gauge/checkpoint identities, and per-case logs in a new result directory. A lower Ritz loss counts as better Ritz loss only; the AMG decision is based on verified convergence and application/time cost.

## Verification and failure handling

Test the exact 33-path sequence and its uniqueness, the parameter-count match, the checkpoint path metadata, and short/long variant mismatch rejection. Run a small smoke training and reload test before a full run. Preserve all running or partial intentional experiments; never interrupt or replace one without explicit user direction. If a required checkpoint, GPU, or configuration is unavailable, report that fact instead of changing the requested design.
