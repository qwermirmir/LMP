# Ritz-loss versus AMG trajectory experiment design

## Objective

Measure how DD-αAMG performance changes along the optimization trajectories of the existing short-path and mixed-long-path 24-vector models. The experiment relates Ritz loss to solver work; it does not treat lower Ritz loss as automatically better AMG performance.

## Models and checkpoint generation

Compare exactly two existing architectures:

- Short: `:short_repeat4`, one identity plus four repetitions of the eight signed one-hop directions.
- Long: `:long`, the existing 33-path lengths `1, 2, 4, 5` model.

Resume each model from its original 2,000-step W-only checkpoint, including Adam state and stored fermion field. Preserve the original architecture, `κ=0.158`, 24 output vectors, BlockScalar `r=2`, batch size 1, learning rate 0.001, Ritz-only objective, 320 training configurations, 20 test configurations, and configuration ordering. Train steps 2,001 through 4,000 and save checkpoints at `2100, 2200, ..., 4000`, yielding 20 checkpoints per model. Use separate non-overwriting directories and never modify the original checkpoints.

The resumed 3,000- and 4,000-step checkpoints must be compared with the corresponding original checkpoints. Record exact equality or numerical differences for all weights, optimizer step/state, and stored fermion field. A mismatch does not authorize silently changing the trajectory; it must be reported and explained before AMG interpretation.

## Loss associated with each checkpoint

Use the 401–420 mean test Ritz loss logged immediately after each selected training step as the primary loss coordinate. Independently evaluate and record the checkpoint Ritz loss on configuration 450 at each target κ. Keep these quantities distinct in the output schema.

## AMG evaluation

Evaluate every checkpoint on configuration 450 at `κ=0.157` and `κ=0.15860`. Use the exact established long/short comparison setup:

- Block size 4 and 24 prototype vectors.
- `setup_mode=:direct_near_null`.
- External-vector smoothing enabled, SAP 3 sweeps with block-MR 4 steps.
- Chirality splitting enabled.
- Deflated initial guess enabled.
- Outer FGMRES tolerance `1e-10`, maximum 200 cycles, restart 50.
- Fine MR pre/post smoothing 4/4.
- Coarse tolerance 0.1, maximum 50 cycles, restart 20, odd-even disabled, `:restart_boundary` mode.
- SAP odd-even disabled and `:twocolor` schedule.

Use the same raw gauge and the same fixed RHS for both models and all checkpoints at a given κ. Construct fresh model and AMG state for every measurement. Perform untimed compilation warm-up before recorded measurements. Do not stop the experiment when a solve does not converge; record its capped counts and independently verified residual.

This produces `2 models × 20 checkpoints × 2 κ = 80` recorded AMG cases.

## Recorded data

For every case record:

- Model, architecture variant, checkpoint step and checkpoint identity/hash.
- Mean test Ritz loss and configuration-450 Ritz loss.
- Fine Dirac, coarse operator, and total application counts.
- Independently verified residual, converged/capped state.
- Inference, setup, initial-guess, solve, and total time.
- Raw output Gram minimum/maximum eigenvalues, condition number, numerical rank, and maximum pair correlation.
- Gauge and RHS paths/hashes, solver settings, execution order, and log path.

Write rows incrementally to a new result directory so completed measurements survive interruption.

## Plots and analysis

Produce two κ-panel figure families:

1. Fine applications versus coarse applications. Distinguish short and long by marker/line style and encode checkpoint loss or step with a shared color scale. Plot all 20 points per model.
2. Mean test Ritz loss versus fine, coarse, and total applications. Connect checkpoints in training-step order separately for each model.

Report Spearman correlations between loss and application counts for each model and κ. These correlations describe one gauge and a serially correlated optimization trajectory; they are not ensemble-level or causal estimates.

## Verification and safety

Before full execution, smoke-test one resumed step per model in disposable directories and verify optimizer advancement, exact architecture/path loading, changed finite weights, selected test-loss logging, and checkpoint reload equality. Verify paired gauge/RHS hashes before every AMG comparison. Do not stop, replace, or delete any unrelated running experiment or intentional partial output.
