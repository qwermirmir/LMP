# Mixed-long 24-vector 4,000→16,000 trajectory design

## Objective

Continue the exact mixed-long 24-vector model used by the current DD-αAMG
comparison from optimizer step 4,000 through step 16,000. Save a checkpoint
every 100 steps so downstream experiments can measure how Ritz loss and AMG
performance evolve after step 4,000.

## Authoritative starting state

The start checkpoint is the original step-4,000 checkpoint from
`LMP_path33_kappa0158_24eig_blockscalar_r2_scratch_adamfix_20260910`, whose
SHA-256 is
`c278e6b6eec51753cf9b04cce142979f2ed47dbe960cc1c75b972c42a20a0b2c`.
This is the checkpoint used as `mixed-long` in the final AMG comparison.
Loading must restore both trainable weights and the Adam state with
`opt_t == 4000`.

The separately regenerated
`LMP_loss_amg_trajectory_long_2000_4000_ckpt100_20260925` branch is not an
authoritative continuation. Its batch-one resume path restarted the training
configuration index because the training loop ignored `epoch_offset` when
selecting configurations. It must not be used as the step-4,000 start or
merged with the new trajectory as if it were the same optimization path.

## Resume correction

For batch size one, training configuration selection and prefetch selection
must use the absolute step, not the local post-resume step. With 320 training
configurations, step 4,000 uses configuration 160 and step 4,001 must use
configuration 161. The correction must preserve fresh-training behavior when
`epoch_offset == 0` and remain consistent with the existing batch-greater-than-
one path.

The correction is part of the generic trainer because the error is in the
trainer's resume semantics, not in this experiment's launcher. A regression
test must fail on the current implementation and prove the absolute-step
mapping before the trainer is modified.

## Training configuration

Only the requested training horizon and checkpoint frequency change. Preserve:

- mixed-long 33-path structure (`path_variant=:long`, maximum path length 5);
- 32 PT layers, 24 output vectors, and BlockScalar radius 2;
- κ = 0.158;
- Ritz-only objective with weight 1.0 and no coarse or zero-mode loss;
- Adam learning rate 0.001 with the restored step-4,000 moments;
- batch size 1, the same 320 training configurations, 20 test configurations,
  and training seed 42;
- test loss every 10 steps and activation checkpointing disabled.

Train for 12,000 additional optimizer steps, ending at absolute step 16,000.
Write checkpoints at steps 4,100, 4,200, …, 16,000 into a new directory. Each
checkpoint is expected to be approximately 112 MB, for roughly 13.4 GB total.
Do not overwrite or append to either existing 4,000-step directory.

## Validation gates

Before the full run:

1. Verify the authoritative checkpoint file hash and `opt_t == 4000`.
2. Run an isolated one-step smoke continuation in its own directory.
3. Verify the smoke log reports absolute step 4,001 and configuration 161/320.
4. Verify the saved checkpoint reports `opt_t == 4001`, retains the exact
   mixed-long PT paths, contains finite weights and optimizer state, and differs
   from the step-4,000 weights.
5. Reload the saved checkpoint and verify that its forward loss is unchanged.

The full run may start only after all gates pass. It should use an available
single GPU without stopping or replacing another process. The launch report
must record the GPU, persistent session, log, output directory, start hash,
and first observed training step.

## Completion and downstream use

Completion requires a clean training-finished marker, checkpoints for every
100-step boundary from 4,100 through 16,000, a final step-16,000 checkpoint,
and matching optimizer steps. The new 4,000→16,000 checkpoints form one
continuous trajectory with the original step-4,000 model. The old regenerated
2,100→4,000 branch remains preserved but separately labeled and is excluded
from this trajectory.
