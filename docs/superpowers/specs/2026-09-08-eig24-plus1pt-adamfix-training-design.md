# Corrected-Adam eig24 PT-Replacement Training Design

## Purpose

Prepare the matched `eig24` model pair needed for the later volume-
generalization DD-alphaAMG experiment. Reuse the completed corrected-Adam
`BlockScalar eig24` run and train only the missing
`No BlockScalar + 1 PTLayer eig24` model.

This phase covers training and checkpoint validation only. Evaluation on the
new `24^3 x 48`, beta 6.2 ensemble and the DD-alphaAMG comparison follow after
the new training finishes.

## Reused Baseline

Use this completed run as the `BlockScalar eig24` comparison model:

`/data1/choi/LMP_path33_kappa0158_24eig_scratch_adamfix_20260902`

Its final checkpoint is:

`/data1/choi/LMP_path33_kappa0158_24eig_scratch_adamfix_20260902/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(24eig)(320config)(r=2)_final.jld2`

The run used the corrected global Adam step and completed 4000 training steps.
Do not modify, resume, or replace this run.

## New Model

Train one model with:

- `NUM_EIG=24`
- `LMP_TAIL_VARIANT=plus1pt_no_blockscalar`

The shared body remains unchanged. The `BlockScalar` tail is replaced by one
additional width-33 `PTLayer`, followed by the width-24 output `PTLayer`, output
normalization, and Ritz loss. The resulting stack has the same depth, 38, as
the baseline run.

Reuse `/home/choi/LMP/scripts/training/train_lmp.jl`. It already implements
this tail variant and output dimension. Add only a small launcher derived from
the successful eig1 PT-replacement launcher; do not add a new training loop,
optimizer, layer, loss, data loader, or checkpoint format.

## Fixed Training Conditions

Match the completed baseline run:

- lattice: `16 x 16 x 16 x 32`
- gauge ensemble: `/data1/choi/heatbath_ensemble_beta6.0/raw_gauge`
- Wilson kappa: `0.158`
- training gauges: 320 IDs selected by the existing `Random.seed!(42)` and
  `randperm(400)` path
- test gauges: IDs 401 through 420
- training steps: 4000
- batch size: 1
- learning rate: 0.001
- test interval: 10 steps
- checkpoint interval: 1000 steps
- loss: Ritz only
- activation checkpoint interval: 0, matching the baseline
- fresh start: no checkpoint resume and no warm start

Use the corrected `Adam` implementation already present in the repository. The
training step must increment `opt.t` once, independently of the number of
trainable layers.

## Output and Non-Overwrite Rules

Use the new directory:

`/data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908`

Use prefix:

`(16x32)33PTlayer33path(kappa=0.158)`

Use suffix:

`(maxlen5)(xavier_init)(24eig)(320config)(no_blockscalar_plus1pt)`

Write checkpoints at steps 1000, 2000, 3000, and 4000, plus a separate final
checkpoint and complete launcher/training logs. Refuse to start if the target
directory or any requested output already exists.

## Reproducibility Limitation

The established `train_lmp.jl` initializes the pseudofermion field and Xavier
weights before it calls `Random.seed!(42)` for the configuration order. The
baseline corrected-Adam run therefore has a reproducible gauge order but not a
recorded bitwise-reproducible initial model state.

Preserve this established path instead of changing initialization behavior for
only the new architecture. The comparison is a single trained model per
architecture and does not separate architecture effects from initialization
variance.

## Preflight and Execution

Before launch:

- verify the completed baseline final checkpoint exists and contains
  `opt_t=4000`;
- verify all 320 selected training gauges and 20 test gauges exist;
- verify `train_lmp.jl` selects the PT-replacement stack for the requested
  environment values;
- verify a source-contract test for all fixed launcher settings;
- require exactly one currently free GPU;
- launch in a persistent `tmux` session and record the Julia PID and log paths.

Do not stop, replace, or alter the training after launch unless the user asks to
stop that exact run. If it fails, preserve its directory, checkpoints, and logs;
do not silently resume it or start a replacement.

## Validation and Completion

At startup, require the metadata to record the requested lattice, kappa,
`num_eig=24`, PT-replacement tail, 320/20 gauge counts, learning rate, and 4000
steps. Check the first finite training loss and the first finite test loss
without stopping the run.

After natural completion, require:

- checkpoints at 1000, 2000, 3000, and 4000 steps;
- a separate final checkpoint;
- `opt_t=4000` in both the step-4000 and final checkpoints;
- the PT-replacement layer stack and absence of `BlockScalar`;
- finite recorded training and test losses through step 4000;
- no surviving training process;
- preserved logs and a checksum/provenance summary.

Only after these checks pass may the new checkpoint be paired with the existing
corrected-Adam `BlockScalar eig24` model for the `24^3 x 48` evaluation.
