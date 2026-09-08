# Eig1 Volume-Generalization Ritz-Loss Comparison Design

## Purpose

Evaluate whether the two established eig1 architectures retain their learned gauge-processing performance when transferred from the `16^3 x 32`, beta 6.0 training distribution to the new `24^3 x 48`, beta 6.2 gauge ensemble. This first experiment compares test Ritz loss only; it does not train models or run DD-alphaAMG.

The result will determine whether retraining the two architectures with the corrected Adam implementation is worth doing before the solver comparison.

## Compared Models

Use the existing matched pair trained at kappa 0.158 for 16,000 steps:

1. BlockScalar model
   - Architecture: 32 PT body layers, eig1 output PT layer, BlockScalar layer, output normalization, Ritz loss.
   - Checkpoint: `/data1/choi/LMP_path33_kappa0158_1eig_scratch_20260807/(16x32)32PTlayer33path(kappa=0.158)(16000epoch)(maxlen5)(xavier_init)(1eig)(320config)(r=2)_final.jld2`
   - Original BlockScalar shape: `(512, 1)` with `block_r=4` on `16^3 x 32`.
   - Evaluation BlockScalar: `block_r=6` on `24^3 x 48`, which also gives 512 blocks and permits exact weight reuse.

2. No-BlockScalar model
   - Architecture: replace the BlockScalar layer with one additional 33-wide PT layer, followed by the eig1 output PT layer, output normalization, and Ritz loss.
   - Checkpoint: `/data1/choi/LMP_path33_kappa0158_1eig_no_blockscalar_plus1pt_scratch_20260813/(16x32)33PTlayer33path(kappa=0.158)(16000epoch)(maxlen5)(xavier_init)(1eig)(320config)(no_blockscalar_plus1pt)_final.jld2`

Both checkpoints predate the Adam global-step correction and record `opt_t=560000`. This experiment compares the two archived architectures as they were trained; it does not attribute their absolute quality to the corrected optimizer.

## Gauge and Physics Conditions

- Gauge root: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/raw_gauge`
- Gauge files: `config_0.gbin` through `config_29.gbin`
- Gauge lattice: `24 x 24 x 24 x 48`
- Gauge beta: `6.2`
- Wilson Dirac kappa: common fixed value `0.1535`
- Ritz objective: the scalar eig1 Ritz loss produced by the existing `Ritz_Layer`, i.e. the Rayleigh quotient of `DdagD` for the normalized model output.

The fixed kappa is a preliminary value informed by prior beta 6.2 critical-kappa estimates on `16^3 x 32`. It is not claimed to be the exact critical kappa of the new volume.

## Random Fermion Field

The checkpoint fermion fields are tied to `16^3 x 32` and cannot be loaded into a `24^3 x 48` field. For both models:

- omit all checkpoint keys matching `layer_*_fermion_*`;
- initialize a `24^3 x 48` Gaussian pseudofermion field with CPU and CUDA seed `20260831`;
- reuse exactly the same field values for both models and all 30 gauge configurations;
- record a SHA-256 digest of the canonical field payload;
- verify the digest before and after every configuration evaluation.

This choice is supported by the earlier same-volume random-fermion ablation. No learned-field interpolation or tiling is introduced in this experiment.

## Derived Checkpoints

Create result-local derived checkpoints containing every original key except fermion-field values, fermion optimizer moments, and fermion layout metadata. Do not modify the archived source checkpoints.

For each derived checkpoint:

- record source and derived SHA-256 hashes;
- record the complete removed-key list;
- require every retained key and value to match the source exactly;
- require no `layer_*_fermion_*` key to remain;
- load the derived checkpoint through the existing `load_checkpoint!` implementation;
- fail if any non-fermion parameter is skipped or has an incompatible shape.

The core checkpoint loader is not modified.

## Memory-Safe Forward Evaluation

The ordinary `layers([U])` path retains per-layer outputs and is expected to exceed an 80 GB GPU at this volume. Reuse the existing `LMP.Layer_sequence._checkpointed_forward_loss!` path with only the input and final activation retained. It clears completed layer caches as the forward pass advances and evaluates the existing Ritz layer without implementing a second forward algorithm.

Evaluate models sequentially, never keeping both full model cache sets on the GPU at the same time:

1. Build and validate the BlockScalar architecture, warm one streaming forward, evaluate configurations 0-29, then release all model caches and reclaim GPU memory.
2. Recreate the identical random field, verify its digest, build and validate the no-BlockScalar architecture, warm one streaming forward, and evaluate configurations 0-29.

Compilation and warm-up timings are not scientific results. Loss values are recorded only after successful warm-up.

## Outputs

Use a new non-overwriting result root:

`/home/choi/LMP/results/experiments/eig1_volume_generalization_ritz_loss_24cube48time_beta62_k1535_20260908`

Produce:

- `README.md` with purpose, conditions, limitations, and exact commands;
- `settings.txt` with checkpoint/gauge hashes, seed, random-field digest, layer signatures, BlockScalar shape, Julia/CUDA information, and source revision;
- two derived non-fermion checkpoints plus their provenance;
- an append-only `losses.csv` with one row per model and configuration;
- per-model logs and a launcher log;
- `ritz_loss_comparison.png` showing paired individual points for configurations 0-29;
- a final validation report.

The CSV records model, variant, configuration ID, gauge path and hash, kappa, random-field seed and digest, Ritz loss, and evaluation status. Evaluation time may be logged diagnostically but is not used as a performance metric.

The plot uses one x position per configuration, a connecting line between the two model losses, distinct markers for the architectures, and no mean, error bar, or standard-deviation summary.

## Existing-Code Reuse

Reuse these known-good paths:

- random-field seeding, digest, and Ritz evaluation pattern: `scripts/experiments/evaluate_eig12_fermion_field_loss_ablation.jl`;
- eig1 architecture constructors and checkpoint paths: `scripts/experiments/evaluate_eig1_absolute_residual_blockscalar_vs_plus1pt.jl`;
- gauge loading: `LMP.Layer_sequence._make_gauge_staging`, `_make_raw_cpu_buffers`, and `_read_raw_gauge_to_gpu!`;
- memory-bounded forward: `LMP.Layer_sequence._checkpointed_forward_loss!` and `_clear_forward_caches!`;
- plotting style: paired individual points from the existing fermion-field loss and eig1 residual comparison scripts.

The only new orchestration capability is filtering dimension-bound fermion keys into derived checkpoints and validating retained checkpoint contents. No solver, layer, optimizer, or core model function is reimplemented.

## Verification and Failure Behavior

Before production, run a one-configuration smoke test for each model and require:

- the raw header matches `24 x 24 x 24 x 48`;
- the filtered checkpoint has no fermion keys and all retained values match its source;
- the target architecture and layer ordering exactly match the source experiment;
- the BlockScalar target and checkpoint weights are both `(512, 1)` and equal after load;
- the no-BlockScalar model contains the additional PT layer and no BlockScalar layer;
- both models have the same random-field digest;
- streaming loss is finite and positive;
- no checkpoint weight-skip warning occurs;
- peak GPU memory remains below the device capacity.

Production refuses to overwrite existing outputs and appends each completed CSV row immediately. If a model or configuration fails, preserve completed rows and logs, mark the failure explicitly, and stop without changing the lattice, kappa, field seed, checkpoint, or evaluation method. An out-of-memory result is reported as a feasibility failure; it does not authorize a smaller volume or altered architecture.

Completion requires exactly 60 successful loss rows, 30 paired configuration IDs, unchanged random-field digests, valid source and gauge hashes, finite positive losses, a readable PNG, and a validation report with every check marked PASS.
