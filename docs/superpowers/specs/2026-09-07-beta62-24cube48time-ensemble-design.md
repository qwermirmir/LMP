# Beta 6.2, 24-Cube by 48-Time Gauge Ensemble Design

## Purpose

Generate the correctly sized gauge ensemble needed before evaluating volume generalization from the established `16^3 x 32` models. The target lattice is `24^3 x 48`, for which the existing BlockScalar model's 512 learned block weights can correspond to `block_r=6` in the later model experiment.

This document covers gauge generation only. Fermion-field transfer, model loss evaluation, kappa selection, and DD-alphaAMG comparisons will be designed separately after the gauge ensemble is complete.

## Reused Implementation

Reuse the completed, validated beta 6.1/6.2 heatbath implementation:

- Driver: `/data1/choi/heatbath_beta61_beta62_20260804/generate_ensemble.jl`
- Helper: `/data1/choi/heatbath_beta61_beta62_20260804/HeatbathExperiment.jl`
- Raw conversion: `LMP.Layer_sequence.convert_gauge_paths_to_raw`
- JLD2/raw validator pattern: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/convert_and_validate_raw.jl`

Create experiment-local copies. Do not modify the validated sources, the completed `36^3 x 48` ensemble, or any earlier ensemble.

The only scientific changes from the validated beta 6.2 run are the target lattice, deterministic seed, and requested count. The output naming follows the established beta 6.0 layout.

## Fixed Parameters

- Gauge group: SU(3)
- Action: Wilson plaquette action
- Action coefficient: `beta / 2`
- Start condition: hot
- Beta: `6.2`
- Lattice: `24 x 24 x 24 x 48`
- Seed: `620244820260907`
- Thermalization: 5000 heatbath sweeps
- Autocorrelation measurement: 500 sweeps
- Autocorrelation window: existing self-consistent rule with `c=5`
- Separation: `max(1, ceil(2 * tau_int_plaquette))`
- Saved configurations: exactly 30, IDs 0 through 29

No parameter may change after launch without explicit user approval.

## Output

Use the new non-overwriting root:

`/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907`

Store:

- `config_0.jld2` through `config_29.jld2`
- `raw_gauge/config_0.gbin` through `raw_gauge/config_29.gbin`
- generation and conversion-validation logs
- parameter metadata
- thermalization, measurement, autocorrelation, and production plaquette CSV files
- separate JLD2 and raw SHA-256 manifests
- final run summary
- experiment-local source and tests

## Execution Flow

1. Require the target root to be absent and check disk, RAM, process, and tmux state.
2. Copy the validated driver and helper, then modify only lattice, count, safe-root handling, and compatible filenames.
3. Adapt the already-tested 30-configuration contract and raw round-trip tests; run red-green verification.
4. Run the production chain in a persistent CPU tmux session using `/home/choi/LMP/LMP` as the Julia project.
5. Atomically save and reload-validate every JLD2 configuration.
6. Start raw conversion only after successful JLD2 completion.
7. Compare every raw header and payload against its JLD2 source and write checksums.

## Safety and Completion

- Refuse to overwrite any existing production entry.
- Preserve partial intentional output if the run exits early; never delete, resume, or resume it without user instruction.
- Do not stop or modify unrelated experiments.
- Keep the completed but incorrectly sized `36^3 x 48` ensemble unchanged.
- Report completion only after 30 contiguous JLD2 files, 30 contiguous raw files, both checksum manifests, exact payload equality, finite plaquettes, recorded SU(3) errors, and zero temporary files are independently verified.

## Resource Estimate

The `24^3 x 48` lattice is 5.0625 times the `16^3 x 32` volume and about 29.6% of the completed `36^3 x 48` volume. Based on the observed large-volume sweep rate, production plus conversion and validation is expected to take approximately 27-30 hours. This is an estimate, not a timeout.
