# Beta 6.2, 36-Cube by 48-Time Gauge Ensemble Design

## Purpose

Generate a small gauge ensemble for testing volume generalization of models trained on the established `16^3 x 32` ensemble. The new ensemble contains exactly 30 configurations on a `36^3 x 48` lattice at beta 6.2, saved in both Gaugefields JLD2 format and the existing LMP raw `.gbin` format.

## Reused, Validated Path

Use the completed beta 6.1/6.2 ensemble implementation as the source of truth:

- Driver: `/data1/choi/heatbath_beta61_beta62_20260804/generate_ensemble.jl`
- Helper module: `/data1/choi/heatbath_beta61_beta62_20260804/HeatbathExperiment.jl`
- Driver SHA-256: `8ae71ec186865004d4a32202a5fe5072d45020ef25f75cacb558f9b3ac819edc`
- Helper SHA-256: `45d2cf475617e5301120869f5427802fdc4fe2dae4b2e5e3557f831bd4ab2645`
- Raw conversion implementation: `LMP.Layer_sequence.convert_gauge_paths_to_raw`

Copy the driver and helper into a new, self-contained experiment directory. Do not modify the validated source files or the existing beta 6.0, 6.1, or 6.2 ensembles.

The existing driver cannot be executed unchanged because it hardcodes lattice `(16, 16, 16, 32)` and requires exactly 100 configurations. The experiment-local copy changes only the requested lattice, configuration count, output naming, and matching validation contract.

## Fixed Scientific Parameters

- Gauge group: SU(3)
- Action: Wilson plaquette action
- Action coefficient: `beta / 2`
- Start: hot
- Beta: `6.2`
- Lattice: `36 x 36 x 36 x 48`
- Deterministic seed: `620364820260903`
- Thermalization: 5000 heatbath sweeps
- Plaquette autocorrelation measurement: 500 sweeps
- Separation rule: `max(1, ceil(2 * tau_int_plaquette))`
- Autocorrelation window rule: self-consistent window with `c=5`
- Saved configurations: exactly 30, with IDs 0 through 29

No scientific parameter may be changed after launch without explicit user approval.

## Output Layout

Use the new, non-overwriting root:

`/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903`

The final layout is:

```text
heatbath_ensemble_beta6.2_36cube48time_20260903/
  HeatbathExperiment.jl
  generate_ensemble.jl
  test_driver.jl
  README.md
  parameters.toml
  plaquette_thermalization.csv
  plaquette_measurement.csv
  plaquette_production.csv
  autocorrelation.csv
  checksums.sha256
  raw_checksums.sha256
  run_summary.txt
  generation.log
  conversion_validation.log
  config_0.jld2
  ...
  config_29.jld2
  raw_gauge/
    config_0.gbin
    ...
    config_29.gbin
```

The unpadded IDs and `raw_gauge` subdirectory match the established beta 6.0 layout used by current training and solver scripts.

## Generation Flow

1. Check that the output root does not already exist, no conflicting generator is running, and `/data1` has enough free space.
2. Copy the validated driver and helper byte-for-byte into the new root, then minimally patch only the experiment-local copies.
3. Run the existing helper tests plus a small-lattice smoke test of the 30-configuration contract. The smoke test uses a temporary directory and must not create production output.
4. Launch the production chain in a persistent tmux session using project `/home/choi/LMP/LMP`.
5. The driver performs a hot start, 5000 thermalization sweeps, 500 plaquette measurements, determines the separation, and atomically saves each JLD2 configuration after a save/load plaquette check.
6. Only after the JLD2 production run completes successfully, call the existing LMP raw conversion function for IDs 0 through 29.
7. Validate all JLD2 and raw files, then write separate SHA-256 manifests.

## Validation and Completion Criteria

The run is complete only when all of the following hold:

- `run_summary.txt` reports `status = COMPLETE` and 30 saved configurations.
- JLD2 IDs and raw IDs are exactly the contiguous range 0 through 29, with no missing or extra configuration files.
- Every file is nonempty and all recorded SHA-256 checksums verify.
- Every JLD2 file loads successfully.
- Every raw header reports NC=3, four directions, and lattice `36 x 36 x 36 x 48`.
- Raw payloads match the corresponding JLD2 gauge arrays exactly.
- Plaquettes are finite and JLD2 save/load plaquettes agree within `1e-12`.
- The final sampled gauge has maximum unitarity and determinant errors recorded in the summary.
- No temporary files remain.

## Failure and Safety Behavior

- Refuse to overwrite an existing output root or any existing configuration.
- Save JLD2 files through a temporary sibling and rename only after successful reload validation.
- Run raw conversion only after successful production completion; do not disguise a partial production run as complete.
- Preserve intentional partial output if the process exits early. Do not delete, resume, or replace it without user instruction.
- Do not stop or interfere with unrelated CPU or GPU experiments.

## Resource Estimate

The lattice volume is about 17.1 times the established `16^3 x 32` volume. The pair of JLD2 and raw files is expected to use roughly 110-120 GB for 30 configurations. With the established CPU implementation and unchanged 5000/500 sweep counts, wall time is expected to be approximately 3.5-4 days; this is an estimate, not a completion deadline.
