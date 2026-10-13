# Beta 6.2, 24-Cube by 48-Time Gauge Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and validate exactly 30 SU(3) Wilson heatbath configurations at beta 6.2 on a `24 x 24 x 24 x 48` lattice in JLD2 and LMP raw formats.

**Architecture:** Reuse the completed `36^3 x 48` experiment-local generator, helper, tests, and raw validator byte-for-byte, then change only the generator lattice and experiment documentation. Run JLD2 production in persistent tmux and conditionally run the existing raw conversion and full-payload validation only after generation succeeds.

**Tech Stack:** Julia 1.11.9, Gaugefields 0.7.1, JLD2, LMP, SHA-256, tmux.

## Global Constraints

- Output root: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907`.
- SU(3), Wilson plaquette action, coefficient `beta / 2`, hot start, beta `6.2`, seed `620244820260907`.
- Lattice `24 x 24 x 24 x 48`, 5000 thermalization sweeps, 500 measurement sweeps, exactly 30 configurations.
- Separation `max(1, ceil(2 * tau_int_plaquette))` with the existing self-consistent window.
- IDs 0 through 29 as root `config_N.jld2` and `raw_gauge/config_N.gbin`.
- Preserve `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903` unchanged.
- Never overwrite production data or alter scientific parameters after launch.

---

### Task 1: Copy the Known-Good 30-Configuration Pipeline

**Files:**
- Copy: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/HeatbathExperiment.jl`
- Copy: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/generate_ensemble.jl`
- Copy: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/convert_and_validate_raw.jl`
- Copy and modify: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/test_driver.jl`
- Create: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/README.md`

**Interfaces:**
- Consumes: the completed 36-cube pipeline, whose helper tests and 30-file raw payload validation passed.
- Produces: a self-contained 24-cube pipeline with the same output and validation contract.

- [ ] **Step 1: Verify the target is absent and source is complete**

Require the exact target not to exist. Require the four source files, completed summary, 30 JLD2 files, and 30 raw files in the 36-cube source experiment.

- [ ] **Step 2: Copy the four implementation/test files**

Create the target root, copy the four files byte-for-byte, and record their pre-edit SHA-256 hashes. Do not copy any generated configuration or result file.

- [ ] **Step 3: Write the failing lattice contract test**

Append this source contract to the copied `test_driver.jl` before modifying the driver:

```julia
@testset "production lattice contract" begin
    source = read(joinpath(@__DIR__, "generate_ensemble.jl"), String)
    @test occursin("lattice=(24, 24, 24, 48)", source)
    @test !occursin("lattice=(36, 36, 36, 48)", source)
    @test occursin("n_therm=5000", source)
    @test occursin("n_measure=500", source)
    @test occursin("n_conf=30", source)
end
```

- [ ] **Step 4: Run the copied tests and verify RED**

Run:

```bash
julia --project=/home/choi/LMP/LMP \
  /data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/test_driver.jl
```

Expected: only the lattice contract fails because the copied source still says `(36, 36, 36, 48)`; helper and raw round-trip tests pass.

- [ ] **Step 5: Change only the lattice constant**

In the copied `generate_ensemble.jl`, replace:

```julia
lattice=(36, 36, 36, 48),
```

with:

```julia
lattice=(24, 24, 24, 48),
```

- [ ] **Step 6: Write experiment documentation**

Record purpose, exact parameters, source experiment and hashes, two-phase execution, overwrite protection, validation criteria, preserved 36-cube data, and the 27-30 hour estimate in `README.md`.

- [ ] **Step 7: Run tests and verify GREEN**

Re-run `test_driver.jl`. Require 22 helper tests, 2 raw round-trip tests, and 5 lattice contract tests to pass. Parse both production scripts and require no syntax error.

### Task 2: Preflight and Persistent Launch

**Files:**
- Runtime: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/generation.log`
- After generation: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/conversion_validation.log`

**Interfaces:**
- Consumes: beta `6.2`, seed `620244820260907`, output root, and tested local scripts.
- Produces: a persistent CPU generator followed conditionally by raw conversion and validation.

- [ ] **Step 1: Verify overwrite refusal**

Invoke the copied driver against a temporary directory containing an unexpected file. Require a nonzero exit with `refusing existing production entries` before gauge initialization.

- [ ] **Step 2: Run final resource and conflict checks**

Check `/data1` free space, RAM, active generators, tmux sessions, exact root entries, driver diff against the 36-cube source, and source hashes. Require no production entries in the target.

- [ ] **Step 3: Launch tmux session**

Use session `heatbath_beta62_24cube48time_30_20260907`. Run:

```bash
JULIA_NUM_THREADS=1 julia --project=/home/choi/LMP/LMP \
  /data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/generate_ensemble.jl \
  6.2 620244820260907 \
  /data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907 \
  > /data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/generation.log 2>&1
```

Chain the copied `convert_and_validate_raw.jl` with `&&`, using the same Julia project, and redirect it to `conversion_validation.log`.

- [ ] **Step 4: Verify live initialization**

Require the tmux session and Julia PID to remain alive. Require metadata to show beta 6.2, seed `620244820260907`, lattice `24x24x24x48`, 5000/500 sweeps, 30 configurations, a finite hot-start plaquette, and the first thermalization sweep. Do not stop or modify the run.

### Task 3: Natural Completion Audit

**Files:**
- Read: all output, logs, metadata, and manifests in the target root.

**Interfaces:**
- Consumes: naturally completed generation and raw-validation phases.
- Produces: evidence-backed completion status and ensemble statistics.

- [ ] **Step 1: Preserve and monitor the run**

On status requests inspect the live process, latest sweep/configuration, counts, and errors without interrupting or replacing the process.

- [ ] **Step 2: Verify JLD2 completion**

Require summary `COMPLETE`, exactly IDs 0-29, 30 production CSV rows, 500 measurement rows, valid checksums, finite plaquettes, successful loads, and recorded SU(3) errors.

- [ ] **Step 3: Verify raw completion**

Require exactly IDs 0-29, valid raw checksums, `24 x 24 x 24 x 48` headers, exact full-payload equality for all JLD2/raw pairs, no temporary files, and `RAW_CONVERSION_VALIDATION_COMPLETE count=30`.
