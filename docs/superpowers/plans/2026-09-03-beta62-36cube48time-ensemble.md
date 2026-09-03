# Beta 6.2, 36-Cube by 48-Time Gauge Ensemble Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and validate exactly 30 SU(3) Wilson heatbath configurations at beta 6.2 on a `36 x 36 x 36 x 48` lattice in JLD2 and LMP raw formats.

**Architecture:** Copy the validated beta 6.1/6.2 driver and helper into a self-contained experiment root, changing only lattice, count, safe root handling, and beta-6.0-compatible filenames. Run production in tmux; only after JLD2 completion, call `LMP.Layer_sequence.convert_gauge_paths_to_raw` and validate every raw payload against its JLD2 source.

**Tech Stack:** Julia 1.11.9, Gaugefields 0.7.1, JLD2, LMP, SHA-256, tmux.

## Global Constraints

- Output root: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903`.
- SU(3), Wilson plaquette action, coefficient `beta / 2`, hot start, beta `6.2`, seed `620364820260903`.
- Lattice `36 x 36 x 36 x 48`, 5000 thermalization sweeps, 500 measurement sweeps, exactly 30 configurations.
- Separation `max(1, ceil(2 * tau_int_plaquette))` with the existing self-consistent window.
- IDs 0 through 29 as root `config_N.jld2` and `raw_gauge/config_N.gbin`.
- Never overwrite prior ensembles or production data; preserve intentional partial output.
- Do not alter scientific parameters or stop the run without explicit user instruction.

---

### Task 1: Self-Contained Driver and Contract Tests

**Files:**
- Copy: `/data1/choi/heatbath_beta61_beta62_20260804/HeatbathExperiment.jl`
- Copy: `/data1/choi/heatbath_beta61_beta62_20260804/generate_ensemble.jl`
- Copy and modify: `/data1/choi/heatbath_beta61_beta62_20260804/test_driver.jl`
- Create: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/README.md`

**Interfaces:**
- Consumes: validated `ExperimentConfig`, `initialize_chain`, and `atomic_save_configuration`.
- Produces: an experiment-local driver accepting `BETA SEED OUTPUT_DIR`, fixed to the approved lattice and count.

- [ ] **Step 1: Run production preflight**

Run `test ! -e` on the exact target, `pgrep -af 'generate_ensemble.jl.*36cube48time'`, and `df -h /data1`. Require an absent target, no conflicting generator, and at least 150 GB free.

- [ ] **Step 2: Copy sources and verify their original hashes**

Before patching, require:

```text
generate_ensemble.jl  8ae71ec186865004d4a32202a5fe5072d45020ef25f75cacb558f9b3ac819edc
HeatbathExperiment.jl 45d2cf475617e5301120869f5427802fdc4fe2dae4b2e5e3557f831bd4ab2645
```

- [ ] **Step 3: Write the failing count-contract test**

Change copied valid fixtures to `n_conf=30`, require 30 to pass, and add:

```julia
@test_throws ArgumentError validate_config(ExperimentConfig(
    beta=6.2, seed=1, output_dir="/tmp/unused",
    lattice=(2, 2, 2, 2), n_therm=1, n_measure=2, n_conf=100,
))
```

- [ ] **Step 4: Verify the new test fails**

Run `julia --project=/home/choi/LMP/LMP /data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/test_driver.jl`. Expected: the copied helper rejects 30 and still accepts 100.

- [ ] **Step 5: Patch only the helper count contract**

Use:

```julia
config.n_conf == 30 || throw(ArgumentError("this experiment requires exactly 30 configurations"))
```

- [ ] **Step 6: Patch the production constants and layout**

Set exactly:

```julia
lattice=(36, 36, 36, 48),
n_therm=5000,
n_measure=500,
n_conf=30,
```

Use `CONFIG_DIR = OUTPUT_DIR`, filename `"config_$(config_id).jld2"`, and root-relative checksum names.

Permit only the self-contained sources, `README.md`, `test_driver.jl`, `convert_and_validate_raw.jl`, and launcher-created `generation.log` before production. Reject any other existing root entry:

```julia
const ALLOWED_PREEXISTING = Set([
    "HeatbathExperiment.jl", "generate_ensemble.jl", "test_driver.jl",
    "convert_and_validate_raw.jl", "README.md", "generation.log",
])
if ispath(OUTPUT_DIR)
    isdir(OUTPUT_DIR) || error("output path is not a directory: $OUTPUT_DIR")
    unexpected = sort(setdiff(readdir(OUTPUT_DIR), ALLOWED_PREEXISTING))
    isempty(unexpected) || error("refusing existing production entries: $(join(unexpected, ", "))")
else
    mkpath(OUTPUT_DIR)
end
```

- [ ] **Step 7: Run tests and static diff checks**

Require copied helper tests to pass and the driver diff to contain only the approved count, lattice, layout, and allowlist changes.

### Task 2: Raw Conversion and Full-Payload Validation

**Files:**
- Create: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/convert_and_validate_raw.jl`
- Modify: `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/test_driver.jl`

**Interfaces:**
- Consumes: 30 root JLD2 files, `run_summary.txt`, and existing `convert_gauge_paths_to_raw`.
- Produces: 30 raw files, `raw_checksums.sha256`, and exit-zero full-payload validation.

- [ ] **Step 1: Write a failing small-lattice raw round-trip test**

Save one cold `2^4` JLD2 gauge, call the existing raw converter, then call the not-yet-defined `validate_raw_matches_jld2`. Expected: `UndefVarError`.

- [ ] **Step 2: Implement the minimum validator**

```julia
function validate_raw_matches_jld2(jld2_path::String, raw_path::String)
    U = JLD2.load(jld2_path, "U")
    open(raw_path, "r") do io
        dims = LMP.Layer_sequence._read_raw_header!(io, U)
        expected = [(Int(u.NC), Int(u.NX), Int(u.NY), Int(u.NZ), Int(u.NT)) for u in U]
        dims == expected || error("raw dimensions differ from JLD2")
        for mu in eachindex(U)
            payload = Array{ComplexF64}(undef, size(U[mu].U))
            read!(io, payload)
            payload == U[mu].U || error("raw payload mismatch at direction $mu")
        end
        eof(io) || error("raw file has trailing bytes")
    end
    return true
end
```

The main path must require `status = COMPLETE`, exactly 30 contiguous JLD2 paths, no existing raw outputs, then convert, validate all pairs, and record SHA-256 values.

- [ ] **Step 3: Run the complete small-lattice tests**

Run `julia --project=/home/choi/LMP/LMP /data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903/test_driver.jl`. Require all helper and raw round-trip tests to pass without creating production data.

### Task 3: Persistent Production Launch

**Files:**
- Runtime: `generation.log`
- After generation: `conversion_validation.log`

**Interfaces:**
- Consumes: exact beta, seed, output root, and tested local sources.
- Produces: a persistent CPU generation phase followed conditionally by raw conversion.

- [ ] **Step 1: Re-run preflight**

Check free disk/RAM, active generators, tmux sessions, source hashes/diffs, test output, and absence of production files.

- [ ] **Step 2: Launch tmux session**

Use `heatbath_beta62_36cube48time_30_20260903`. Run the production driver with beta `6.2`, seed `620364820260903`, and output `/data1/choi/heatbath_ensemble_beta6.2_36cube48time_20260903`, redirecting to `generation.log`; chain raw conversion with `&&` so it runs only after successful JLD2 completion and writes `conversion_validation.log`.

- [ ] **Step 3: Verify live initialization**

Require a live Julia PID and metadata showing the exact beta, lattice, seed, count, hot-start plaquette, and first thermalization progress. Do not stop or change the run.

### Task 4: Completion Audit

**Files:**
- Read: all outputs and manifests under the experiment root.

**Interfaces:**
- Consumes: naturally completed generation and conversion phases.
- Produces: evidence-backed completion status and ensemble statistics.

- [ ] **Step 1: Wait for natural completion**

On status requests inspect process state, latest sweep/configuration, and errors without interrupting or replacing the run.

- [ ] **Step 2: Verify JLD2 completion**

Require `status = COMPLETE`, exactly IDs 0-29, 30 CSV rows, valid checksums, finite plaquettes, loadable files, and recorded group errors.

- [ ] **Step 3: Verify raw completion**

Require exactly IDs 0-29, valid raw checksums, `36 x 36 x 36 x 48` headers, exact payload equality for all pairs, no temporary files, and exit-zero conversion validation.
