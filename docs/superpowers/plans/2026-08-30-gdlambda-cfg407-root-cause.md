# GD-λAMG cfg407 Root-Cause Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify the first GD-λAMG stage at which the learned-refined cfg407 run loses the target low-mode branch, without changing the numerical algorithm.

**Architecture:** Add a default-no-op trace callback to the existing generalized Davidson routine and emit snapshots of the complete harmonic Ritz candidate set around extraction, locking, and restart. A dedicated cfg407 driver writes the trace into a new result directory and an analysis script classifies the first divergence against the reference eigenpairs already produced by the Standard run.

**Tech Stack:** Julia 1.11, LMP, CUDA.jl, DelimitedFiles, existing GD-λAMG test harness.

## Global Constraints

- Preserve κ=0.157, cfg407, 24 initial vectors, checkpoint, tolerances, seeds, subspace sizes, and solver iteration limits from the failed main experiment.
- Do not alter candidate ordering, locking, restart, correction equations, or AMG setup behavior.
- Do not overwrite or delete any existing experiment output.
- Do not stop any running experiment.

---

### Task 1: Trace callback contract

**Files:**
- Modify: `test/gdlambda_amg.jl`
- Modify: `src/GDlambdaAMG.jl`

**Interfaces:**
- Consumes: `generalized_davidson(apply_Q!, initial_block; ...)`
- Produces: keyword `diagnostic_callback=(event)->nothing`; event fields `stage`, `expansion`, `restarts`, `locked_values`, and `candidates`.

- [ ] **Step 1: Write a failing deterministic dense test**

Add a diagonal Hermitian problem that collects callback events and asserts the presence of extraction, lock, pre-restart, and post-restart snapshots. Run the same problem without a callback and assert identical eigenvalues, residuals, expansion count, restart count, and application count.

- [ ] **Step 2: Run the focused test and confirm failure**

Run: `julia --project=. test/gdlambda_amg.jl`

Expected: failure because `diagnostic_callback` is not accepted.

- [ ] **Step 3: Implement the no-op callback and immutable snapshots**

Add the optional keyword and invoke it only after already-computed harmonic extractions and at lock/restart boundaries. Copy scalar values and coefficient-derived diagnostics into CPU `NamedTuple`/`Vector` objects; do not call `apply_Q!` or mutate basis vectors from the callback path.

- [ ] **Step 4: Run the focused and stress tests**

Run:

```bash
julia --project=. test/gdlambda_amg.jl
julia --project=. test/gdlambda_100lock_stress.jl
```

Expected: all tests pass and callback-enabled/disabled runs have identical numerical counters.

### Task 2: cfg407 diagnostic driver

**Files:**
- Create: `/home/choi/LMP/scripts/experiments/diagnose_gdlambda_cfg407_ritz_flow.jl`

**Interfaces:**
- Consumes: the existing experiment setup helpers and `diagnostic_callback`.
- Produces: `ritz_trace.csv`, `summary.csv`, `eigenpair_audit.csv`, `history.csv`, and `settings.txt` in a caller-supplied new directory.

- [ ] **Step 1: Add a driver smoke-mode test path**

Support `GD_DIAGNOSTIC_SMOKE=1`, which executes the same setup but limits only the diagnostic replay after verifying that all production settings were loaded. The production run omits this variable.

- [ ] **Step 2: Implement trace serialization**

Write one row per candidate with configuration, case, stage, expansion, restart, locked count, rank, value, residual, sign, coefficient norm, and locked overlap. Flush at each event so partial intentional results survive interruption.

- [ ] **Step 3: Run a GPU smoke test in a new directory**

Use GPU1 if free. Verify nonempty trace data, both signs where present, monotone event counters, and no change to the ordinary history schema.

### Task 3: Exact cfg407 replay and classification

**Files:**
- Create: `/home/choi/LMP/scripts/experiments/analyze_gdlambda_cfg407_ritz_flow.jl`

**Interfaces:**
- Consumes: `ritz_trace.csv`, `summary.csv`, `eigenpair_audit.csv`, and the preserved main-run cfg407 audit.
- Produces: `root_cause_report.txt` with an evidence-backed classification.

- [ ] **Step 1: Start the exact replay in a new directory**

Run Standard AMG-24 and learned-refined AMG-24 on cfg407 using the exact failed-main settings and seed on an available GPU. Preserve the run even if the learned case reaches the existing expansion cap.

- [ ] **Step 2: Analyze the first divergence**

Locate the earliest extraction/lock/restart event where the reference low-mode branch is absent or ceases improving. Classify only when the trace distinguishes initial-subspace, extraction, locking, restart, or correction failure.

- [ ] **Step 3: Verify the report against raw CSV rows**

Check reported values, expansion indices, restart indices, residuals, and signs directly against the trace. If no stage is uniquely supported, report `inconclusive` and the missing observable.

- [ ] **Step 4: Run final regression verification**

Run the focused GD-λAMG and 100-lock stress suites again, confirm no experiment is still running unexpectedly, and link the trace and report.
