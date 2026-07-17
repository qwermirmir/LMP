# DD-alphaAMG LMP Seed Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare reference DD-alphaAMG adaptive setup using random initial vectors against the same setup using LMP-produced initial vectors.

**Architecture:** Add a seed-specific setup API to the local reference library while preserving the default setup path. Extend the existing Julia/C diagnostic bridge to load an LMP checkpoint, convert vectors to the validated reference layout, verify a round trip, and run matched random-seed and LMP-seed reference solves.

**Tech Stack:** C11, DDalphaAMG reference C library, Julia 1.11, CUDA.jl, LMP checkpoint loader.

## Global Constraints

- Do not stop or replace running experiments.
- Print all experimental settings before execution.
- Preserve gauge, kappa, RHS, Nv, setup iterations, and solver settings between comparison arms.
- Inject unsplit LMP prototype vectors and let reference DD-alphaAMG perform gamma5-compatible aggregation.
- Require seed-layout round-trip relative error below `1e-12` before solving.

---

### Task 1: Reference seeded-setup API

**Files:**
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/src/dd_alpha_amg.h`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/include/dd_alpha_amg.h`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/src/dd_alpha_amg.c`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/src/setup_generic.c`
- Test: `/home/choi/LMP/external/DDalphaAMG_nosse/seeded_setup_diag.c`

**Interfaces:**
- Consumes: consecutive public-layout fine vectors.
- Produces: `dd_alpha_amg_setup_with_seed_vectors(double *, int, int, int *)`.

- [ ] Write a diagnostic that fails to link because the seeded setup API is absent.
- [ ] Build it and confirm the expected undefined-symbol failure.
- [ ] Add the public declaration and implementation with vector-count validation.
- [ ] Make the non-null interpolation seed branch run the existing three smoother stages and normalization.
- [ ] Build the library and diagnostic, then confirm random and seeded setup both complete.

### Task 2: Seed round-trip verification

**Files:**
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/src/dd_alpha_amg.h`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/include/dd_alpha_amg.h`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/src/dd_alpha_amg.c`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/seeded_setup_diag.c`

**Interfaces:**
- Consumes: seeded setup input.
- Produces: a pre-smoothing seed dump for layout validation.

- [ ] Add a diagnostic assertion that fails until a pre-smoothing dump is available.
- [ ] Add a narrowly scoped seed dump hook used only by the diagnostic.
- [ ] Confirm byte-order/site/spin conversion round trips below `1e-12`.

### Task 3: LMP checkpoint comparison driver

**Files:**
- Create: `/home/choi/LMP/scripts/diagnostics/compare_lmp_seed_mrottmann.jl`
- Modify: `/home/choi/LMP/external/DDalphaAMG_nosse/seeded_setup_diag.c`

**Interfaces:**
- Consumes: `LMP_CHECKPOINT`, gauge path, kappa, Nv, and matched setup/solver settings.
- Produces: archived settings and metrics for random and LMP seed arms.

- [ ] Add a dry-run test that fails because learned-vector loading is not wired.
- [ ] Reuse the validated gauge, RHS, spin, and site conversion helpers.
- [ ] Load the checkpoint and evaluate exactly `Nv` unsplit LMP prototypes.
- [ ] Print and archive all settings before launching either arm.
- [ ] Run the round-trip gate, random baseline, and LMP-seeded reference solve.
- [ ] Record setup time, solve time, counters, reported residual, and Julia true residual.

### Task 4: Matched experiment

**Files:**
- Create: `/home/choi/LMP/results/diagnostics/lmp_seed_reference_<run-id>/settings.txt`
- Create: `/home/choi/LMP/results/diagnostics/lmp_seed_reference_<run-id>/summary.csv`
- Create: `/home/choi/LMP/results/diagnostics/lmp_seed_reference_<run-id>/run.log`

**Interfaces:**
- Consumes: the user-selected checkpoint and exact experimental settings.
- Produces: reproducible matched comparison results.

- [ ] Print the full settings block and verify both arms differ only in seed source.
- [ ] Execute the comparison without interrupting existing processes.
- [ ] Verify both final solutions with the Julia fine operator.
- [ ] Summarize setup and solve differences without claiming equivalence beyond measured configurations.
