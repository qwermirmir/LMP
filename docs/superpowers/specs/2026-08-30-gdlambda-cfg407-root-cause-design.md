# GD-λAMG cfg407 Root-Cause Diagnostic Design

## Objective

Determine exactly where the learned-refined cfg407 run loses the ability to
converge the second eigenpair, without changing the GD-λAMG numerical path or
overwriting any existing experiment.

## Scope

The diagnostic covers the cfg407 Standard AMG-24 and learned-refined AMG-24
runs with the same seeds and solver settings used by the failed main
experiment. It records information that the existing `history.csv` omits:
the complete harmonic Ritz candidate set, candidate residuals and signs,
locking overlap, and restart retention. It also compares the locked and active
candidates with an independently computed reference set of eigenvalues closest
to zero.

The diagnostic does not alter kappa, configuration ID, tolerances, vector
count, checkpoint, solver iteration limits, candidate ordering, locking,
restart, or correction equations.

## Instrumentation Architecture

Add an optional no-op diagnostic callback to `generalized_davidson`. At four
state boundaries it receives immutable scalar/dense diagnostic data:

1. harmonic extraction before candidate selection;
2. immediately after a converged candidate is locked;
3. immediately before thick restart;
4. immediately after the restarted basis is constructed and re-extracted.

Production callers that omit the callback retain the existing execution path.
The callback writes a separate CSV in a new cfg407 diagnostic result directory.
It records expansion, restart, locked count, event, candidate rank, Ritz value,
direct residual, coefficient norm, and overlap with the locked space. Aggregate
rows record positive/negative candidate counts and the retained set.

## Reference and Classification

Use the repository's previously validated reference eigenvalue workflow on the
same cfg407 Hermitian operator. The diagnostic classifies the first divergence
as one of:

- `initial_subspace`: required negative low modes are absent before the first
  correction;
- `harmonic_extraction`: the projected pencil misorders a represented target
  mode;
- `locking`: the target direction disappears immediately after locking;
- `restart`: it exists before restart but is absent afterward;
- `correction`: it remains represented across restarts, but correction steps do
  not reduce its residual.

A classification is accepted only when the trace identifies the first event
where a reference low mode changes from represented to absent or ceases to
improve. If the available observables cannot distinguish two cases, the result
is reported as inconclusive rather than guessed.

## Verification

Dense unit tests verify that enabling the callback does not change returned
eigenvalues, residuals, expansion counts, restart counts, or application counts.
A restart test verifies that logged candidate order and retained candidates
match the actual harmonic Ritz ordering. The existing GD-λAMG test suites must
continue to pass.

The GPU diagnostic uses a new directory, preserves all partial data, and runs
Standard and learned-refined cases with identical cfg407 settings. No run is
stopped or replaced without explicit user instruction.

## Deliverables

- optional GD-λAMG diagnostic callback and unit tests;
- cfg407 full candidate/restart trace CSV;
- independent reference eigenvalue comparison;
- a concise root-cause report naming the first failing stage and its evidence.
