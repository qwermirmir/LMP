# 24-Cube-by-48-Time LMP-versus-V-cycle Quick Reproduction Design

## Purpose

Reproduce the established two-panel LMP-versus-V-cycle DD-alphaAMG
application-count experiment on one gauge from the new `24^3 x 48`, beta 6.2
ensemble. This is a quick feasibility reproduction, not a new critical-kappa
determination and not an ensemble average.

The comparison asks whether the archived `Nv=12,24,48,64` learned networks,
trained on beta 6.0 gauges at `16^3 x 32`, remain useful when both the ensemble
and volume change. The output has the same scientific axes and marker mapping as
the supplied reference figure: fine Dirac applications versus coarse operator
applications, with LMP and V-cycle points at an easy kappa and an approximately
critical kappa.

## Known-Good Source and Scope

Copy the execution flow from:

`/home/choi/LMP/scripts/experiments/compare_lmp_nv_vs_vcycle_nv_application_counts.jl`

The source result to reproduce is:

`/home/choi/LMP/results/tables/lmp_nv12_24_48_64_vs_vcycle_nv12_24_48_64_sap3_mr4_nvcycle6_tol1e10_maxiter200_k157_k15775_cfg401_jitwarmup_rngreset_20260907.csv`

Reuse the publication renderer:

`/home/choi/LMP/scripts/experiments/plot_lmp_nv_vs_vcycle_publication.py`

Do not add a new solver, setup algorithm, layer implementation, or counting
method. The only experiment-specific changes are the gauge/lattice, beta-6.2
kappas, volume-compatible BlockScalar partition, random fermion field, and
memory-bounded learned-vector inference.

## Fixed Gauge and Kappa Conditions

- Gauge: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/raw_gauge/config_0.gbin`
- Gauge beta: `6.2`
- Lattice: `24 x 24 x 24 x 48`
- CUDA blocks: `(4,4,4,4)`
- Easy point: `kappa=0.1527`
- Approximately critical point: `kappa=0.1535`

The quick reproduction uses `0.1535` as a preliminary near-critical value. It
is informed by accepted beta-6.2 estimates `0.1533530242` and `0.1536323075`
from separate `16^3 x 32` gauges, but it is not presented as a measured critical
kappa for this `24^3 x 48` gauge. A precise volume-specific critical-kappa solve
is explicitly outside this experiment.

## Compared Cases

At each kappa, run eight cases in the source ordering:

1. LMP `Nv=12`
2. LMP `Nv=24`
3. LMP `Nv=48`
4. LMP `Nv=64`
5. V-cycle `Nv=12`
6. V-cycle `Nv=24`
7. V-cycle `Nv=48`
8. V-cycle `Nv=64`

Use the four exact archived checkpoints already named by the source runner:

- `/data1/choi/LMP_path33_kappa0158_12eig_scratch_20260702/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(12eig)(320config)(r=2).jld2`
- `/data1/choi/LMP_path33_kappa0158_24eig_scratch_20260703/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(24eig)(320config)(r=2)_final.jld2`
- `/data1/choi/LMP_path33_kappa0158_48eig_scratch_20260704/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(48eig)(320config)(r=2)_final.jld2`
- `/data1/choi/LMP_path33_kappa0158_64eig_scratch_20260706/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(64eig)(320config)(r=2)_final.jld2`

Their BlockScalar weights have verified shapes `(512,12)`, `(512,24)`,
`(512,48)`, and `(512,64)`. Set `block_r=6` on `24^3 x 48`, which creates the
same 512 blocks and permits exact weight reuse. Require each loaded
`layer_36_W` to equal the corresponding checkpoint value.

## Random Fermion Field and Checkpoint Loading

The archived checkpoint fermion fields have `16^3 x 32` extent and cannot be
loaded into the new volume. For all four LMP cases:

- create result-local derived checkpoints that omit only keys matching
  `layer_*_fermion_*`;
- reuse `create_nonfermion_checkpoint` and
  `validate_nonfermion_checkpoint` from the validated eig1 volume-loss
  experiment;
- preserve every non-fermion checkpoint key and value exactly;
- initialize the live `24^3 x 48` fermion field with CPU and CUDA seed
  `20260831`;
- require the same recorded SHA-256 field digest in every LMP case;
- never modify the archived source checkpoints.

The same-volume ablation already established that the random fermion field does
not degrade the learned setup. This experiment therefore does not attempt to
interpolate or tile the archived field.

## Memory-Bounded Learned-Vector Inference

Ordinary `layers[1:end-1]([U])` retains forward caches and is unsafe at the new
volume. The repository has a cache-clearing scalar-loss forward path, but no
existing inference helper that returns the final vector collection while
evicting preceding caches. Add only a local orchestration loop that calls the
existing layer forward methods in their original order and calls the existing
`LMP.Layer_sequence._clear_layer_forward_cache!` after each consumed
activation. It must not reimplement layer mathematics.

Keep the returned `Nv` vectors alive for `setup_DDalphaAMG`, then release model
layers and temporary caches before the solve. Run cases sequentially and reclaim
GPU memory between cases. If the `Nv=64` case exceeds device memory, preserve all
completed data and report the feasibility failure; do not lower `Nv`, change the
volume, or replace the checkpoint without approval.

## DD-alphaAMG and Solve Settings

Preserve the known-successful source settings exactly:

- setup mode: `direct_near_null`
- aggregation block size: `4`
- external-vector smoothing: enabled, 16 MR steps
- chirality: enabled
- V-cycle setup iterations: `6`
- setup smoother: SAP with 3 iterations
- setup block solver: MR with 4 steps
- V-cycle post smoother: SAP with 1 iteration and MR(4) block solves
- solve tolerance: `1e-10`
- solve maximum iterations: `200`
- restart: `50`
- solve smoother: MR with 4 pre- and 4 post-smoothing steps
- coarse tolerance: `0.1`
- coarse maximum iterations/restart: `50/20`
- coarse odd-even: disabled
- coarse GMRES mode: `restart_boundary`
- SAP odd-even: disabled
- SAP schedule: `twocolor`
- deflated initial guess: enabled for LMP only, as in the source
- normalized RHS seed: `42 + round(1_000_000*kappa)`; create one RHS per kappa
  and reuse that identical field for every case at that kappa

Count fine and coarse applications only through the existing verbose counters
emitted by `solve_DDalphaAMG`. Recompute and record the true relative residual.
A case is converged only when its residual is at most `1e-10`.

## Warm-up and Timing

Run the existing unrecorded first-case warm-up before the first measured point,
discard its counts and timings, then recreate the RHS and RNG state exactly as
the source runner does. Synchronize CUDA around setup and solve timings. The
recorded `solve_seconds` must exclude the discarded warm-up/JIT work.

The application counts are the primary metric. Timing annotations are retained
to reproduce the reference plot but are interpreted as secondary diagnostics.

## Outputs

Use a dedicated non-overwriting experiment root:

`/home/choi/LMP/results/experiments/24cube48time_lmp_vcycle_quick_reproduction_beta62_k1527_k1535_cfg0_20260908`

Use a new non-overwriting result ID under the established locations:

- table: `/home/choi/LMP/results/tables/`
- per-case logs: `/home/choi/LMP/results/logs/<result-id>/`
- raw plot: `/home/choi/LMP/results/plots/amg/`
- publication PNG and PDF: `/home/choi/LMP/results/plots/amg/`
- result-local derived checkpoints and provenance: the dedicated experiment
  root above

The CSV must contain exactly 16 data rows: two kappas, two methods, and four Nv
values. Each row records fine, coarse, total applications, setup/solve time,
relative residual, checkpoint, gauge path, and case log. The publication figure
uses the supplied reference style, including distinct Nv markers, method colors,
log-log axes, convergence annotations, and no diagonal guide lines.

## Preflight, Validation, and Failure Behavior

Before production:

- require the gauge and all four checkpoints to exist;
- verify the raw gauge header is `24 x 24 x 24 x 48`;
- verify the four BlockScalar weight shapes and exact post-load equality;
- verify every derived checkpoint removes only fermion keys;
- verify identical random-field digests across LMP cases;
- run source-contract tests for lattice, kappas, Nv matrix, checkpoint paths,
  solver settings, warm-up, and non-overwrite behavior;
- launch only on one currently free GPU in tmux.

Append each completed row and case log immediately. On any error, non-finite
residual, or out-of-memory condition, preserve the partial intentional run and
stop without changing experimental values. Never interrupt another run or an
unrelated GPU process.

Completion requires 16 recorded rows, all expected case keys exactly once,
matching gauge/checkpoint provenance, finite positive counts and timings,
residual-based convergence labels, an internally consistent CSV, a readable
publication PNG/PDF, and no surviving experiment process.
