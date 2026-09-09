# eig24 Volume-Generalization DD-alphaAMG Comparison Design

## Purpose

Compare the solver performance of the two corrected-Adam `eig24` architectures
and a conventional `V-cycle Nv=24` baseline on one gauge from the new
`24 x 24 x 24 x 48`, beta 6.2 ensemble.

The three cases are:

1. corrected-Adam `BlockScalar eig24`
2. corrected-Adam `No BlockScalar + 1 PTLayer eig24`
3. `V-cycle Nv=24`

This is a single-configuration feasibility comparison. It is not an ensemble
average and does not determine a new volume-specific critical kappa.

## Existing-Code-First Basis

Reuse the setup, solve, counting, residual, logging, and warm-up behavior from:

`/home/choi/LMP/scripts/experiments/compare_lmp_nv_vs_vcycle_nv_application_counts.jl`

Reuse the validated random-field and dimension-bound checkpoint filtering from:

`/home/choi/LMP/scripts/experiments/evaluate_eig1_volume_generalization_ritz_loss.jl`

Reuse the exact PT-replacement architecture established by:

`/home/choi/LMP/scripts/training/train_lmp.jl`

No solver, layer, optimizer, loss, or DD-alphaAMG routine may be replaced or
reimplemented. The only new code is experiment orchestration for two learned
architectures at a larger volume and a three-condition plot.

## Fixed Gauge and Physics Conditions

- gauge: `/data1/choi/heatbath_ensemble_beta6.2_24cube48time_20260907/raw_gauge/config_0.gbin`
- beta: 6.2
- lattice: `24 x 24 x 24 x 48`
- CUDA blocks: `(4,4,4,4)`
- kappa values: `0.1527` and `0.1535`
- near-critical label: `0.1535`

The `0.1535` value is only a quick near-critical approximation informed by
earlier beta-6.2 estimates on `16^3 x 32`. The plot must not claim it is a newly
measured critical kappa for `config_0` at the larger volume.

## Model Checkpoints

Use these final corrected-Adam checkpoints without modification:

- BlockScalar:
  `/data1/choi/LMP_path33_kappa0158_24eig_scratch_adamfix_20260902/(16x32)32PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(24eig)(320config)(r=2)_final.jld2`
- PT replacement:
  `/data1/choi/LMP_path33_kappa0158_24eig_no_blockscalar_plus1pt_scratch_adamfix_20260908/(16x32)33PTlayer33path(kappa=0.158)(4000epoch)(maxlen5)(xavier_init)(24eig)(320config)(no_blockscalar_plus1pt)_final.jld2`

Require `opt_t=4000` in both sources. Build the BlockScalar model with
`block_r=6`, giving 512 blocks on the new lattice and matching its
`(512,24)` checkpoint weights. Build the PT-replacement model with the exact
38-layer training stack: 32 effective width-33 PT body layers, a width-24 output
PT layer, no BlockScalar layer, normalization, and Ritz loss.

## Random Fermion Field and Derived Checkpoints

The archived checkpoint fermion fields have `16^3 x 32` extent. For both learned
models:

- create result-local derived checkpoints that omit only
  `layer_*_fermion_*` keys;
- require every retained key and value to equal the source exactly;
- initialize one `24^3 x 48` Gaussian fermion field with CPU and CUDA seed
  `20260831`;
- require the same SHA-256 field digest for both models;
- verify the digest before and after inference;
- never modify the source checkpoints.

Use a memory-bounded forward loop that invokes the existing layers in their
original order and clears each consumed layer cache through the existing
`LMP.Layer_sequence._clear_layer_forward_cache!`. This loop returns the final
24 vectors without implementing new layer mathematics.

## DD-alphaAMG and Solve Settings

Preserve the settings from the successful reference comparison:

- setup mode: `direct_near_null`
- aggregation block size: 4
- learned-vector smoothing: enabled, 16 MR steps
- chirality: enabled
- V-cycle setup iterations: 6
- setup smoother: SAP, 3 iterations
- setup block solver: MR, 4 steps
- V-cycle post smoother: SAP, 1 iteration, block MR 4
- solve tolerance: `1e-10`
- maximum iterations: 200
- restart: 50
- solve smoother: MR
- pre/post smoothing: 4/4
- coarse tolerance: 0.1
- coarse maximum iterations/restart: 50/20
- coarse odd-even: disabled
- coarse GMRES mode: `restart_boundary`
- SAP odd-even: disabled
- SAP schedule: `twocolor`
- deflated initial guess: enabled for both learned cases, disabled for V-cycle

Use RHS seed `42 + round(1_000_000*kappa)`. Recreate the identical normalized
RHS before every case so that model-field randomization cannot alter the setup
RNG state. Count fine and coarse applications only from the existing
`solve_DDalphaAMG` counters and independently recompute the relative residual.

## Warm-up and Measurement

At kappa 0.1527, run one complete unrecorded setup-and-solve warm-up for each of
the three cases. Discard all warm-up counts and timings. Recreate the RHS and
model state before measured runs. The second kappa reuses the compiled kernels
but creates a fresh Dirac operator, RHS, learned vectors, and DD-alphaAMG setup.

Do not impose a runtime timeout. Execute cases sequentially on one GPU and
reclaim model/setup memory between cases. If any case fails or exceeds memory,
record and preserve the completed rows and logs, then stop without changing the
volume, kappa, tolerance, Nv, checkpoint, or solver settings.

## Outputs

Use a new non-overwriting result root under:

`/home/choi/LMP/results/experiments/`

Produce:

- a six-row CSV: two kappas times three cases;
- one setup/solve log per case and kappa;
- source and derived checkpoint hashes and removed-key provenance;
- random-field digest and full settings record;
- validation report;
- publication-style PNG and PDF with two log-log panels.

The plot uses fine Dirac applications on the horizontal axis and coarse operator
applications on the vertical axis. Because every case has `Nv=24`, use distinct
marker shapes and colors for the three methods rather than encoding Nv by
marker. Annotate solve time for converged cases and `not conv.` otherwise. Do
not draw diagonal guide lines.

## Validation

Before launch, require both source checkpoints, the gauge, free GPU capacity,
derived-checkpoint equality, correct model stacks, identical field digests, and
source-contract tests. Refuse existing result paths.

Completion requires exactly six unique case rows, finite positive application
counts and timings, residual-based convergence labels, matching kappa and gauge
provenance, readable PNG/PDF files, an all-pass validation report, and no
surviving experiment process.
