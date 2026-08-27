# Fair Learned-versus-Standard AMG Setup Design

## Goal

Compare Standard AMG-24 and learned-refined AMG-24 after changing only the
source of the 24 initial prototype vectors.  Remove the raw-learned case.

## Compared cases

- `standard_amg_24`: 24 deterministic random prototypes.
- `learned_refined_amg_24`: 24 vectors emitted by the trained model.

Both cases use the same configuration, shared Davidson initial direction,
solver tolerances, subspace parameters, and random-seed metadata.

## Common adaptive setup pipeline

`setup_DDalphaAMG` shall accept either internally generated random prototypes
or caller-supplied prototypes, then send both through one shared pipeline:

1. copy the prototypes so caller-owned vectors are not mutated;
2. apply the same inverse-SAP initial smoothing stages 1, 2, and 3;
3. perform six adaptive setup iterations;
4. globally orthonormalize all 24 prototypes before every iteration;
5. rebuild the chiral interpolation and Galerkin coarse operator every
   iteration;
6. apply the same V-cycle update, SAP post-smoothing, block solver, coarse
   tolerance, restart, and iteration settings;
7. return the final unsplit prototypes and build the final chiral hierarchy.

No separate learned-only V-cycle loop or fixed learned hierarchy is allowed.

## Experiment driver

The driver performs model inference once per configuration and passes the
model vectors directly to the common adaptive setup pipeline.  It writes only
the Standard and learned-refined rows to summary/history files and plots two
points per configuration.  Existing result directories remain untouched.

## Verification

- A setup-path regression test proves externally supplied prototypes execute
  the same initial and adaptive stages as internally generated prototypes.
- A driver source/integration test proves the raw case and old learned-only
  refinement loop are absent.
- Existing DD-alphaAMG and GD-lambdaAMG CPU tests must pass.
- A cfg401 one-expansion GPU smoke must produce exactly two result rows.
- Only after the smoke passes may a new cfg401 full run be started.

