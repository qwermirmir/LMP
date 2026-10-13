# DD-alphaAMG LMP Seed Injection Design

## Goal

Measure whether LMP-produced prototype vectors improve the reference DD-alphaAMG adaptive setup when they replace only the random initial test vectors.

## Experimental Definition

The reference path remains unchanged after seed selection:

1. load either random or LMP prototype vectors;
2. apply the reference initial three SAP smoothing stages;
3. normalize and aggregate-orthonormalize the prototypes;
4. construct the initial interpolation and coarse operator;
5. perform six reference adaptive setup iterations;
6. solve with identical reference solver parameters.

The LMP vectors are unsplit fine-grid prototypes. Reference DD-alphaAMG performs its own gamma5-compatible aggregation and doubles the coarse spin-sector dimension internally.

## API

Add `dd_alpha_amg_setup_with_seed_vectors(double *vectors, int num_vectors, int iterations, int *status)` to the local reference library. `vectors` contains `num_vectors` consecutive fine vectors in the public API layout selected by `vector_index_fct`.

The new API validates the vector count, builds a pointer table, and invokes the existing setup with those pointers. A non-null seed path in interpolation setup loads each vector and then executes the same three smoother calls and normalization used by the random path. Existing `dd_alpha_amg_setup` behavior remains unchanged.

## Comparison Driver

Extend the existing external-test-vector diagnostic into a learned-seed comparison driver. It loads an LMP checkpoint, evaluates the model on the selected gauge, converts each prototype through the already validated spin and site mapping, and passes the contiguous vectors to a reference diagnostic executable.

The driver runs two arms with identical gauge, kappa, RHS, Nv, setup iterations, and solver settings:

- reference random seeds;
- LMP seeds.

Before each run it prints and archives every setting. It records setup time, solve time, setup/solve iteration counters, reported residual, and a Julia-operator true residual.

## Validation

Before the comparison solve:

- dump the loaded seed from the reference code before smoothing and compare it to the source LMP vector after inverse conversion;
- require relative error below `1e-12` in double precision;
- verify that both arms execute three initial smoothing stages and six adaptive setup iterations;
- verify identical plaquette, gauge, kappa, boundary condition, RHS, Nv, and solver settings.

No running experiment is stopped or replaced.
