# LMP Eigenvector Subspace Comparison

## Objective

Quantify how closely the six vectors produced by the trained LMP model match
the low-mode eigenspace of the Wilson operator on gauge configurations 401 and
405. Produce reproducible numerical results and slide-ready figures.

## Fixed Inputs

- Physical GPU: 1 (`CUDA_VISIBLE_DEVICES=1`)
- Lattice: `16^3 x 32`
- Gauge files:
  - `/data1/choi/heatbath_ensemble_beta6.0/raw_gauge/config_401.gbin`
  - `/data1/choi/heatbath_ensemble_beta6.0/raw_gauge/config_405.gbin`
- Wilson hopping parameter: `kappa = 0.158`
- Number of exact modes: 6
- Eigenpair relative residual tolerance: `1e-10`
- LMP checkpoint:
  `/data1/choi/LMP_path33_kappa0158_6eig_scratch_20260614/(16x32)32PTlayer33path(kappa=0.158)(16000epoch)(maxlen5)(xavier_init)(6eig)(320config)(r=2)_final.jld2`
- LMP output: six prototype vectors per configuration

## Operator Definition

The exact target space is the eigenspace associated with the six smallest
eigenvalues of

```text
A = D' * D = (gamma5 * D)^2.
```

This Hermitian positive-semidefinite operator measures the low singular modes
of the Wilson Dirac operator. It avoids basis-dependent sign and ordering
ambiguities in direct comparisons with `gamma5 * D` while targeting the same
near-null space.

For every computed pair `(lambda_i, q_i)`, require

```text
norm(A*q_i - lambda_i*q_i) / max(norm(A*q_i), eps()) <= 1e-10.
```

An output is invalid if any requested mode fails this check.

## Comparison Metrics

Orthonormalize the exact vectors and LMP vectors to obtain `Q_E` and `Q_L`.
Define the cross-Gram matrix

```text
C = Q_E' * Q_L.
```

Report:

1. Principal-angle cosines: the six singular values of `C`.
2. Principal angles in degrees.
3. Mean target-space capture:
   `norm(C, Frobenius)^2 / 6`, bounded by zero and one.
4. Worst-direction capture: `minimum(svdvals(C))^2`.
5. Per-eigenmode capture: `norm(Q_L' * q_i)^2` for each exact mode.
6. Pairwise squared overlaps `abs(q_i' * l_j)^2` as a heatmap. This is
   supplementary because individual vectors can rotate within a subspace.
7. Exact low-space overlap between configurations 401 and 405, using the same
   principal-angle metrics after mapping both fields to the common canonical
   lattice ordering.

The primary slide number is mean target-space capture. Principal angles and
per-mode capture show whether that average hides a missed direction.

## Execution And Outputs

Before the eigensolve begins, print all fixed inputs, operator definition,
checkpoint path, solver tolerance, and output paths. Process configurations
sequentially on GPU 1 and save each validated eigensystem immediately so a
later interruption does not discard completed work.

Create:

- A Julia script under `scripts/experiments/` that reproduces the calculation.
- Validated eigenvector JLD2 files under `results/eigenvectors/`.
- A detailed CSV under `results/tables/` containing eigenvalues, residuals,
  captures, singular values, and principal angles.
- Slide-ready PNG and PDF figures under `results/plots/`:
  - per-mode capture for configurations 401 and 405;
  - principal-angle cosine summary;
  - optional pairwise-overlap heatmaps.
- A settings text file beside the results recording the git commit, package
  environment, GPU, gauge paths, and checkpoint path.

## Validation

- Verify gauge files exist and print their sizes and checksums.
- Verify the checkpoint exists and record its checksum.
- Verify exact vectors are orthonormal to numerical precision.
- Verify all six eigenpair residuals satisfy `1e-10` independently.
- Verify LMP vectors have rank six before comparison.
- Recompute subspace metrics after a random unitary rotation of each basis;
  the principal-angle results must remain unchanged within numerical error.
- Do not publish pairwise vector matching as the primary similarity metric.
