# Material Nonlinearity Candidates

Date: 2026-06-21

## Purpose

Select concrete material-inspired nonlinear transfer models and test whether
they are better treated as:

- calibrated distortions to remove from the MLP-up projection; or
- physical activation candidates after the MLP-up projection.

## Source Basis

Candidate families came from the project source log and recent photonic neural
network literature:

- saturable absorbers;
- reverse saturable absorption;
- two-photon absorption in silicon/germanium-style platforms;
- Kerr/free-carrier cubic phase-like distortion;
- hybrid electro-optic sigmoid-style activation.

See `research/source-log.md` for the source links.

## Local Artifacts

- `artifacts/material-nonlinearity-full-2026-06-21/material_nonlinearity/material_nonlinearity_rows.csv`
- `artifacts/material-nonlinearity-full-2026-06-21/material_nonlinearity/material_nonlinearity_summary.csv`
- `artifacts/material-nonlinearity-full-2026-06-21/material_nonlinearity/material_nonlinearity_summary.md`
- `artifacts/material-nonlinearity-full-2026-06-21/material_nonlinearity/material_nonlinearity_manifest.json`
- `artifacts/material-training-full-2026-06-21/material_training/material_training_rows.csv`
- `artifacts/material-training-full-2026-06-21/material_training/material_training_summary.csv`
- `artifacts/material-training-full-2026-06-21/material_training/material_training_summary.md`
- `artifacts/material-training-full-2026-06-21/material_training/material_training_manifest.json`
- `artifacts/material-strength-sweep-full-2026-06-21/material_strength_sweep/material_strength_sweep_rows.csv`
- `artifacts/material-strength-sweep-full-2026-06-21/material_strength_sweep/material_strength_sweep_summary.csv`
- `artifacts/material-strength-sweep-full-2026-06-21/material_strength_sweep/material_strength_sweep_summary.md`
- `artifacts/material-strength-sweep-full-2026-06-21/material_strength_sweep/material_strength_sweep_manifest.json`

The artifact directory is intentionally uncommitted.

## Run Context

- Target: transformer-style MLP-up projection, 64x256.
- Seeds: 7, 11, and 13.
- Dataset: full Fashion-MNIST split.
- Epochs: 5.
- Calibration samples: 512.
- Rows: 60.
- Aggregate rows: 20.
- Baseline mean accuracy: 86.86%.

Each material model was evaluated as:

- `distortion_raw`: apply the material transfer to the MLP-up projection and
  keep the rest of the digital model unchanged;
- `distortion_calibrated`: affine-calibrate the distorted projection back toward
  the ideal projection;
- `activation_raw`: replace ReLU after the MLP-up projection with the material
  activation candidate;
- `activation_calibrated`: affine-calibrate the material activation toward the
  original ReLU activation.

## Results

Mean accuracy drop by scenario:

- Saturable absorber:
  - Distortion raw: 0.12 point improvement.
  - Distortion calibrated: 0.01 point drop.
  - Activation raw: 0.12 point improvement.
  - Activation calibrated: 0.09 point drop.
- Reverse saturable absorber:
  - Distortion raw: 0.12 point improvement.
  - Distortion calibrated: 0.02 point improvement.
  - Activation raw: 0.12 point improvement.
  - Activation calibrated: 0.10 point drop.
- Two-photon absorption:
  - Distortion raw: 0.11 point improvement.
  - Distortion calibrated: 0.05 point drop.
  - Activation raw: 0.11 point improvement.
  - Activation calibrated: 0.09 point drop.
- Kerr/free-carrier cubic phase:
  - Distortion raw: 0.28 point drop.
  - Distortion calibrated: 0.49 point drop.
  - Activation raw: 0.28 point drop.
  - Activation calibrated: 0.08 point drop.
- Hybrid electro-optic sigmoid:
  - Distortion raw: 0.17 point drop.
  - Distortion calibrated: 0.10 point drop.
  - Activation raw: 21.46 point drop.
  - Activation calibrated: 2.86 point drop.

## Interpretation

Saturable absorption, reverse saturable absorption, and two-photon absorption
look safe as first material candidates. In this trained MLP-up model they are
not merely tolerable distortions; as raw activation replacements they slightly
improve mean accuracy. That is probably regularization/shape luck, not a law of
physics handed down on stone tablets. Still, it is enough to justify
hardware-aware training with these activation forms in the loop.

Kerr/free-carrier cubic behavior is more mixed. It is still usable after
activation calibration, but it is not as naturally aligned with the trained ReLU
block. Treat it as a secondary candidate or as a distortion to co-train around.

The sigmoid-style hybrid electro-optic activation is a poor drop-in replacement.
It works as a projection distortion after affine calibration, but replacing ReLU
with sigmoid-like behavior without retraining loses too much accuracy. This
does not kill electro-optic activations; it says they need to be present during
training.

## Decision

First material candidates:

1. Saturable absorber.
2. Reverse saturable absorber.
3. Two-photon absorption.

Secondary candidate:

- Kerr/free-carrier cubic distortion.

Do not use sigmoid-style electro-optic activation as a drop-in ReLU substitute.
Only revisit it with end-to-end hardware-aware training.

## End-to-End Material Activation Training

The follow-up trained the MLP-up model from scratch with material-inspired
activation functions in the forward pass, rather than swapping them in after
ReLU training.

Run context:

- Seeds: 7, 11, and 13.
- Dataset: full Fashion-MNIST split.
- Epochs: 5.
- Candidates: saturable absorber, reverse saturable absorber, two-photon
  absorption, Kerr/free-carrier cubic.
- Baseline: the same ReLU MLP-up model per seed.

Mean test accuracy and delta vs ReLU:

- Reverse saturable absorption: 87.09%, +0.23 points.
- Saturable absorption: 86.97%, +0.11 points.
- Two-photon absorption: 86.74%, -0.12 points.
- Kerr/free-carrier cubic: 86.57%, -0.29 points.

Per-seed behavior matters:

- Seed 13 improved for all four material candidates.
- Seed 7 got worse for all four.
- Seed 11 favored reverse saturable absorption and nearly tolerated saturable
  absorption.

Interpretation: reverse saturable absorption is the strongest first activation
candidate once trained in the loop. Saturable absorption remains viable but
needs strength tuning. Two-photon absorption and Kerr/free-carrier cubic are not
dead, but this parameterization does not beat ReLU on average.

The earlier post-hoc result slightly overstated the case for two-photon
absorption. Once the model is trained end-to-end, TPA no longer has a mean
advantage. Good catch by the data, which remains irritatingly better than
guessing.

## Strength Sweep and Fixed Transfer

The next pass swept activation strength for saturable and reverse-saturable
absorption, then applied the severe constrained fixed-fabrication model to the
trained MLP-up weights. That fixed model includes signed differential encoding,
fan-in/fan-out tile pressure, excess optical loss, geometry-linked crosstalk,
detector noise, and differential path noise.

Run context:

- Seeds: 7, 11, and 13.
- Dataset: full Fashion-MNIST split.
- Epochs: 5.
- Strengths: 0.1, 0.2, and 0.35.
- Calibration samples: 512.
- Constrained scenario: `signed_tiled_severe`.
- Rows: 18.
- Aggregate rows: 6.

Mean test accuracy:

- Reverse saturable, strength 0.1: ideal 87.00%, constrained raw 86.60%,
  constrained calibrated 86.92%.
- Reverse saturable, strength 0.2: ideal 87.09%, constrained raw 86.45%,
  constrained calibrated 86.75%.
- Reverse saturable, strength 0.35: ideal 86.75%, constrained raw 86.14%,
  constrained calibrated 86.66%.
- Saturable, strength 0.1: ideal 87.00%, constrained raw 86.50%,
  constrained calibrated 86.68%.
- Saturable, strength 0.2: ideal 87.09%, constrained raw 86.42%,
  constrained calibrated 86.79%.
- Saturable, strength 0.35: ideal 86.75%, constrained raw 85.86%,
  constrained calibrated 86.57%.

The best ideal trained accuracy came from strength 0.2 for both saturable and
reverse-saturable absorption: 87.09%, or 0.23 points above the ReLU baseline.
After severe constrained transfer and affine calibration, reverse saturable
strength 0.1 had the best transferred accuracy at 86.92%, only 0.08 points
below its ideal material model. Reverse saturable strength 0.35 transferred
almost as cleanly, but its ideal model was weaker.

Interpretation: lower-strength reverse saturable absorption is the leading
hardware target. Strength 0.2 is a small training win in the ideal digital
model, but strength 0.1 survives the severe constrained transfer better. The
practical recommendation is to treat 0.1 to 0.2 as the first device-parameter
window, then spend future experiments on chaining blocks and converter
amortization instead of chasing more activation micro-tuning.

## Next Work

Explore chaining multiple optical blocks before ADC to amortize converter
latency and energy.
