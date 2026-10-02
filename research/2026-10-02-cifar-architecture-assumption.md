# CIFAR-10 Architecture Assumption

Date: 2026-10-02

## Purpose

Convert the CIFAR-10 feature stress matrix into an explicit manuscript-facing
architecture assumption. The matrix showed that `signed_tiled_mild` and
`signed_tiled_moderate` preserve the bounded CIFAR-10 spatial-feature target,
while `signed_tiled_severe` remains a failure boundary.

This note does not claim hardware validation. It defines which modeled
fabrication and layout assumptions are plausible enough to use as the
paper-relevant branch, and which scenario should stay in the limits analysis.

## Evidence Anchor

The comparison is recorded in
[`research/2026-10-01-cifar-feature-stress-matrix.md`](2026-10-01-cifar-feature-stress-matrix.md).
The bounded CIFAR-10 recipe changed only the constrained fixed-projection
scenario.

Summary:

| Constraint | Target Accuracy | Mean Calibrated Accuracy | Min Calibrated Accuracy | Rows Below 50% | Mean Projection MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `signed_tiled_mild` | 52.80% | 52.76% | 51.70% | 0 / 192 | 0.056102 |
| `signed_tiled_moderate` | 52.80% | 52.83% | 52.00% | 0 / 192 | 0.158675 |
| `signed_tiled_severe` | 52.80% | 43.97% | 29.95% | 192 / 192 | 3.201442 |

## Scenario Meaning

The constrained fixed-projection scenarios are intentionally simple simulation
branches. They should be read as architecture envelopes, not device claims.

`signed_tiled_mild` is the optimistic branch:

- 8-bit signed symmetric weight quantization;
- low fabrication error (`0.005`);
- larger effective fan-in/fan-out limits (`128` / `256`);
- low geometry crosstalk and differential noise;
- modest optical loss budget.

This branch is credible only for a relatively well-controlled tiled weight
bank: enough routing and calibration support to avoid aggressive subdivision,
low coupling error between nearby channels, and an optical/readout chain whose
noise is not the dominant accuracy limiter.

`signed_tiled_moderate` is the paper-relevant baseline branch:

- 6-bit signed symmetric weight quantization;
- moderate fabrication error (`0.01`);
- smaller but still useful fan-in/fan-out limits (`64` / `128`);
- higher optical loss and geometry crosstalk than the mild branch;
- detector and differential noise that still remain within calibration reach.

This branch is the strongest manuscript assumption because it is meaningfully
constrained but not deliberately hostile. It allows the architecture story to
depend on compact signed/tiled optical projection with active calibration,
without asking the reader to accept either near-ideal precision or a passive
open-loop device.

`signed_tiled_severe` is the failure-boundary branch:

- 4-bit signed symmetric weight quantization;
- high fabrication error (`0.025`);
- tight fan-in/fan-out limits (`32` / `64`);
- high geometry crosstalk, optical loss, detector noise, and differential
  noise.

This branch is valuable because it shows where the current recipe breaks. It is
not the default manuscript architecture for CIFAR-10 because the measured
projection damage dominates every calibration/source-coding choice in the
matrix.

## Manuscript Positioning

The CIFAR-10 result can support this narrower claim:

> Under a bounded CIFAR-10 spatial-feature target, the modeled signed/tiled
> optical projection remains viable under mild and moderate constrained
> fabrication envelopes after calibration, while the severe envelope exposes a
> hard failure boundary.

It should not support these stronger claims:

- CIFAR-10 harder-workload success under `signed_tiled_severe`;
- device-level validation of an MRR or other fabricated weight bank;
- modern image-classifier performance;
- generalization to vision workloads beyond this bounded spatial-feature
  target.

The severe case should appear in the paper as a stress/failure boundary and as
evidence that architecture assumptions dominate the current bounded CIFAR path.
The mild/moderate cases are the branch to use when discussing a plausible
harder-workload path.

## Follow-Up Work

A better projection calibration model and a stronger CIFAR target should be
separate work items.

Better calibration is justified because `signed_tiled_severe` still has high
projection MSE after affine calibration. The next calibration work should test
whether a richer projection correction or hardware-aware retraining can reduce
that damage without hiding the architecture limit.

A stronger CIFAR target is also separate. The current bounded spatial-feature
target is useful because it is deterministic, cheap enough for the NumPy runner,
and above the earlier weak CIFAR baselines. It is not a modern CIFAR classifier.
If the manuscript needs a stronger harder-workload claim, the target itself
needs more margin before optical transfer is the interesting bottleneck.

## Decision

Use `signed_tiled_moderate` as the manuscript's default CIFAR architecture
assumption, with `signed_tiled_mild` as the optimistic comparison and
`signed_tiled_severe` as the failure-boundary stress case.

