# CIFAR-10 Feature Stress Diagnosis

Date: 2026-09-04

## Purpose

Diagnose the CIFAR-10 feature-projection stress loss by separating severe
signed-tiled projection error, source coding, calibration sample count, trigger
threshold, and target-margin effects before using CIFAR-10 as harder-workload
optical-transfer evidence.

## Inputs

Primary stress artifact:

- `artifacts/cifar10-feature-source-coding-stress-2026-09-04/cifar_feature_source_coding_stress/cifar_feature_source_coding_stress_rows.csv`

Calibration sensitivity artifact:

- `artifacts/cifar10-feature-stress-diagnosis-calibration-2026-09-04/cifar_feature_source_coding_stress/cifar_feature_source_coding_stress_rows.csv`

Direct factor artifact:

- `artifacts/cifar10-feature-stress-diagnosis-2026-09-04/direct_factor_rows.csv`

## Calibration Count Sweep

Command:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-cifar-feature-source-coding-stress \
  --dataset cifar10 \
  --epochs 30 \
  --hidden-dim 256 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --calibration-sample-counts 32,128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/cifar10-feature-stress-diagnosis-calibration-2026-09-04
```

Result:

| Calibration Samples | Mean Accuracy | Minimum Accuracy | Maximum Accuracy |
| ---: | ---: | ---: | ---: |
| 32 | 43.63% | 29.90% | 47.40% |
| 128 | 43.97% | 29.95% | 47.75% |
| 512 | 44.18% | 28.65% | 48.15% |
| 2048 | 44.17% | 29.05% | 47.95% |

Interpretation: calibration sample count is not the dominant limiter. Moving
from `32` to `2048` samples changes mean accuracy by about `0.55` points.

## Recalibration Policy

| Policy | Mean Accuracy | Minimum Accuracy | Maximum Accuracy | Triggered Recalibrations |
| --- | ---: | ---: | ---: | ---: |
| scheduled_only | 43.79% | 28.65% | 47.80% | 0 |
| scheduled_plus_projection_mse_trigger | 44.19% | 29.95% | 48.15% | 300 |

Interpretation: projection-MSE-triggered recalibration helps only marginally at
the current threshold and calibration model. It is not enough to recover the
severe projection loss.

## Source Coding

| Source Coding | Mean Accuracy | Minimum Accuracy | Maximum Accuracy | Mean Source MSE |
| --- | ---: | ---: | ---: | ---: |
| direct_7bit | 43.97% | 28.65% | 47.95% | 0.000077 |
| redundant_6bit_2phase | 43.95% | 32.55% | 47.75% | 0.000316 |
| redundant_7bit_2phase | 44.05% | 29.05% | 48.15% | 0.000077 |

Direct source-only checks against ideal weights:

| Source Coding | Accuracy | Drop vs Target | Projection MSE |
| --- | ---: | ---: | ---: |
| redundant_6bit_2phase | 52.15% | 0.65 points | 0.060789 |
| direct_7bit | 52.80% | 0.00 points | 0.016536 |
| redundant_7bit_2phase | 52.90% | -0.10 points | 0.014714 |

Interpretation: source coding is not the loss driver. It is mostly drowned by
the projection constraint.

## Projection Constraint

Direct constrained projection checks without thermal drift/noise:

| Constraint | Condition | Accuracy | Drop vs Target | Projection MSE |
| --- | --- | ---: | ---: | ---: |
| signed_tiled_mild | raw | 52.70% | 0.10 points | 0.043822 |
| signed_tiled_mild | calibrated, 128 samples | 52.90% | -0.10 points | 0.011356 |
| signed_tiled_moderate | raw | 52.10% | 0.70 points | 1.468497 |
| signed_tiled_moderate | calibrated, 128 samples | 52.55% | 0.25 points | 0.113067 |
| signed_tiled_severe | raw | 25.70% | 27.10 points | 12.450365 |
| signed_tiled_severe | calibrated, 128 samples | 46.80% | 6.00 points | 1.738135 |
| signed_tiled_severe | calibrated, 512 samples | 47.55% | 5.25 points | 1.719805 |
| signed_tiled_severe | calibrated, 2048 samples | 47.30% | 5.50 points | 1.712988 |

Severe constraint plus source coding, still without thermal drift/noise:

| Source Coding | Accuracy | Drop vs Target | Projection MSE |
| --- | ---: | ---: | ---: |
| redundant_6bit_2phase | 47.75% | 5.05 points | 1.814777 |
| direct_7bit | 47.80% | 5.00 points | 1.764575 |
| redundant_7bit_2phase | 48.00% | 4.80 points | 1.762921 |

Interpretation: the severe signed-tiled projection constraint is the dominant
accuracy loss. Mild and moderate constraints are essentially recoverable with
calibration. Severe constraints remain about `5` to `6` points below target
even before thermal drift/noise, then drift/noise pushes the full stress run
down into the `44%` to `46%` nominal/readout range and about `30%` in thermal
edge cases.

## Target Margin

The feature target itself reaches `52.80%` final test accuracy but `93.63%`
final train accuracy on the bounded `10000/2000` slice. That overfit gap makes
the target brittle, but it is secondary to the severe projection constraint:
mild and moderate projection checks still preserve target accuracy after
calibration.

## Diagnosis

Dominant limiter: severe signed-tiled projection error.

Secondary limiters:

- thermal-edge drift/noise after severe projection damage;
- target overfitting and limited test margin;
- current affine calibration is not enough to recover severe projection MSE.

Not dominant in the current bounded run:

- source coding choice;
- calibration sample count above `32`;
- projection-MSE-triggered recalibration at the current threshold.

## Follow-Up

The next useful experiment is not another source-coding tweak. Run the CIFAR
feature stress path across mild, moderate, and severe constrained projection
scenarios, then decide whether the paper-relevant harder-workload story should
use a less severe architecture assumption, a better projection calibration
model, or a stronger CIFAR target with more test margin.
