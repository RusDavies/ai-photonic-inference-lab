# CIFAR-10 Spatial Feature Transfer Target

Date: 2026-09-04

## Purpose

Add a stronger CIFAR-10 transfer target before using CIFAR-10 as
harder-workload optical-transfer evidence.

The dense flattened-pixel and transformer-style MLP-up targets learned above
chance, but they topped out below `41%` on a bounded CIFAR-10 slice. CIFAR-10
needs spatial/color bias, so this pass adds a deterministic frozen feature
extractor followed by the existing trainable optical-candidate projection.

## Implementation

New CLI target:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-cifar-feature-target \
  --dataset cifar10
```

The target extracts `460` deterministic features from each CIFAR-10 image:

- pooled RGB grids at `8x8`, `4x4`, and `2x2`;
- pooled grayscale grids at `8x8` and `4x4`;
- pooled edge-magnitude grids at `8x8` and `4x4`;
- local RGB standard-deviation features over a `4x4` grid.

Those frozen features are z-scored from the training split, then passed through
a learned optical-candidate projection `460xhidden_dim` and classifier head.
The command writes metrics and weights, including feature normalization state,
under `cifar_feature_target/`.

## Bounded Run

Command:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-cifar-feature-target \
  --dataset cifar10 \
  --epochs 30 \
  --hidden-dim 256 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --quantization-bits 8,6,4 \
  --output-dir artifacts/cifar10-feature-target-2026-09-04
```

Artifacts:

- `artifacts/cifar10-feature-target-2026-09-04/cifar_feature_target/cifar_feature_target.json`
- `artifacts/cifar10-feature-target-2026-09-04/cifar_feature_target/cifar_feature_target_weights.npz`

## Results

| Metric | Value |
| --- | ---: |
| Train samples | 10,000 |
| Test samples | 2,000 |
| Feature dimension | 460 |
| Projection shape | 460x256 |
| Best test accuracy | 53.30% |
| Final test accuracy | 52.80% |
| Final train accuracy | 93.63% |

Projection quantization result:

| Bits | Test Accuracy | Accuracy Drop |
| ---: | ---: | ---: |
| 8 | 52.85% | -0.05 points |
| 6 | 53.15% | -0.35 points |
| 4 | 51.70% | 1.10 points |

## Interpretation

This target is substantially stronger than the prior bounded CIFAR candidates:

| Target | Best/Key Test Accuracy |
| --- | ---: |
| Dense flattened-pixel baseline | 35.15% best |
| Transformer-style MLP-up target | 38.35% best |
| MLP-up moderate fixed + tail fine-tuning | 40.30% adapted |
| CIFAR spatial feature projection target | 53.30% best |

The new result is still not a modern CIFAR-10 classifier, and the gap between
train and test accuracy shows obvious overfitting on the bounded slice. But it
does solve the immediate blocker: the project now has a CIFAR-10 target with
spatial bias and a learned optical-candidate projection that is meaningfully
above chance.

## Source-Coding Stress Integration

The feature target now has a comparable constrained/source-coding stress runner:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-cifar-feature-source-coding-stress \
  --dataset cifar10 \
  --epochs 30 \
  --hidden-dim 256 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --calibration-sample-counts 128 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/cifar10-feature-source-coding-stress-2026-09-04
```

Artifacts:

- `artifacts/cifar10-feature-source-coding-stress-2026-09-04/cifar_feature_source_coding_stress/cifar_feature_source_coding_stress_rows.csv`
- `artifacts/cifar10-feature-source-coding-stress-2026-09-04/cifar_feature_source_coding_stress/cifar_feature_source_coding_stress_summary.csv`
- `artifacts/cifar10-feature-source-coding-stress-2026-09-04/cifar_feature_source_coding_stress/cifar_feature_source_coding_stress_summary.md`
- `artifacts/cifar10-feature-source-coding-stress-2026-09-04/cifar_feature_source_coding_stress/cifar_feature_source_coding_stress_manifest.json`

Result:

| Metric | Value |
| --- | ---: |
| Rows | 192 |
| Summary rows | 24 |
| Target accuracy | 52.80% |
| Calibrated accuracy minimum | 29.95% |
| Calibrated accuracy mean | 43.97% |
| Calibrated accuracy maximum | 47.75% |
| Rows below 50% | 192 |

Best minimum summary row:

| Source Coding | Physical Scenario | Policy | Mean Accuracy | Minimum Accuracy |
| --- | --- | --- | ---: | ---: |
| redundant_6bit_2phase | athermal_compensated_mrr_nominal | scheduled_plus_projection_mse_trigger | 45.96% | 44.85% |

Best mean summary row:

| Source Coding | Physical Scenario | Policy | Mean Accuracy | Minimum Accuracy |
| --- | --- | --- | ---: | ---: |
| redundant_6bit_2phase | athermal_compensated_mrr_sparse_refresh | scheduled_plus_projection_mse_trigger | 46.12% | 44.40% |

Interpretation: the project now has a real CIFAR-10 feature-projection
source-coding stress path, but the severe constrained projection plus current
calibration policy loses too much accuracy. The nominal/readout cases preserve
roughly `44%` to `46%`, while the thermal-edge case can fall to about `30%`.

This is still not manuscript-grade harder-workload evidence. It is the first
proper CIFAR-specific stress result, and it points to the next experiment:
diagnose whether the loss is dominated by severe signed-tiled projection error,
source coding, calibration sample count, MSE-trigger thresholds, or the target's
overfitted margin.
