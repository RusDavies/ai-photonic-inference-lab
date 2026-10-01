# CIFAR-10 Feature Stress Matrix

Date: 2026-10-01

## Purpose

Run the CIFAR-10 spatial-feature source-coding stress path across the mild,
moderate, and severe constrained fixed-projection scenarios. The goal is to
decide whether the harder-workload story needs a less severe architecture
assumption, a better projection calibration model, or a stronger CIFAR target
with more test margin.

## Command Shape

Each row used the same bounded CIFAR recipe and changed only the constrained
fixed-projection scenario:

```bash
.venv/bin/python -m optical_spike \
  --run-cifar-feature-source-coding-stress \
  --dataset cifar10 \
  --epochs 30 \
  --hidden-dim 256 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --calibration-sample-counts 128 \
  --source-coding-projection-mse-threshold 0.18 \
  --constrained-fixed-scenario <signed_tiled_mild|signed_tiled_moderate|signed_tiled_severe> \
  --output-dir artifacts/PIL-001-cifar-feature-stress-matrix-2026-10-01/<scenario>
```

The CLI gained `--constrained-fixed-scenario` for this comparison. The default
remains `signed_tiled_severe`, preserving prior behavior.

## Matrix Summary

| Constraint | Target Accuracy | Mean Calibrated Accuracy | Min Calibrated Accuracy | Max Calibrated Accuracy | Rows Below 50% | Weight MSE | Mean Projection MSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `signed_tiled_mild` | 52.80% | 52.76% | 51.70% | 53.75% | 0 / 192 | 0.000055 | 0.056102 |
| `signed_tiled_moderate` | 52.80% | 52.83% | 52.00% | 54.00% | 0 / 192 | 0.001422 | 0.158675 |
| `signed_tiled_severe` | 52.80% | 43.97% | 29.95% | 47.75% | 192 / 192 | 0.011824 | 3.201442 |

## Best Rows

Best minimum-accuracy rows:

| Constraint | Source Coding | Physical Scenario | Policy | Mean Accuracy | Minimum Accuracy |
| --- | --- | --- | --- | ---: | ---: |
| `signed_tiled_mild` | `direct_7bit` | `athermal_compensated_mrr_readout_edge` | `scheduled_only` | 52.89% | 52.65% |
| `signed_tiled_moderate` | `direct_7bit` | `athermal_compensated_mrr_nominal` | `scheduled_only` | 53.00% | 52.75% |
| `signed_tiled_severe` | `redundant_6bit_2phase` | `athermal_compensated_mrr_nominal` | `scheduled_plus_projection_mse_trigger` | 45.96% | 44.85% |

Best mean-accuracy rows:

| Constraint | Source Coding | Physical Scenario | Policy | Mean Accuracy | Minimum Accuracy |
| --- | --- | --- | --- | ---: | ---: |
| `signed_tiled_mild` | `direct_7bit` | `athermal_compensated_mrr_sparse_refresh` | `scheduled_plus_projection_mse_trigger` | 53.00% | 52.45% |
| `signed_tiled_moderate` | `direct_7bit` | `athermal_compensated_mrr_sparse_refresh` | `scheduled_only` | 53.03% | 52.65% |
| `signed_tiled_severe` | `redundant_6bit_2phase` | `athermal_compensated_mrr_sparse_refresh` | `scheduled_plus_projection_mse_trigger` | 46.12% | 44.40% |

## Interpretation

Mild and moderate constrained fixed-projection assumptions preserve the bounded
CIFAR-10 spatial-feature target after calibration. Both stay close to the
`52.80%` target accuracy, and neither produces below-50% rows.

The severe signed-tiled projection scenario remains the dominant failure mode.
It drops mean calibrated accuracy to `43.97%`, has a minimum of `29.95%`, and
puts every row below `50%`. Its projection MSE is roughly twenty times the
moderate scenario and roughly fifty-seven times the mild scenario.

The result is consistent with the 2026-09-04 diagnosis: source coding,
calibration sample count, and MSE-triggered recalibration are not the main
limiters for the current bounded CIFAR target. The architecture constraint is.

## Decision

For a harder-workload manuscript story, use the mild/moderate constrained
architecture branch as the plausible path. The severe branch should be treated
as a stress/failure boundary unless a materially stronger calibration model or
hardware-aware training method is added.

Immediate next research work should justify the architecture assumption:

- define what physical/fabrication assumptions make `signed_tiled_mild` and
  `signed_tiled_moderate` credible;
- explain why `signed_tiled_severe` is retained as a failure-boundary stress
  case rather than the default manuscript architecture;
- avoid claiming CIFAR-10 harder-workload success under the severe constraint.

## Verification

- Focused CLI regression:
  `.venv/bin/python -m pytest tests/test_cli.py::test_cifar_feature_source_coding_stress_saves_metrics`
- Focused lint:
  `.venv/bin/python -m ruff check src/optical_spike/cli.py tests/test_cli.py`
