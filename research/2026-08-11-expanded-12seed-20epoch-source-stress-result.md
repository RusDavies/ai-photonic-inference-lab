# Expanded 12-Seed 20-Epoch Source-Coding Stress Result

Date: 2026-08-11

Purpose: rerun Gate A with stronger training after the 10-epoch expanded-seed
attempt failed at seed `17` and the bounded seed-`17` 20-epoch check cleared.

## Command

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --seeds 7,11,13,17,19,23,29,31,37,41,43,47 \
  --epochs 20 \
  --calibration-sample-counts 128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-full-12seed-20epoch-2026-08-11
```

## Artifact Paths

The generated artifact directory is ignored by git:

```text
artifacts/source-coding-calibration-stress-full-12seed-20epoch-2026-08-11/source_coding_calibration_stress/
```

Key files:

- `source_coding_calibration_stress_rows.csv`;
- `source_coding_calibration_stress_summary.csv`;
- `source_coding_calibration_stress_summary.md`;
- `source_coding_calibration_stress_manifest.json`;
- `seed_slices/seed_<n>_rows.csv`;
- `seed_slices/seed_<n>_manifest.json`.

## Result

Gate A clears under the 20-epoch expanded-seed run.

| Metric | Value |
| --- | ---: |
| Seeds completed | 12 |
| Row observations | 6912 |
| Summary rows | 72 |
| Passing summary rows | 72 |
| Below-85 row observations | 0 |
| Lowest passing source bits | 6 |
| Minimum calibrated accuracy | 0.8584 |
| Mean calibrated accuracy | 0.872415 |

The strict zero-below-85 criterion is satisfied across the full requested seed
set. This replaces the failed 10-epoch expanded attempt as the current Gate A
robustness result.

## Seed Summary

| Seed | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Trained-chain accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 576 | 0 | 0.8622 | 0.867276 | 0.8737 |
| 11 | 576 | 0 | 0.8697 | 0.875146 | 0.8822 |
| 13 | 576 | 0 | 0.8711 | 0.875885 | 0.8837 |
| 17 | 576 | 0 | 0.8643 | 0.869960 | 0.8797 |
| 19 | 576 | 0 | 0.8690 | 0.875840 | 0.8815 |
| 23 | 576 | 0 | 0.8584 | 0.864856 | 0.8665 |
| 29 | 576 | 0 | 0.8620 | 0.869737 | 0.8773 |
| 31 | 576 | 0 | 0.8748 | 0.879165 | 0.8843 |
| 37 | 576 | 0 | 0.8683 | 0.872308 | 0.8792 |
| 41 | 576 | 0 | 0.8709 | 0.874429 | 0.8825 |
| 43 | 576 | 0 | 0.8655 | 0.872551 | 0.8801 |
| 47 | 576 | 0 | 0.8668 | 0.871831 | 0.8778 |

Seed `23` is the weakest completed seed. It still clears the floor, but only
with a `0.84` percentage point minimum-accuracy margin.

## Source-Coding Shape

| Source coding | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_7bit` | 2304 | 0 | 0.8612 | 0.872980 | 0.0146 |
| `redundant_6bit_2phase` | 2304 | 0 | 0.8584 | 0.871286 | 0.0154 |
| `redundant_7bit_2phase` | 2304 | 0 | 0.8612 | 0.872981 | 0.0153 |

The weakest source-coding branch remains `redundant_6bit_2phase`, but it passes
the strict floor in the expanded 20-epoch run.

## Calibration And Policy Shape

| Calibration samples | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 2304 | 0 | 0.8620 | 0.872216 | 0.0154 |
| 512 | 2304 | 0 | 0.8607 | 0.872570 | 0.0140 |
| 2048 | 2304 | 0 | 0.8584 | 0.872460 | 0.0131 |

| Recalibration policy | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scheduled_only` | 3456 | 0 | 0.8584 | 0.872375 | 0.0153 |
| `scheduled_plus_projection_mse_trigger` | 3456 | 0 | 0.8591 | 0.872456 | 0.0154 |

The 20-epoch result does not depend on a single calibration sample count or on
projection-MSE-triggered recalibration.

## Physical Scenario Shape

| Physical scenario | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `athermal_compensated_mrr_nominal` | 1728 | 0 | 0.8591 | 0.872522 | 0.0126 |
| `athermal_compensated_mrr_readout_edge` | 1728 | 0 | 0.8608 | 0.872505 | 0.0139 |
| `athermal_compensated_mrr_sparse_refresh` | 1728 | 0 | 0.8610 | 0.872565 | 0.0146 |
| `athermal_compensated_mrr_thermal_edge` | 1728 | 0 | 0.8584 | 0.872070 | 0.0154 |

The worst physical scenario remains `athermal_compensated_mrr_thermal_edge`.
It clears the floor at 20 epochs, but it is still the right stress case to keep
visible in the manuscript.

## Worst Summary Rows

The lowest summary rows all remain above the floor:

| Source coding | Calibration samples | Recalibration policy | Physical scenario | Mean accuracy | Minimum accuracy | Below-85 steps | Rows |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `redundant_6bit_2phase` | 2048 | `scheduled_only` | `athermal_compensated_mrr_thermal_edge` | 0.871004 | 0.8584 | 0 | 96 |
| `redundant_6bit_2phase` | 2048 | `scheduled_only` | `athermal_compensated_mrr_nominal` | 0.871555 | 0.8591 | 0 | 96 |
| `redundant_6bit_2phase` | 2048 | `scheduled_plus_projection_mse_trigger` | `athermal_compensated_mrr_thermal_edge` | 0.871089 | 0.8591 | 0 | 96 |

## Decision

Gate A is now cleared for the modeled full-split, 12-seed, 20-epoch
source-coding calibration stress confirmation.

The manuscript can claim expanded-seed robustness for this specific 20-epoch
training setting, subject to the existing scope limits: Fashion-MNIST workload,
the modeled athermal-compensated MRR physical scenarios, the current
source-coding assumptions, and the specific calibration policies tested here.

The next paper-gate item should move to calibration-sample sensitivity rather
than rerunning more seeds immediately.
