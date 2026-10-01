# 20-Epoch Calibration-Sample Sensitivity Result

Date: 2026-08-12

Purpose: test whether the 20-epoch source-coding stress result depends on a
particular calibration sample count by sweeping from `32` to `4096` samples
across the same 12-seed, full-split matrix.

## Command

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --seeds 7,11,13,17,19,23,29,31,37,41,43,47 \
  --epochs 20 \
  --calibration-sample-counts 32,64,128,256,512,1024,2048,4096 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-sensitivity-12seed-20epoch-2026-08-11
```

## Artifact Paths

The generated artifact directory is ignored by git:

```text
artifacts/source-coding-calibration-sensitivity-12seed-20epoch-2026-08-11/source_coding_calibration_stress/
```

Key files:

- `source_coding_calibration_stress_rows.csv`;
- `source_coding_calibration_stress_summary.csv`;
- `source_coding_calibration_stress_summary.md`;
- `source_coding_calibration_stress_manifest.json`;
- `seed_slices/seed_<n>_rows.csv`;
- `seed_slices/seed_<n>_manifest.json`.

## Result

The calibration-sample sensitivity pass clears the strict floor for every
tested sample count.

| Metric | Value |
| --- | ---: |
| Seeds completed | 12 |
| Calibration sample counts | 8 |
| Row observations | 18432 |
| Summary rows | 192 |
| Passing summary rows | 192 |
| Below-85 row observations | 0 |
| Lowest passing source bits | 6 |
| Minimum calibrated accuracy | 0.8574 |
| Mean calibrated accuracy | 0.872348 |

The expanded 20-epoch result is not supported only by the original
`128,512,2048` calibration sample choices. It still passes with the wider
`32,64,128,256,512,1024,2048,4096` sweep.

## Seed Summary

| Seed | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Trained-chain accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 1536 | 0 | 0.8611 | 0.867201 | 0.8737 |
| 11 | 1536 | 0 | 0.8677 | 0.875676 | 0.8822 |
| 13 | 1536 | 0 | 0.8703 | 0.875897 | 0.8837 |
| 17 | 1536 | 0 | 0.8638 | 0.870007 | 0.8797 |
| 19 | 1536 | 0 | 0.8690 | 0.875502 | 0.8815 |
| 23 | 1536 | 0 | 0.8574 | 0.865160 | 0.8665 |
| 29 | 1536 | 0 | 0.8617 | 0.869550 | 0.8773 |
| 31 | 1536 | 0 | 0.8745 | 0.879030 | 0.8843 |
| 37 | 1536 | 0 | 0.8666 | 0.871891 | 0.8792 |
| 41 | 1536 | 0 | 0.8677 | 0.874238 | 0.8825 |
| 43 | 1536 | 0 | 0.8655 | 0.872394 | 0.8801 |
| 47 | 1536 | 0 | 0.8642 | 0.871633 | 0.8778 |

Seed `23` remains the weakest seed. It still clears the floor, but only with a
`0.74` percentage point minimum-accuracy margin.

## Calibration-Sample Shape

| Calibration samples | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 2304 | 0 | 0.8611 | 0.871667 | 0.0159 |
| 64 | 2304 | 0 | 0.8617 | 0.872361 | 0.0156 |
| 128 | 2304 | 0 | 0.8620 | 0.872216 | 0.0154 |
| 256 | 2304 | 0 | 0.8574 | 0.872519 | 0.0127 |
| 512 | 2304 | 0 | 0.8607 | 0.872570 | 0.0140 |
| 1024 | 2304 | 0 | 0.8584 | 0.872539 | 0.0145 |
| 2048 | 2304 | 0 | 0.8584 | 0.872460 | 0.0131 |
| 4096 | 2304 | 0 | 0.8592 | 0.872455 | 0.0156 |

The weakest row occurs at `256` calibration samples, not at the lowest sample
count. The calibration-count sweep therefore does not show a simple monotonic
sample-count failure inside this modeled setting.

## Factor Shape

| Source coding | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_7bit` | 6144 | 0 | 0.8612 | 0.872892 | 0.0146 |
| `redundant_6bit_2phase` | 6144 | 0 | 0.8584 | 0.871228 | 0.0159 |
| `redundant_7bit_2phase` | 6144 | 0 | 0.8574 | 0.872925 | 0.0156 |

| Recalibration policy | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scheduled_only` | 9216 | 0 | 0.8574 | 0.872301 | 0.0157 |
| `scheduled_plus_projection_mse_trigger` | 9216 | 0 | 0.8591 | 0.872395 | 0.0159 |

| Physical scenario | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `athermal_compensated_mrr_nominal` | 4608 | 0 | 0.8591 | 0.872477 | 0.0136 |
| `athermal_compensated_mrr_readout_edge` | 4608 | 0 | 0.8599 | 0.872441 | 0.0142 |
| `athermal_compensated_mrr_sparse_refresh` | 4608 | 0 | 0.8605 | 0.872488 | 0.0146 |
| `athermal_compensated_mrr_thermal_edge` | 4608 | 0 | 0.8574 | 0.871987 | 0.0159 |

The weakest physical scenario remains `athermal_compensated_mrr_thermal_edge`.
The triggered recalibration policy gives a slightly better minimum row than
scheduled-only recalibration, but both policies pass.

## Worst Rows

| Seed | Calibration samples | Source coding | Recalibration policy | Physical scenario | Time step | Accuracy | Trained-chain accuracy | Drop vs trained |
| ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| 23 | 256 | `redundant_7bit_2phase` | `scheduled_only` | `athermal_compensated_mrr_thermal_edge` | 7 | 0.8574 | 0.8665 | 0.0091 |
| 23 | 1024 | `redundant_6bit_2phase` | `scheduled_only` | `athermal_compensated_mrr_thermal_edge` | 7 | 0.8584 | 0.8665 | 0.0081 |
| 23 | 2048 | `redundant_6bit_2phase` | `scheduled_only` | `athermal_compensated_mrr_thermal_edge` | 7 | 0.8584 | 0.8665 | 0.0081 |
| 23 | 2048 | `redundant_6bit_2phase` | `scheduled_only` | `athermal_compensated_mrr_nominal` | 4 | 0.8591 | 0.8665 | 0.0074 |
| 23 | 2048 | `redundant_6bit_2phase` | `scheduled_plus_projection_mse_trigger` | `athermal_compensated_mrr_thermal_edge` | 5 | 0.8591 | 0.8665 | 0.0074 |

## Decision

The 20-epoch paper-track Gate A result is robust to the tested calibration
sample counts from `32` through `4096`.

This supports a manuscript claim that, within the current Fashion-MNIST,
athermal-compensated MRR, source-coding, and calibration-policy scope, the
strict no-below-85 floor is not delicately dependent on calibration sample
count. The caveat is still margin: seed `23` is close enough to the floor that
factor-isolation ablations should be run before polishing the claim language.
