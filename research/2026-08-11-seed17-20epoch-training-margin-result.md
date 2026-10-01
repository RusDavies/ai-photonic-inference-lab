# Seed 17 20-Epoch Training-Margin Result

Date: 2026-08-11

Purpose: test whether stronger training clears the seed-`17` expanded-seed
Gate A failure diagnosed in
`research/2026-08-11-seed17-gate-a-diagnosis.md`.

## Command

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --seeds 17 \
  --epochs 20 \
  --calibration-sample-counts 128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-seed17-epochs20-2026-08-11
```

## Artifact Paths

The generated artifact directory is ignored by git:

```text
artifacts/source-coding-calibration-stress-seed17-epochs20-2026-08-11/source_coding_calibration_stress/
```

Key files:

- `source_coding_calibration_stress_rows.csv`;
- `source_coding_calibration_stress_summary.csv`;
- `source_coding_calibration_stress_summary.md`;
- `source_coding_calibration_stress_manifest.json`;
- `seed_slices/seed_17_rows.csv`;
- `seed_slices/seed_17_manifest.json`.

## Result

The bounded seed-`17` 20-epoch run clears the strict seed-level stress matrix.

| Metric | Value |
| --- | ---: |
| Rows | 576 |
| Summary rows | 72 |
| Passing summary rows | 72 |
| Below-85 row observations | 0 |
| Lowest passing source bits | 6 |
| Baseline accuracy | 0.8828 |
| Trained-chain accuracy | 0.8797 |
| Trained-chain margin above 85% | 0.0297 |
| Minimum calibrated accuracy | 0.8643 |
| Mean calibrated accuracy | 0.869960 |
| Maximum stress drop | 0.0154 |

The 20-epoch run lifts seed `17` from `85.62%` trained-chain accuracy in the
10-epoch attempt to `87.97%`, a `2.35` percentage point improvement. That moves
seed `17` above the passing completed seed range from the 10-epoch expanded
attempt and gives enough margin for the same source-coding and physical stress
matrix to remain above `85%`.

## Source-Coding Shape

| Source coding | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `direct_7bit` | 192 | 0 | 0.8651 | 0.870003 | 0.0146 |
| `redundant_6bit_2phase` | 192 | 0 | 0.8643 | 0.870116 | 0.0154 |
| `redundant_7bit_2phase` | 192 | 0 | 0.8657 | 0.869761 | 0.0140 |

The previously weakest branch, `redundant_6bit_2phase`, now clears all rows.

## Calibration And Policy Shape

| Calibration samples | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 192 | 0 | 0.8643 | 0.868813 | 0.0154 |
| 512 | 192 | 0 | 0.8657 | 0.870741 | 0.0140 |
| 2048 | 192 | 0 | 0.8666 | 0.870326 | 0.0131 |

| Recalibration policy | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `scheduled_only` | 288 | 0 | 0.8657 | 0.869898 | 0.0140 |
| `scheduled_plus_projection_mse_trigger` | 288 | 0 | 0.8643 | 0.870022 | 0.0154 |

The result is not dependent on a particular calibration sample count or
recalibration policy.

## Physical Scenario Shape

| Physical scenario | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| `athermal_compensated_mrr_nominal` | 144 | 0 | 0.8672 | 0.869994 | 0.0125 |
| `athermal_compensated_mrr_readout_edge` | 144 | 0 | 0.8658 | 0.870055 | 0.0139 |
| `athermal_compensated_mrr_sparse_refresh` | 144 | 0 | 0.8651 | 0.870122 | 0.0146 |
| `athermal_compensated_mrr_thermal_edge` | 144 | 0 | 0.8643 | 0.869669 | 0.0154 |

The worst case remains the thermal-edge scenario, but stronger training leaves
enough margin to clear it.

## Decision

The seed-`17` failure is fixable with stronger training in this bounded check.
The next scientific step is to rerun the expanded Gate A confirmation through
the resumable runner with the stronger training setting, not to narrow the
manuscript claim yet.

The expanded rerun should still be treated as an open gate until all requested
seeds complete. This single-seed result shows the training-margin fix is worth
testing; it does not by itself prove expanded-seed robustness.
