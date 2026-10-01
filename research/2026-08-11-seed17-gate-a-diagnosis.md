# Seed 17 Gate A Diagnosis

Date: 2026-08-11

Purpose: diagnose why seed `17` failed the expanded 10-epoch Gate A source-coding
stress attempt while seeds `7`, `11`, and `13` passed.

## Evidence Used

Completed per-seed slices from:

```text
artifacts/source-coding-calibration-stress-full-12seed-10epoch-2026-08-10/source_coding_calibration_stress/seed_slices/
```

The diagnosis compares:

- `seed_7_rows.csv` and `seed_7_manifest.json`;
- `seed_11_rows.csv` and `seed_11_manifest.json`;
- `seed_13_rows.csv` and `seed_13_manifest.json`;
- `seed_17_rows.csv` and `seed_17_manifest.json`.

## Trained-Chain Margin

| Seed | Baseline accuracy | Trained-chain accuracy | Margin above 85% | Minimum stressed accuracy | Mean stressed accuracy | Maximum stress drop | Below-85 rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `7` | 0.8763 | 0.8689 | 0.0189 | 0.8536 | 0.860916 | 0.0153 | 0 |
| `11` | 0.8768 | 0.8749 | 0.0249 | 0.8644 | 0.869047 | 0.0105 | 0 |
| `13` | 0.8770 | 0.8675 | 0.0175 | 0.8574 | 0.861637 | 0.0101 | 0 |
| `17` | 0.8735 | 0.8562 | 0.0062 | 0.8458 | 0.850764 | 0.0104 | 137 |

Seed `17` is different before the optical/source-coding stress is applied. Its
trained-chain accuracy is only `85.62%`, leaving `0.62` percentage points of
margin above the `85%` floor. The weakest passing seed, seed `13`, starts at
`86.75%`, leaving `1.75` percentage points of margin.

The observed maximum seed-17 stress drop is `1.04` percentage points. With the
same stress shape, seed `17` would need trained-chain accuracy of at least
`86.04%` to clear the observed minimum row. That is only a `0.42` percentage
point lift, but the passing completed seeds suggest a healthier target is closer
to the seed-13 margin, about `86.75%` or better.

## Source-Coding Shape

| Seed | Source coding | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `7` | `direct_7bit` | 192 | 0 | 0.8581 | 0.861168 | 0.0108 |
| `7` | `redundant_6bit_2phase` | 192 | 0 | 0.8536 | 0.860518 | 0.0153 |
| `7` | `redundant_7bit_2phase` | 192 | 0 | 0.8585 | 0.861064 | 0.0104 |
| `11` | `direct_7bit` | 192 | 0 | 0.8657 | 0.869304 | 0.0092 |
| `11` | `redundant_6bit_2phase` | 192 | 0 | 0.8651 | 0.868563 | 0.0098 |
| `11` | `redundant_7bit_2phase` | 192 | 0 | 0.8644 | 0.869276 | 0.0105 |
| `13` | `direct_7bit` | 192 | 0 | 0.8574 | 0.861444 | 0.0101 |
| `13` | `redundant_6bit_2phase` | 192 | 0 | 0.8590 | 0.861782 | 0.0085 |
| `13` | `redundant_7bit_2phase` | 192 | 0 | 0.8589 | 0.861685 | 0.0086 |
| `17` | `direct_7bit` | 192 | 16 | 0.8480 | 0.851316 | 0.0082 |
| `17` | `redundant_6bit_2phase` | 192 | 110 | 0.8458 | 0.849725 | 0.0104 |
| `17` | `redundant_7bit_2phase` | 192 | 11 | 0.8473 | 0.851249 | 0.0089 |

The 6-bit redundant branch concentrates most of the failures, but seed `17` also
fails under direct 7-bit and redundant 7-bit coding. That makes this broader
than a single source-coding variant.

## Calibration And Physical-Scenario Shape

Calibration sample count does not explain the failure:

| Seed | Calibration samples | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `17` | 128 | 192 | 49 | 0.8475 | 0.850698 | 0.0087 |
| `17` | 512 | 192 | 47 | 0.8473 | 0.850767 | 0.0089 |
| `17` | 2048 | 192 | 41 | 0.8458 | 0.850826 | 0.0104 |

The projection-MSE trigger also does not explain the failure:

| Seed | Recalibration policy | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `17` | `scheduled_only` | 288 | 69 | 0.8458 | 0.850789 | 0.0104 |
| `17` | `scheduled_plus_projection_mse_trigger` | 288 | 68 | 0.8472 | 0.850738 | 0.0090 |

The thermal-edge physical scenario is the worst row source, but failures are
present across all four physical scenarios:

| Seed | Physical scenario | Rows | Below-85 rows | Minimum accuracy | Mean accuracy | Maximum stress drop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `17` | `athermal_compensated_mrr_nominal` | 144 | 34 | 0.8478 | 0.850792 | 0.0084 |
| `17` | `athermal_compensated_mrr_readout_edge` | 144 | 27 | 0.8478 | 0.850824 | 0.0084 |
| `17` | `athermal_compensated_mrr_sparse_refresh` | 144 | 35 | 0.8474 | 0.850742 | 0.0088 |
| `17` | `athermal_compensated_mrr_thermal_edge` | 144 | 41 | 0.8458 | 0.850697 | 0.0104 |

## Diagnosis

Seed `17` is a trained-chain margin failure. Calibration sample count,
projection-MSE-triggered recalibration, and physical scenario all move the exact
miss size, but none is the first-order cause. The seed begins too close to the
floor, so ordinary source-coding and physical-stress drops push many rows below
`85%`.

This is the same class of problem as the earlier seed-`11` weak case, but seed
`17` is weaker under the 10-epoch expanded-seed run than seed `11` became after
the previous 10-epoch mitigation.

## Decision

Run a bounded seed-17 training-margin fix next. The smallest useful test is a
single-seed source-coding stress rerun for seed `17` with stronger training,
such as `15` or `20` epochs, using the same source-coding, calibration-sample,
recalibration-policy, and physical-scenario matrix.

If the stronger seed-17 run clears zero below-85 rows with trained-chain accuracy
at or above the passing-seed range, rerun the expanded Gate A confirmation
through the resumable runner. If it does not clear, narrow the manuscript claim
and treat seed sensitivity as a central limitation rather than an incidental
caveat.
