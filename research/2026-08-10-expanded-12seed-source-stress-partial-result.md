# Expanded 12-Seed Source-Coding Stress Partial Result

Date: 2026-08-10

Purpose: summarize the resumable Gate A rerun after the runner was changed to
write per-seed slices.

## Runner Change

The source-coding calibration stress runner now writes per-seed slice files
under:

```text
artifacts/source-coding-calibration-stress-full-12seed-10epoch-2026-08-10/source_coding_calibration_stress/seed_slices/
```

Each completed seed writes:

- `seed_<n>_rows.csv`
- `seed_<n>_manifest.json`

Reruns skip existing seed slices and combine available slices into the final
summary when the full requested set completes.

## Attempted Command

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --seeds 7,11,13,17,19,23,29,31,37,41,43,47 \
  --epochs 10 \
  --calibration-sample-counts 128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-full-12seed-10epoch-2026-08-10
```

## Completed Slices

The run completed four seed slices before being stopped:

| Seed | Rows | Below-85 rows | Minimum accuracy | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| 7 | 576 | 0 | 0.8536 | 0.860916 |
| 11 | 576 | 0 | 0.8644 | 0.869047 |
| 13 | 576 | 0 | 0.8574 | 0.861637 |
| 17 | 576 | 137 | 0.8458 | 0.850764 |

Combined partial rows:

- completed rows: `2304`;
- completed seeds: `7,11,13,17`;
- below-85 rows: `137`;
- minimum calibrated accuracy: `0.8458`;
- mean calibrated accuracy: `0.860591`.

## Gate A Verdict

Gate A fails.

The planned Gate A criterion was zero row-level observations below `85%` across
the expanded seed set. Seed `17` alone produced `137` below-floor rows. That is
enough to answer the strict gate question without finishing the remaining eight
seeds.

No claim should be made that the 10-epoch source-coding stress result survives
expanded seed robustness.

## Failure Shape For Seed 17

By source coding:

| Source coding | Rows | Below-85 rows | Minimum accuracy | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| `redundant_6bit_2phase` | 192 | 110 | 0.8458 | 0.849725 |
| `direct_7bit` | 192 | 16 | 0.8480 | 0.851316 |
| `redundant_7bit_2phase` | 192 | 11 | 0.8473 | 0.851249 |

By calibration samples:

| Calibration samples | Rows | Below-85 rows | Minimum accuracy | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| 128 | 192 | 49 | 0.8475 | 0.850698 |
| 512 | 192 | 47 | 0.8473 | 0.850767 |
| 2048 | 192 | 41 | 0.8458 | 0.850826 |

By recalibration policy:

| Recalibration policy | Rows | Below-85 rows | Minimum accuracy | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| `scheduled_only` | 288 | 69 | 0.8458 | 0.850789 |
| `scheduled_plus_projection_mse_trigger` | 288 | 68 | 0.8472 | 0.850738 |

By physical scenario:

| Physical scenario | Rows | Below-85 rows | Minimum accuracy | Mean accuracy |
| --- | ---: | ---: | ---: | ---: |
| `athermal_compensated_mrr_thermal_edge` | 144 | 41 | 0.8458 | 0.850697 |
| `athermal_compensated_mrr_sparse_refresh` | 144 | 35 | 0.8474 | 0.850742 |
| `athermal_compensated_mrr_nominal` | 144 | 34 | 0.8478 | 0.850792 |
| `athermal_compensated_mrr_readout_edge` | 144 | 27 | 0.8478 | 0.850824 |

## Interpretation

This does not look like a simple calibration-sample-count problem. Increasing
from `128` to `2048` samples reduces the below-floor count only modestly
(`49` to `41`) and does not lift the minimum above `85%`.

This also does not look like the projection-MSE trigger solving the issue.
Scheduled-only and triggered-policy rows are nearly tied on below-floor count
(`69` vs `68`).

The first-order interpretation is that seed `17` has weak training margin under
the current source-coding stress matrix. The strongest failure concentration is
`redundant_6bit_2phase`, but direct and redundant 7-bit branches also fail, so
the issue is broader than the 6-bit redundant branch alone.

## Manuscript Consequence

The paper cannot say the current 10-epoch result is robust to expanded seeds.

The defensible next step is a seed-17 diagnosis:

- compare seed `17` trained-chain baseline margin against seeds `7,11,13`;
- test whether a bounded training-margin fix, such as more epochs or adjusted
  learning schedule, clears seed `17`;
- if it clears, rerun a bounded expanded-seed confirmation through the
  resumable runner;
- if it does not clear, narrow the paper to a methods/architecture limits result
  and report seed sensitivity as a central limitation.

## Decision

Treat Gate A as failed under the current 10-epoch expanded-seed attempt. The
next item should diagnose seed `17` as a training-margin/generalization failure
before spending more compute on the remaining expanded seeds.
