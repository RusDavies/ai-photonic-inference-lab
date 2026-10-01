# Source-Coding Calibration Ablation Report

Access date: 2026-09-03

## Inputs

- Source rows:
  `artifacts/source-coding-calibration-sensitivity-12seed-20epoch-2026-08-11/source_coding_calibration_stress/source_coding_calibration_stress_rows.csv`
- Reduction command:
  `PYTHONPATH=src python -m optical_spike --run-source-coding-ablation-report --output-dir artifacts/source-coding-ablation-12seed-20epoch-2026-09-03`
- Generated report:
  `artifacts/source-coding-ablation-12seed-20epoch-2026-09-03/source_coding_ablation/source_coding_ablation_report.md`

## Result

The ablation reduction covered `18,432` row observations and `2,304` run
scenarios from the 12-seed, 20-epoch source-coding calibration sensitivity
matrix. No row fell below the `85%` accuracy floor. The minimum calibrated
accuracy remained `85.74%`, with mean calibrated accuracy `87.23%`.

Factor ranking by accuracy-range and worst minimum margin:

| factor | levels | accuracy_range_pp | worst_min_margin_pp | worst_drop_vs_trained_pp | steps_below_floor | worst_level |
| --- | --- | --- | --- | --- | --- | --- |
| training_margin | 12 | 1.38693 | 0.74 | 1.59 | 0 | seed=23, trained_chain_accuracy=0.8665 |
| source_coding | 3 | 0.169714 | 0.74 | 1.59 | 0 | source_coding=redundant_7bit_2phase, source_bits=7, source_redundant_phases=2 |
| recalibration_policy | 16 | 0.101658 | 0.74 | 1.59 | 0 | calibration_samples=256, recalibration_policy=scheduled_only |
| thermal_drift | 2 | 0.0481952 | 0.74 | 1.59 | 0 | thermal_drift_rate=0.005, thermal_excursion_k_per_step=1.7595307917888563 |
| detector_noise | 2 | 0.0481952 | 0.74 | 1.59 | 0 | detector_noise=0.03, detector_photons_per_sample=1111.111111111111 |
| output_readout_noise | 2 | 0.0123517 | 0.74 | 1.59 | 0 | output_readout_noise=0.005, output_enob_equivalent=7.352259121807248 |

## Interpretation

The separable report does not show a hidden single hardware factor breaking the
gate. Thermal drift and detector noise move together in the current
architecture-scenario grid, and their aggregate effect is only about `0.05`
percentage points across tested levels. Output readout noise is smaller again at
about `0.01` percentage points.

Source-coding choice matters more than the physical-noise axes, but still only
changes mean accuracy by about `0.17` percentage points across the tested
coding levels. Direct 7-bit coding has the healthiest minimum margin (`1.12`
percentage points over the floor), while redundant 6-bit and redundant 7-bit
both remain passing.

The main paper-relevant sensitivity remains trained-chain margin. Seed `23` is
the weakest 20-epoch chain and sets the observed floor at `85.74%`, despite all
factor combinations passing. The right framing is therefore not "readout,
thermal, or detector noise is solved"; it is "under the modeled athermal MRR
profile, source-chain and calibration stress are survivable when the trained
chain has enough headroom."

The older training-margin history points the same way:

- The original 5-epoch/full source-coding stress result had persistent failures
  concentrated in seed `11`, whose trained chain was only `85.19%`.
- The 10-epoch seed-`11` mitigation lifted that trained chain to `87.53%`; the
  worst stressed accuracy became `86.80%`, and all `72` seed-`11` summary rows
  passed.
- The later expanded 12-seed, 10-epoch attempt then exposed seed `17` at
  `85.62%` trained-chain accuracy, with `137` below-floor rows and minimum
  calibrated accuracy `84.58%`.
- The 20-epoch seed-`17` mitigation lifted that trained chain to `87.97%`; the
  single-seed stress matrix cleared with minimum calibrated accuracy `86.43%`.
- The full 12-seed, 20-epoch confirmation cleared, but seed `23` remains
  marginal at `86.65%` trained-chain accuracy and `85.74%` worst stressed
  accuracy in the wider calibration-sample sensitivity pass.

So the factor-isolation answer is consistent across the 5-to-10 and
10-to-20-epoch history: the first-order failure mode is insufficient trained
chain headroom, while source coding, recalibration policy, thermal drift,
detector noise, and output readout noise decide how much of that headroom gets
spent.

## Implication

The ablation gate is cleared for the current Fashion-MNIST 4-block modeled
chain, but the paper claim should keep the training-margin dependency explicit.
The next experiment should test whether that headroom survives a harder
workload or whether the result must be scoped as Fashion-MNIST-only.
