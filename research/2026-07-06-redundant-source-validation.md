# Redundant 6-bit Source Validation

Date: 2026-07-06

## Question

The DAC source-coding sweep found only one surviving branch: 6-bit redundant
two-phase source coding. This note validates that branch inside the actual
unshared 4-block optical transfer simulation instead of only using the CSV-level
decision model.

## Implementation

Added `--run-redundant-source-validation`.

The validation reuses the unshared 4-block physical drift/noise evaluator with
the selected `athermal_compensated_mrr` physical scenarios. It injects source
coding directly into each MLP-up optical transfer:

- two dithered 6-bit source phases;
- averaged redundant source value before detector noise;
- calibration sees the same source-coded observed projection as evaluation;
- rows record source quantization MSE and simple source-DAC energy/latency
  accounting.

Energy/latency accounting for the source branch:

- DAC/source cost: about `4.08 pJ/input`
- shared-boundary source energy: about `261 pJ`
- per-block source energy: about `1045 pJ`
- source-coding latency: `0.2 ns`

Artifacts:

- `artifacts/redundant-source-validation-2026-07-06/redundant_source_validation/unshared_physical_drift_noise_rows.csv`
- `artifacts/redundant-source-validation-2026-07-06/redundant_source_validation/unshared_physical_drift_noise_summary.csv`
- `artifacts/redundant-source-validation-2026-07-06/redundant_source_validation/unshared_physical_drift_noise_summary.md`
- `artifacts/redundant-source-validation-2026-07-06/redundant_source_validation/unshared_physical_drift_noise_accuracy.png`
- `artifacts/redundant-source-validation-2026-07-06/redundant_source_validation/unshared_physical_drift_noise_projection_mse.png`

## Result

The full run used seeds `7,11,13`, 5 epochs, 512 calibration samples, and the
full Fashion-MNIST split. It generated 96 rows and 4 summary rows.

Mean calibrated accuracy stayed around 85.6% in every physical scenario, but the
strict seed/time-step floor was not clean:

- nominal: mean `85.66%`, min `84.91%`, 2 below-floor rows;
- readout edge: mean `85.63%`, min `84.93%`, 3 below-floor rows;
- sparse refresh: mean `85.67%`, min `85.03%`, 0 below-floor rows;
- thermal edge: mean `85.61%`, min `84.91%`, 2 below-floor rows.

The branch is therefore marginal. It is not dead, but it does not satisfy the
previous “no below-85 rows” robustness gate.

## Interpretation

The 6-bit redundant source path is the best DAC-relaxation branch so far, but it
is too close to the accuracy floor. Calling it solved would be spreadsheet
optimism with a lab coat.

The likely next lever is calibration quality, not another broad DAC search:

- increase or stratify calibration samples;
- recalibrate when projection MSE crosses a warning threshold;
- add a source-aware fine-tuning/adaptation mode;
- compare 6-bit redundant against 7-bit single or 7-bit redundant coding.

## Decision

Keep the branch alive, but mark it marginal. Do not treat 6-bit redundant source
coding as a robust DAC escape hatch yet.

## Next Step

Stress calibration policy and source precision around the marginal branch:
compare 6-bit redundant, 7-bit direct, and 7-bit redundant coding across
calibration sample count and projection-MSE-triggered recalibration.
