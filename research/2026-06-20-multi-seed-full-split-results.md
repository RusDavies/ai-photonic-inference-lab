# Multi-Seed Full-Split Results

Date: 2026-06-20
Diagnostic refresh: 2026-06-21

## Run Context

Purpose: repeat the optical projection spike over multiple seeds on the full
Fashion-MNIST split.

Configuration:

- Seeds: 7, 11, 13.
- Dataset: Fashion-MNIST full split.
- Train samples: 60,000.
- Test samples: 10,000.
- Epochs: 5.
- Calibration samples: 512.
- Projection: 784x64.
- Quantization sweep: 8, 6, 4, 3, 2 bits.

Local artifacts:

- `artifacts/multi-seed-full-2026-06-20/multi_seed/multi_seed_rows.csv`
- `artifacts/multi-seed-full-2026-06-20/multi_seed/multi_seed_summary.csv`
- `artifacts/multi-seed-full-2026-06-20/multi_seed/multi_seed_summary.md`
- `artifacts/multi-seed-full-2026-06-20/multi_seed/multi_seed_manifest.json`
- `artifacts/multi-seed-full-diagnostics-2026-06-21/multi_seed/multi_seed_rows.csv`
- `artifacts/multi-seed-full-diagnostics-2026-06-21/multi_seed/multi_seed_summary.csv`
- `artifacts/multi-seed-full-diagnostics-2026-06-21/multi_seed/multi_seed_summary.md`
- `artifacts/multi-seed-full-diagnostics-2026-06-21/multi_seed/multi_seed_manifest.json`

The artifact directory is intentionally uncommitted.

## Aggregate Results

Baseline:

- Mean test accuracy: 86.72%.
- Test accuracy standard deviation: 0.33 percentage points.

Quantization:

- 8-bit: mean accuracy 86.74%, no measured loss.
- 6-bit: mean accuracy 86.67%, 0.05 point mean drop.
- 4-bit: mean accuracy 85.87%, 0.85 point mean drop.
- 3-bit: mean accuracy 62.35%, 24.37 point mean drop.
- 2-bit: mean accuracy 11.85%, 74.87 point mean drop.

Interpretation: the full-split multi-seed run moves the useful precision
boundary upward. 4-bit still passes the 1-2 point target. 3-bit is not viable.

Tunable optical mode:

- Mild/moderate scenarios stayed effectively at baseline.
- Severe raw: mean accuracy 84.27%, 2.45 point mean drop.
- Severe calibrated: mean accuracy 85.43%, 1.29 point mean drop.

Interpretation: tunable mode passes the current toy threshold after calibration,
even under the severe scenario.

Fixed fabricated mode:

- Moderate raw: mean accuracy 85.96%, 0.76 point mean drop.
- Moderate calibrated: mean accuracy 86.55%, 0.17 point mean drop.
- Severe raw: mean accuracy 82.26%, 4.46 point mean drop.
- Severe calibrated: mean accuracy 85.36%, 1.36 point mean drop.

Interpretation: fixed mode still looks viable at this scale, but severe fixed
fabrication has high run-to-run variation. Calibration matters.

Nonlinear variants:

- Mild clipping and intensity distortion stayed near baseline.
- Strong clipping stayed within 0.68 point after compensation.
- Strong saturation raw: 0.95 point mean drop.
- Strong saturation compensated: 2.21 point mean drop.

Interpretation: the earlier small-run result overstated polynomial compensation
for strong saturation. Compensation is not automatically beneficial; it needs
validation by nonlinearity type and seed.

Adaptation / combined-stress proxy:

- No adaptation: 0.48 point mean drop.
- Calibration-only: 0.25 point mean drop.
- Retrained downstream head: 1.17 point mean drop.
- Distillation: 0.60 point mean drop.
- Hardware-aware fine-tuning: 0.29 point mean improvement over baseline.

Interpretation: hardware-aware fine-tuning is the adaptation path to keep.
Retraining the head from scratch remains weaker.

## Feature-Level Diagnostic Refresh

The 2026-06-21 rerun used the same full-split seeds and added projection MSE,
cosine similarity, feature distribution drift, confusion-matrix deltas, and
prediction-change rates to the aggregate report.

Baseline rerun:

- Mean test accuracy: 86.72%.
- Test accuracy standard deviation: 0.33 percentage points.

Quantization diagnostics:

- 8-bit: projection MSE 0.0007, mean cosine 0.99998, prediction-change rate
  0.26%.
- 6-bit: projection MSE 0.0122, mean cosine 0.99961, prediction-change rate
  0.90%.
- 4-bit: projection MSE 0.3066, mean cosine 0.99047, prediction-change rate
  4.63%.
- 3-bit: projection MSE 6.2375, mean cosine 0.80038, prediction-change rate
  34.62%.
- 2-bit: projection MSE 15.2176, mean cosine 0.38146, prediction-change rate
  88.19%.

Interpretation: the accuracy cliff has a clear feature-fidelity signature. The
4-bit run still preserves the projection direction well enough for the digital
head to survive. The 3-bit run is not merely a small weight-noise problem; it
changes a large fraction of predictions and heavily distorts the projected
feature geometry.

Optical error diagnostics:

- Severe tunable raw: projection MSE 0.9628, mean cosine 0.97165, prediction
  changes 8.65% of test examples.
- Severe tunable calibrated: projection MSE 0.3294, mean cosine 0.98521,
  prediction changes 5.72%.
- Severe fixed raw: projection MSE 1.3703, mean cosine 0.95930, prediction
  changes 11.22%.
- Severe fixed calibrated: projection MSE 0.5174, mean cosine 0.97772,
  prediction changes 6.72%.

Interpretation: calibration is doing real feature-space repair, not only
nudging logits. It substantially reduces distribution drift and prediction
churn, especially in severe fixed and tunable cases.

Nonlinear diagnostics:

- Strong saturation raw: projection MSE 6.0804, mean cosine 0.96476,
  prediction-change rate 4.30%, accuracy drop 0.95 points.
- Strong saturation compensated: projection MSE 0.3387, mean cosine 0.98810,
  prediction-change rate 6.12%, accuracy drop 2.21 points.
- Strong clipping compensated: projection MSE 1.1114, mean cosine 0.97745,
  prediction-change rate 3.63%, accuracy drop 0.68 points.
- Strong intensity distortion compensated: projection MSE 0.0461, mean cosine
  0.99825, prediction-change rate 1.81%, accuracy drop 0.16 points.

Interpretation: feature-space repair does not guarantee classifier improvement
for every nonlinearity. Strong saturation is the warning case: polynomial
compensation improves projection-level metrics but worsens classification.
Future compensation needs downstream validation, not just projection MSE.

Combined-stress adaptation diagnostics:

- Calibration-only: projection MSE 0.0536, mean cosine 0.99757, prediction
  changes 2.01%, accuracy drop 0.25 points.
- No adaptation: projection MSE 0.1257, mean cosine 0.99648, prediction changes
  3.07%, accuracy drop 0.48 points.
- Hardware-aware fine-tuning: same projection distortion as no adaptation, but
  accuracy improves by 0.29 points despite changing 3.33% of predictions.
- Retrained downstream head: same projection distortion, prediction changes
  6.85%, and accuracy drops 1.17 points.

Interpretation: hardware-aware fine-tuning looks best because it adapts the head
without discarding the useful digital baseline. Head retraining from scratch
adds avoidable prediction churn.

## Updated Research Judgment

The multi-seed full-split run strengthens the qualified-go conclusion for
deeper simulation, with two corrections:

1. Treat 4-bit equivalent precision as the current lower practical boundary for
   this projection. Do not plan around 3-bit without changing the architecture.
2. Treat nonlinear compensation as conditional. It helped some distortions, but
   strong saturation compensation got worse on the full multi-seed run.

Still no-go for fabrication planning. The next work should measure feature
fidelity and move to a transformer-relevant block before physical geometry
claims.

The diagnostic refresh completes the feature-fidelity measurement step. The
next research move is now the transformer-relevant block target.
