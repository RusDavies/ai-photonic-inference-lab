# Calibration Cost Model

Date: 2026-06-21

## Purpose

Model the practical cost of keeping the transformer-style MLP-up optical target
calibrated under drift. Previous runs showed that affine calibration repairs a
large fraction of optical transfer error. This pass asks what that repair costs
in samples, recalibration frequency, drift tolerance, and always-on digital
control overhead.

## Local Artifacts

- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_rows.csv`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.md`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_manifest.json`

The artifact directory is intentionally uncommitted.

## Model

Target:

- Transformer-style MLP-up projection, 64x256.
- Constrained fixed scenario: `signed_tiled_severe`.
- Seeds: 7, 11, and 13.
- Full Fashion-MNIST split.
- Epochs: 5.

Swept parameters:

- Calibration samples: 32, 128, 512, 2048.
- Recalibration interval: every 1, 4, or 16 simulated time steps.
- Drift rate: 0.0, 0.002, or 0.005 per time step, scaled by trained weight
  magnitude and square-root time.
- Time steps: 16.

Cost proxy:

- Each recalibration consumes `calibration_samples` examples.
- Each recalibration writes an affine correction for the 256-channel projection:
  256 gains and 256 biases, so 512 control values.
- Each inference pays a 512-operation digital correction: gain and bias per
  projection channel.

This is still a count model, not an energy model. It deliberately avoids fake
precision about ADC/DAC, controller, memory, and thermal-control power.

## Results

Baseline MLP-up mean accuracy across seeds: 86.86%.

Best mean accuracy by drift bucket:

- Drift 0.0: 128 samples, recalibrate every step, 86.78% mean accuracy,
  0.08 point mean drop.
- Drift 0.002: 128 samples, recalibrate every step, 86.78% mean accuracy,
  0.08 point mean drop.
- Drift 0.005: 32 samples, recalibrate every 16 steps, 86.75% mean accuracy,
  0.11 point mean drop.

Lowest-overhead useful setting:

- 32 samples, recalibrate every 16 steps.
- Calibration examples per time step: 2.
- Affine control writes per time step: 32.
- Always-on digital correction: 512 operations per inference.
- Mean calibrated drop:
  - Drift 0.0: 0.12 points.
  - Drift 0.002: 0.13 points.
  - Drift 0.005: 0.11 points.
- Worst mean drop over time at drift 0.005: 0.35 points.

128 samples is a conservative setting:

- 128 samples, recalibrate every 16 steps.
- Calibration examples per time step: 8.
- Affine control writes per time step: 32.
- Mean calibrated drop:
  - Drift 0.0: 0.09 points.
  - Drift 0.002: 0.09 points.
  - Drift 0.005: 0.12 points.

Larger calibration sets did not consistently help in this affine-only model.
512 and 2048 samples often matched or slightly trailed 128 samples. The likely
reason is that the correction is only per-channel gain/bias; more samples cannot
express unmodeled higher-order drift or crosstalk, so they mainly add overhead.

## Interpretation

Calibration sample count is not currently the dominant problem. The MLP-up block
stays inside the 1-point comfort zone even with small calibration sets and sparse
recalibration in this drift model.

The first real overhead concern is the always-on digital correction: 512 simple
operations per inference for a 256-channel projection. That is small beside a
digital 64x256 matrix multiply, but it is not free. The next cost question is
whether ADC/DAC, storage, controller wakeups, temperature stabilization, and
control writes wipe out the optical advantage. That needs an energy/latency
budget, not just an operation count.

Conservative planning setting: 128 calibration samples and recalibration every
16 drift steps. Aggressive setting: 32 samples and recalibration every 16 drift
steps. The aggressive setting looks surprisingly fine here, but it should not be
trusted until the drift model is tied to a physical material and thermal profile.

## Next Work

See `research/2026-06-21-energy-latency-budget.md` for the first energy and
latency budget pass. Next, refine the model with platform-specific measured
ADC/DAC, laser, detector, and control electronics data.
