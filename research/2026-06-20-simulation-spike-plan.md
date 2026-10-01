# Simulation Spike Plan

Date: 2026-06-20

## Purpose

Plan a small simulation spike to test whether a trained 784x64 projection from
a Fashion-MNIST classifier can be transferred into a simulated optical block
while preserving useful task performance under realistic optical-style errors.

This is a plan, not the implementation.

## Core Question

Can an optical implementation of a fixed learned projection preserve useful
trained-model behavior after quantization, fabrication variation, detector noise,
drift, crosstalk, and calibration error?

## Workload

Dataset:

- Fashion-MNIST.
- Standard 60,000 train / 10,000 test split.
- Hold out a calibration set from training data, for example 5,000 samples.

Baseline model:

- Input: `x`, flattened 28x28 image, shape `[784]`.
- Optical candidate layer: `W_opt`, shape `[784, 64]`.
- Optional digital bias: `b_opt`, shape `[64]`.
- Digital activation: ReLU first; GELU or saturating activation optional.
- Classifier head: linear or small two-layer MLP from 64 features to 10 classes.

Initial baseline target:

- Train digital model to a stable Fashion-MNIST accuracy.
- Exact baseline target can be set after a first run, but the model should be
  strong enough that optical degradation is meaningful.

## Simulation Modes

### Mode 0: Digital Baseline

Full precision digital inference.

Purpose:

- Establish baseline accuracy.
- Save trained `W_opt` and downstream classifier parameters.

### Mode 1: Quantized Digital Projection

Quantize `W_opt` without optical-specific noise.

Suggested sweeps:

- 8-bit symmetric quantization.
- 6-bit symmetric quantization.
- 4-bit symmetric quantization.
- 3-bit and 2-bit only if early results are robust.

Purpose:

- Separate ordinary weight quantization sensitivity from optical-specific
  effects.

### Mode 2: Tunable Optical Projection

Simulate a reconfigurable photonic implementation.

Error model:

- Weight-setting quantization.
- Additive phase/attenuation setting error.
- Small multiplicative gain error per input/output channel.
- Crosstalk matrix perturbation.
- Detector noise.
- Slow drift between calibration events.
- Optional calibration correction.

Purpose:

- Approximate a tunable photonic mesh or weight bank where weights can be
  adjusted and calibrated.

### Mode 3: Fixed Fabricated Projection

Simulate a static fabricated implementation.

Error model:

- One-time fabrication perturbation applied to `W_opt`.
- Global and per-channel optical loss.
- Alignment-like multiplicative input/output errors.
- Crosstalk.
- Detector noise.
- No direct weight tuning after fabrication.
- Optional digital post-calibration layer.

Purpose:

- Approximate a fixed diffractive/printed/bulk-waveguide implementation where
  cheap replication is the long-term target.

### Mode 4: Nonlinear Optical Variants

Apply nonlinear transfer variants to Mode 2 and Mode 3.

Variants:

- Saturating transmission: `y = y / (1 + alpha * abs(y))`.
- Detector clipping / saturation.
- Intensity-dependent phase-like distortion approximated as amplitude-dependent
  gain or mixing error.
- Two-stage compensating nonlinearity, trained or fit on calibration data.
- Hardware-aware fine-tuning around the nonlinear optical block.

Purpose:

- Test whether material nonlinearity is mainly harmful, useful as an activation,
  or recoverable through compensation.

## Error Sweep Parameters

Initial sweep dimensions:

- Quantization: 8, 6, 4, 3, 2 bits.
- Additive weight noise: 0.1%, 0.5%, 1%, 2%, 5% of weight scale.
- Multiplicative gain noise: 0.1%, 0.5%, 1%, 2%, 5%.
- Crosstalk: 0%, 0.1%, 0.5%, 1%, 2% off-diagonal leakage.
- Detector noise: low, medium, high relative to activation scale.
- Drift: none, mild, moderate, severe across evaluation batches.
- Nonlinearity strength: none, mild, moderate, severe.

Keep the first spike small by sweeping one or two dimensions at a time, then run
a combined-stress scenario for promising modes.

## Calibration and Adaptation

Calibration set:

- Use held-out training samples.
- Estimate per-output gain and bias correction.
- Optionally fit a small digital correction layer after the optical projection.

Adaptation variants:

1. No adaptation.
2. Calibration-only correction.
3. Retrain downstream classifier head with optical projection frozen.
4. Hardware-aware fine-tuning of surrounding digital layers.
5. Optional distillation from full-precision digital baseline.

The optical projection should remain constrained by the selected optical mode.
Do not silently turn it back into an unconstrained digital layer. That would be
cheating, and not even the fun kind.

## Primary Metrics

Functional preservation:

- Test accuracy vs digital baseline.
- Absolute accuracy drop.
- Target: less than 1-2 percentage points absolute accuracy loss after
  hardware-aware adaptation.
- Accuracy cliff location for each error family.
- Confusion-matrix changes vs baseline.
- Stability over repeated random seeds / fabrication draws.

Optical-block fidelity:

- Mean squared error between ideal projection and simulated optical projection.
- Cosine similarity of projected features.
- Feature distribution drift.
- Equivalent useful precision estimate.

Calibration burden:

- Calibration samples required.
- Correction strength required.
- Performance decay under drift.
- Recalibration frequency implied by drift scenarios.

Secondary estimates:

- Relative operation count moved into optics.
- Coarse latency estimate for optical projection.
- Coarse energy estimate for optical projection.
- Sensitivity of those estimates to source/detector/control overhead.

## Expected Plots and Tables

Minimum outputs:

- Baseline accuracy table.
- Accuracy drop vs quantization bits.
- Accuracy drop vs additive/multiplicative noise.
- Accuracy drop vs crosstalk.
- Accuracy drop vs detector noise.
- Tunable vs fixed comparison table.
- No adaptation vs calibration vs hardware-aware adaptation table.
- Nonlinearity variant comparison.
- One combined-stress scenario table.

Useful plots:

- Error tolerance curves.
- Accuracy cliff plot.
- Feature cosine similarity vs noise.
- Calibration recovery plot.
- Drift over evaluation batches.

## Kill Criteria

Stop or change target if:

- The 784x64 projection cannot tolerate 8-bit-like quantization without severe
  loss.
- Mild optical noise causes more than 5 percentage points absolute accuracy loss
  and adaptation cannot recover below the 1-2 point target.
- Fixed fabricated mode fails under small fabrication perturbations and cannot
  be rescued by calibration or downstream adaptation.
- Tunable mode requires unrealistically frequent recalibration.
- Nonlinear variants consistently reduce recoverability without producing any
  robustness benefit.

A kill result is useful. It means the next move is to change the transform,
increase dimensionality, use a different architecture, or abandon that optical
mapping before spending time on prettier lies.

## Go Criteria

Proceed to implementation planning if:

- Tunable mode preserves useful performance under plausible error levels.
- Fixed mode shows at least one realistic path to recovery through calibration
  or downstream adaptation.
- Hardware-aware adaptation gives measurable robustness improvement.
- Nonlinear compensation improves at least one meaningful failure mode.
- The results reveal a clear next architecture or fabrication assumption to
  investigate.

## Implementation Notes for Later

Likely stack:

- Python.
- PyTorch for the tiny classifier and hardware-aware fine-tuning.
- NumPy/PyTorch tensor transforms for optical error injection.
- Matplotlib or Plotly for result plots.
- A single reproducible script or notebook-equivalent script for the spike.

Keep the first implementation CPU-friendly. If the tiny Fashion-MNIST spike
needs a GPU cluster, we have already lost the plot in a very on-brand way.

