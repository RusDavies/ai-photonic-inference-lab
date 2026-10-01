# First Toy Workload and Success Metrics

Date: 2026-06-20

## Selected Workload

Use Fashion-MNIST with a tiny hybrid classifier:

1. Input: 28x28 grayscale image, flattened to 784 values.
2. Optical target: one fixed learned projection matrix from 784 input features
   to 64 optical features.
3. Digital wrapper: activation, normalization if needed, and a small digital
   classifier head from 64 features to 10 classes.

This keeps the first optical block focused on a single fixed matrix multiply /
projection while the rest of the system stays digital and easy to inspect.

## Why Fashion-MNIST Instead of MNIST

Fashion-MNIST is still small and cheap to run, but less trivial than digit
classification. It is a better first filter for whether the optical projection
preserves useful structure rather than merely succeeding on a solved toy.

MNIST can remain a sanity-check fallback if the simulation infrastructure itself
needs debugging.

## Why This Workload

This workload directly matches the research decisions:

- cheap inference first;
- transfer an existing trained component, not train optically from scratch;
- start with a fixed matrix multiply / projection;
- compare fixed and tunable optical modes;
- judge success by transform preservation under realistic optical errors before
  energy or latency claims.

It also gives a clean ablation path:

1. Digital baseline.
2. Digital model with quantized 784x64 projection.
3. Simulated tunable optical projection.
4. Simulated fixed fabricated optical projection.
5. Optical projection with unwanted nonlinearity.
6. Optical projection with trained compensating nonlinearity.

## Model Shape

Baseline model:

- `x`: flattened Fashion-MNIST image, shape `[784]`.
- `W_opt`: learned projection, shape `[784, 64]`.
- `b_opt`: optional bias, handled digitally unless optical bias encoding is
  explicitly tested.
- Activation: ReLU, GELU, or simple saturating nonlinearity in the digital
  wrapper.
- Classifier: small digital linear or two-layer head to 10 classes.

The first optical transfer target is `x @ W_opt`.

## Optical Simulation Modes

### Ideal Digital Baseline

Normal digital inference with full-precision `W_opt`.

### Quantized Baseline

Digital inference with `W_opt` quantized to levels that approximate plausible
phase, attenuation, or coupling settings.

### Tunable Optical Mode

Simulate a reconfigurable optical projection:

- quantized weights;
- calibration correction;
- drift over time;
- phase/attenuation setting error;
- crosstalk;
- detector noise.

### Fixed Fabricated Mode

Simulate a fixed fabricated projection:

- one-time weight mapping;
- fabrication variation;
- optical loss;
- alignment error;
- no per-inference weight tuning;
- optional post-fabrication digital calibration layer.

### Nonlinear Variants

For both tunable and fixed modes:

- unwanted saturating transfer curve;
- unwanted phase/intensity nonlinearity;
- detector saturation;
- trained compensating nonlinearity;
- hardware-aware fine-tuning around the nonlinear optical block.

## Primary Success Metrics

Functional preservation first:

- Absolute accuracy drop vs digital baseline after hardware-aware adaptation.
- Target: less than 1-2 percentage points absolute accuracy loss on
  Fashion-MNIST.
- Equivalent useful precision of the optical projection.
- Error tolerance curve across quantization, drift, loss, crosstalk, detector
  noise, and nonlinear transfer.
- Whether calibration or compensating nonlinearity measurably recovers
  performance.

Secondary metrics:

- Estimated energy per inference for the optical block.
- Estimated latency for the optical projection.
- Calibration frequency and overhead.
- Sensitivity to fixed vs tunable deployment assumptions.

## Decision Gates

Gate 1: Digital baseline

- Train the tiny hybrid classifier digitally.
- Establish baseline accuracy and stability.

Gate 2: Transfer tolerance

- Replace `W_opt` with simulated optical projection variants.
- Sweep errors and map performance cliffs.

Gate 3: Adaptation value

- Apply hardware-aware fine-tuning or distillation.
- Measure whether adaptation recovers meaningful performance.

Gate 4: Architecture comparison

- Compare tunable vs fixed assumptions.
- Identify whether fixed fabricated mode is viable or whether the first real
  prototype must be tunable.

Gate 5: Go / no-go for deeper simulation

Proceed only if the optical block preserves useful performance under plausible
error/noise ranges. If it does not, the next research action is to change the
target transform or architecture before pretending energy estimates matter.

## Immediate Output Expected From the Next Work Item

The next work item should produce a simulation spike plan, not yet a full
implementation. That plan should define:

- exact model architecture;
- datasets and splits;
- error models;
- nonlinear variants;
- success thresholds;
- expected plots/tables;
- what would count as a kill result.

