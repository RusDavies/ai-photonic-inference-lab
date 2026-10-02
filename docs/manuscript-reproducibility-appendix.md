# Manuscript Reproducibility Appendix

Date: 2026-10-02

This appendix collects the command shapes, artifact paths, seeds, assumptions,
and pass/fail gates for the current manuscript-facing evidence set. It is a
reader-facing reproduction map, not a claim that every generated artifact is
stored in this repository.

Generated metrics, plots, intermediate data, and downloaded datasets are kept
under ignored local directories such as `artifacts/` and `data/`. Re-run the
listed commands when an artifact path is missing locally.

## Environment

Commands assume an editable install from the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Basic verification:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m optical_spike --dry-run
```

## Evidence Status Classes

- `Checked`: supports a current manuscript-facing claim when reported with the
  listed caveats.
- `Diagnostic`: useful for scope, failure analysis, target selection, or
  architecture-boundary discussion, but not a standalone positive claim.
- `Exploratory`: useful development or search evidence; do not cite as a
  manuscript result without rerunning and promoting it through a checked gate.

## Checked Results

### Fashion-MNIST Source-Coding Stress Confirmation

Status: `Checked`

Primary claim:

The current Fashion-MNIST simulation branch cleared the strict source-coding
stress gate for the 3-seed, 10-epoch confirmation under modeled calibration and
physical assumptions.

Research anchors:

- [`research/2026-08-10-claim-evidence-ledger.md`](../research/2026-08-10-claim-evidence-ledger.md)
- [`research/2026-08-10-paper-track-positioning.md`](../research/2026-08-10-paper-track-positioning.md)
- [`research/2026-08-10-paper-outline.md`](../research/2026-08-10-paper-outline.md)

Command:

```bash
.venv/bin/python -m optical_spike \
  --run-source-coding-calibration-stress \
  --seeds 7,11,13 \
  --epochs 10 \
  --calibration-sample-counts 128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-full-3seed-10epoch-2026-07-06
```

Artifacts:

- `artifacts/source-coding-calibration-stress-full-3seed-10epoch-2026-07-06/`

Seeds:

- `7,11,13`

Assumptions:

- Fashion-MNIST controlled transfer benchmark.
- Simulation result, not physical hardware measurement.
- Source-coding projection MSE threshold: `0.18`.
- Calibration sample counts: `128`, `512`, `2048`.
- Stronger 10-epoch training is treated as a training-margin mitigation, not a
  hardware fix.

Pass/fail gate:

- `0` row-level observations below `85%`.
- Every summary row passes the strict no-below-`85%` gate.
- Worst calibrated accuracy, mean calibrated accuracy, and per-seed minimums
  are recorded.

Recorded result:

- `1,728` rows and `72` summary rows.
- `0` below-`85%` row-level observations.
- Worst calibrated accuracy `85.96%`.
- Mean calibrated accuracy `86.81%`.

Caveats:

- Seed count remains limited to `7,11,13`.
- The claim is Fashion-MNIST scoped.
- Energy and hardware-facing conclusions require the converter/device evidence
  below.

### Chained Optical Island Energy/Accuracy Pareto

Status: `Checked`

Primary claim:

The unshared 4-block chain is the strongest local chain target because it
combines usable modeled accuracy with converter amortization.

Research anchors:

- [`research/2026-06-21-functional-chained-blocks.md`](../research/2026-06-21-functional-chained-blocks.md)
- [`research/2026-08-10-claim-evidence-ledger.md`](../research/2026-08-10-claim-evidence-ledger.md)
- [`research/2026-08-10-paper-outline.md`](../research/2026-08-10-paper-outline.md)

Commands:

```bash
.venv/bin/python -m optical_spike \
  --run-unshared-chained-material-training \
  --seeds 7,11,13 \
  --epochs 5 \
  --chain-lengths 2,4 \
  --calibration-samples 512

.venv/bin/python -m optical_spike \
  --run-energy-accuracy-pareto-report \
  --output-dir artifacts/energy-accuracy-pareto-2026-06-22
```

Artifacts:

- `artifacts/unshared-chained-material-training-2026-06-22/`
- `artifacts/energy-accuracy-pareto-2026-06-22/`

Seeds:

- `7,11,13`

Assumptions:

- Fashion-MNIST controlled transfer benchmark.
- Converter amortization is modeled by keeping multiple optical operations
  inside one analog island.
- Energy and latency are model estimates, not measurements.

Pass/fail gate:

- The candidate must retain useful calibrated accuracy while improving modeled
  energy/latency relative to the selected digital chain.
- Converter placement must be explicit.

Recorded result:

- Unshared 4-block calibrated accuracy: `85.67%`.
- Best modeled energy ratio: `0.256993x` versus digital chain.
- Best modeled latency ratio: `0.3125x` versus digital chain.

Caveats:

- Energy/latency depend on converter assumptions and selected digital baseline.
- The result does not prove deployable hardware.

### Stabilized Device Assumption Boundary

Status: `Checked`

Primary claim:

The modeled tolerance envelope requires stabilized or compensated device
assumptions; passive open-loop resonant behavior is not a credible default.

Research anchors:

- [`research/2026-06-22-device-tolerance-reality-check.md`](../research/2026-06-22-device-tolerance-reality-check.md)
- [`research/2026-08-10-claim-evidence-ledger.md`](../research/2026-08-10-claim-evidence-ledger.md)

Commands:

```bash
.venv/bin/python -m optical_spike \
  --run-calibration-bridge-report \
  --output-dir artifacts/calibration-bridge-2026-06-22

.venv/bin/python -m optical_spike \
  --run-unshared-physical-drift-noise \
  --physical-architecture-profile athermal_compensated_mrr \
  --seeds 7,11,13 \
  --epochs 5 \
  --calibration-samples 512 \
  --output-dir artifacts/athermal-physical-drift-noise-2026-06-22
```

Artifacts:

- `artifacts/calibration-bridge-2026-06-22/`
- `artifacts/athermal-physical-drift-noise-2026-06-22/`

Seeds:

- `7,11,13` for the physical drift/noise run.

Assumptions:

- MRR-style profiles are modeled through sourced anchors and bridge
  assumptions.
- Athermal-compensated MRR is a plausibility branch, not a measured local
  device.

Pass/fail gate:

- Device assumptions must be mapped into resonance, thermal, photon/SNR,
  readout, and recalibration quantities.
- Passive open-loop language must not be used for results that require active
  control or stabilization.

Recorded result:

- Tight/source-log-anchor MRR sensitivity rows require active control.
- Athermal profile maps recommended drift to about `0.704 K` per step and
  about `2.82 K` recalibration window while preserving the modeled readout
  burden.

Caveats:

- Device-specific measured anchors are still needed before hardware claims.

## Diagnostic Results

### CIFAR-10 Spatial-Feature Architecture Matrix

Status: `Diagnostic`

Primary use:

Use as bounded harder-workload architecture evidence. Do not use as broad
CIFAR-10, modern vision, or severe-constraint success.

Research anchors:

- [`research/2026-10-01-cifar-feature-stress-matrix.md`](../research/2026-10-01-cifar-feature-stress-matrix.md)
- [`research/2026-10-02-cifar-architecture-assumption.md`](../research/2026-10-02-cifar-architecture-assumption.md)

Command:

```bash
.venv/bin/python -m optical_spike \
  --run-cifar-feature-source-coding-stress \
  --dataset cifar10 \
  --epochs 30 \
  --hidden-dim 256 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --calibration-sample-counts 128 \
  --source-coding-projection-mse-threshold 0.18 \
  --constrained-fixed-scenario <signed_tiled_mild|signed_tiled_moderate|signed_tiled_severe> \
  --output-dir artifacts/PIL-001-cifar-feature-stress-matrix-2026-10-01/<scenario>
```

Artifacts:

- `artifacts/PIL-001-cifar-feature-stress-matrix-2026-10-01/signed_tiled_mild/`
- `artifacts/PIL-001-cifar-feature-stress-matrix-2026-10-01/signed_tiled_moderate/`
- `artifacts/PIL-001-cifar-feature-stress-matrix-2026-10-01/signed_tiled_severe/`

Seeds:

- Deterministic bounded CIFAR-10 slice configured by the runner.

Assumptions:

- CIFAR-10 target is a bounded deterministic spatial-feature target.
- Train/test slice: `10000 / 2000`.
- Calibration samples: `128`.
- Mild/moderate/severe are modeled architecture envelopes, not hardware
  measurements.

Pass/fail gate:

- Mild/moderate can support a bounded architecture diagnostic if they preserve
  the target and produce no below-`50%` rows.
- Severe remains a failure boundary unless a stronger calibration model or
  hardware-aware training method changes the result.

Recorded result:

- `signed_tiled_mild`: mean calibrated accuracy `52.76%`, minimum `51.70%`,
  `0 / 192` rows below `50%`.
- `signed_tiled_moderate`: mean calibrated accuracy `52.83%`, minimum `52.00%`,
  `0 / 192` rows below `50%`.
- `signed_tiled_severe`: mean calibrated accuracy `43.97%`, minimum `29.95%`,
  `192 / 192` rows below `50%`.

Caveats:

- The result does not demonstrate modern CIFAR-10 classification.
- The severe branch must be reported as a stress/failure boundary.

### CIFAR-10 Target-Selection Ladder

Status: `Diagnostic`

Primary use:

Use to explain why the project moved from weak flattened-pixel CIFAR targets to
the bounded spatial-feature target.

Research anchors:

- [`research/2026-09-04-cifar10-digital-target-ladder.md`](../research/2026-09-04-cifar10-digital-target-ladder.md)
- [`research/2026-09-04-cifar10-spatial-feature-target.md`](../research/2026-09-04-cifar10-spatial-feature-target.md)

Representative commands:

```bash
.venv/bin/python -m optical_spike \
  --run-cifar-feature-target \
  --dataset cifar10 \
  --epochs 30 \
  --hidden-dim 256 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --quantization-bits 8,6,4 \
  --output-dir artifacts/cifar10-feature-target-2026-09-04
```

Artifacts:

- `artifacts/cifar10-feature-target-2026-09-04/`
- earlier ladder artifacts recorded in the dated research notes.

Seeds:

- Deterministic bounded CIFAR-10 slice configured by the runner.

Assumptions:

- The target uses frozen spatial/color/edge features followed by a trainable
  optical-candidate projection and classifier head.

Pass/fail gate:

- Establish that the target is meaningfully stronger than the earlier bounded
  CIFAR candidates before running optical/source-coding stress.

Recorded result:

- CIFAR spatial-feature projection target: `53.30%` best test accuracy and
  `52.80%` final test accuracy on the bounded `10000 / 2000` slice.

Caveats:

- Not a modern CIFAR-10 classifier.
- Obvious train/test gap remains.

## Exploratory Results

Status: `Exploratory`

The following experiment families are useful background and development
evidence, but should not be treated as final manuscript claims unless rerun
under a checked gate:

- early baseline and quantized projection sweeps;
- tunable and fixed fabricated error sweeps;
- nonlinear optical variant sweeps;
- material activation strength sweeps;
- single-step synthetic smoke tests;
- current source/converter component searches that require source refresh before
  publication.

## Claims Not Supported By This Appendix

- Physical photonic accelerator demonstration.
- Passive open-loop resonant inference.
- Frontier-model or broad vision-model inference economics.
- CIFAR-10 success under `signed_tiled_severe`.
- Generalization beyond the controlled Fashion-MNIST result and bounded CIFAR
  diagnostic.
