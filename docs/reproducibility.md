# Reproducibility

This guide collects the public command recipes for reproducing smoke tests and
key experiment families in Photonic Inference Lab.

Commands assume an editable install from the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

## Sanity Checks

Print the planned simulation stages:

```bash
.venv/bin/python -m optical_spike --dry-run
```

Run tests:

```bash
.venv/bin/python -m pytest
```

Run a no-download baseline smoke test:

```bash
.venv/bin/python -m optical_spike --run-baseline --synthetic --epochs 1 --max-train-samples 256 --max-test-samples 64
```

## Baseline And Projection Stress

Run the Fashion-MNIST baseline:

```bash
.venv/bin/python -m optical_spike --run-baseline --epochs 5
```

Run quantized projection sweeps:

```bash
.venv/bin/python -m optical_spike --run-quantized-sweeps --epochs 5 --quantization-bits 8,6,4,3,2
```

Run tunable optical error sweeps:

```bash
.venv/bin/python -m optical_spike --run-tunable-errors --epochs 5 --calibration-samples 512
```

Run fixed fabricated optical error sweeps:

```bash
.venv/bin/python -m optical_spike --run-fixed-errors --epochs 5 --calibration-samples 512
```

Run nonlinear optical variant sweeps:

```bash
.venv/bin/python -m optical_spike --run-nonlinear-variants --epochs 5 --calibration-samples 512
```

Run calibration/adaptation mode sweeps:

```bash
.venv/bin/python -m optical_spike --run-adaptation-modes --epochs 5 --calibration-samples 512 --adaptation-epochs 8
```

Generate summary tables and plots:

```bash
.venv/bin/python -m optical_spike --generate-report --output-dir artifacts/spike
```

Run the full spike over multiple seeds:

```bash
.venv/bin/python -m optical_spike --run-multi-seed --seeds 7,11,13 --epochs 5 --output-dir artifacts/spike
```

## Transformer-Style And Chained Targets

Run a transformer-style MLP block target:

```bash
.venv/bin/python -m optical_spike --run-transformer-mlp-target --synthetic --epochs 1 --quantization-bits 8,4
```

Compare constrained fixed-fabrication assumptions across the original input
projection and transformer-style MLP-up target:

```bash
.venv/bin/python -m optical_spike --run-constrained-fixed-comparison --seeds 7,11,13 --epochs 5 --calibration-samples 512
```

Train MLP-up models end-to-end with material-inspired activation functions:

```bash
.venv/bin/python -m optical_spike --run-material-training --seeds 7,11,13 --epochs 5
```

Simulate repeated material MLP blocks with accumulated constrained optical
error:

```bash
.venv/bin/python -m optical_spike --run-functional-chain --seeds 7,11,13 --epochs 5 --chain-lengths 1,2,4 --calibration-samples 512
```

Train repeated material MLP blocks end-to-end for the chained objective:

```bash
.venv/bin/python -m optical_spike --run-chained-material-training --seeds 7,11,13 --epochs 5 --chain-lengths 2,4 --calibration-samples 512
```

Train separate material MLP weights for each chained block:

```bash
.venv/bin/python -m optical_spike --run-unshared-chained-material-training --seeds 7,11,13 --epochs 5 --chain-lengths 2,4 --calibration-samples 512
```

## Energy, Calibration, And Device Assumptions

Model calibration sample count, recalibration cadence, drift, and control
overhead:

```bash
.venv/bin/python -m optical_spike --run-calibration-cost --seeds 7,11,13 --epochs 5 --calibration-sample-counts 32,128,512,2048 --recalibration-intervals 1,4,16 --drift-rates 0.0,0.002,0.005 --time-steps 16
```

Estimate calibrated MLP-up energy and latency budgets:

```bash
.venv/bin/python -m optical_spike --run-energy-budget --calibration-summary-path artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv --inferences-per-drift-step 10000 --hidden-dim 64 --mlp-dim 256
```

Estimate energy and latency for chains sharing one converter boundary:

```bash
.venv/bin/python -m optical_spike --run-chained-block-budget --calibration-summary-path artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv --inferences-per-drift-step 10000 --hidden-dim 64 --mlp-dim 256 --chain-lengths 1,2,4,8
```

Compare CMOS column readout with CCD-style shared charge-bucket readout:

```bash
.venv/bin/python -m optical_spike --run-charge-readout-budget --calibration-summary-path artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv --inferences-per-drift-step 10000 --hidden-dim 64 --mlp-dim 256
```

Build an energy/accuracy Pareto report:

```bash
.venv/bin/python -m optical_spike --run-energy-accuracy-pareto-report --output-dir artifacts/energy-accuracy-pareto
```

## CIFAR-10 Experimental Path

Run a CIFAR-10 baseline smoke test:

```bash
.venv/bin/python -m optical_spike --run-baseline --dataset cifar10 --epochs 1 --max-train-samples 512 --max-test-samples 256
```

Run the CIFAR-10 spatial feature target:

```bash
.venv/bin/python -m optical_spike --run-cifar-feature-target --dataset cifar10 --epochs 30 --hidden-dim 256 --max-train-samples 10000 --max-test-samples 2000 --quantization-bits 8,6,4
```

Run the CIFAR-10 spatial feature source-coding stress check:

```bash
.venv/bin/python -m optical_spike --run-cifar-feature-source-coding-stress --dataset cifar10 --epochs 30 --hidden-dim 256 --max-train-samples 10000 --max-test-samples 2000 --calibration-sample-counts 128
```

The CIFAR-10 path is exploratory. Treat its current results as diagnostic
evidence for target selection and calibration stress, not as a finished
positive benchmark.

## Data And Outputs

- Dataset downloads go under `data/`.
- Generated reports, metrics, plots, and intermediate files go under
  `artifacts/`.
- Neither directory should be committed.
- Use `--synthetic` for no-download smoke tests.

Several long-running reports depend on intermediate artifact paths. If a command
references a missing CSV or metrics file, regenerate the upstream experiment
first or adjust the path to your local artifact location.
