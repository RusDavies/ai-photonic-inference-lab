# Photonic Inference Lab

Photonic Inference Lab is a small research codebase for testing whether
optical-style projection blocks can preserve useful neural-network behavior
under quantization, fabrication error, detector noise, drift, calibration
limits, converter costs, and source-coding constraints.

The current implementation is a NumPy simulation scaffold. It is not hardware
control software, not a product SDK, and not a claim that the modeled photonic
architecture is ready to build. Its purpose is narrower and more useful:
turn optimistic optical-inference ideas into runnable stress tests with clear
failure modes.

## What Is Included

- A command-line runner, `optical-spike`, for simulation experiments.
- Fashion-MNIST and CIFAR-10 dataset loaders, plus synthetic-data smoke tests.
- Digital baselines with optical-candidate projection layers.
- Quantization, tunable-error, fixed-fabrication, nonlinear, adaptation, and
  source-coding stress experiments.
- Energy, latency, calibration, converter, readout, and device-tolerance
  reports for candidate photonic inference chains.
- Research notes and a source log documenting assumptions, evidence, and open
  caveats.

## Status

This repository is research-grade. The Fashion-MNIST path has the most complete
evidence trail, including multi-seed stress tests, calibration sensitivity,
source-coding analysis, and device-tolerance notes. The CIFAR-10 work is newer
and more mixed: it has better routing and a stronger spatial-feature target,
but it does not yet provide a manuscript-grade harder-workload success story.

The main result so far is not a triumphant hardware claim. It is a limits-and-
co-design picture: optical projection chains can look promising under selected
conditions, but converter placement, source precision, calibration overhead,
trained-target margin, and physical drift/noise assumptions dominate whether
the idea survives contact with reality.

## Install

Create an isolated environment and install the package in editable mode:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

Run the CLI scaffold check:

```bash
.venv/bin/python -m optical_spike --dry-run
```

Run the test suite:

```bash
.venv/bin/python -m pytest
```

## Quick Start

Run a deterministic synthetic smoke test that does not download datasets:

```bash
.venv/bin/python -m optical_spike --run-baseline --synthetic --epochs 1 --max-train-samples 256 --max-test-samples 64
```

Run quick synthetic projection stress checks:

```bash
.venv/bin/python -m optical_spike --run-quantized-sweeps --synthetic --epochs 1 --quantization-bits 8,4
.venv/bin/python -m optical_spike --run-tunable-errors --synthetic --epochs 1 --calibration-samples 32
.venv/bin/python -m optical_spike --run-fixed-errors --synthetic --epochs 1 --calibration-samples 32
.venv/bin/python -m optical_spike --run-nonlinear-variants --synthetic --epochs 1 --calibration-samples 32
```

Generated metrics, plots, and intermediate outputs go under `artifacts/` by
default. Dataset archives and extracted data go under `data/`. Both directories
are intended to stay out of version control.

## Datasets

The code can download Fashion-MNIST and CIFAR-10 from their public upstream
sources when a non-synthetic run needs them. The datasets are not bundled in
this repository.

Use `--synthetic` for smoke tests that should avoid network downloads and large
local data.

## Reproducibility

See [docs/reproducibility.md](docs/reproducibility.md) for command recipes,
dataset notes, and expected output locations.

The research notes under `research/` record dated experiment summaries,
assumptions, and caveats. `research/source-log.md` tracks primary sources used
for device, converter, photonic, and calibration assumptions.

## Limitations

- The experiments are simulations, not hardware measurements.
- Energy and latency models are coarse estimates built from literature and
  product-source assumptions.
- Several reports reference generated artifacts that may need to be regenerated
  locally rather than downloaded from the repository.
- CIFAR-10 evidence is still exploratory and should not be oversold.
- The package API is not stable; the CLI and research scripts are the primary
  interface for now.

## Citation And Sources

If you use this work academically, cite the repository once a public release or
archive DOI exists, and cite the primary device/photonic/converter sources
listed in `research/source-log.md` for any hardware-facing claims.

Until a formal citation file is added, include the repository name, commit hash,
and access date in derived reports.
