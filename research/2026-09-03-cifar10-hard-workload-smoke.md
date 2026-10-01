# CIFAR-10 Harder-Workload Smoke Check

Date: 2026-09-03

## Purpose

Add harder-workload support beyond Fashion-MNIST and run a bounded transfer
check, or explicitly keep the manuscript scoped to Fashion-MNIST if a real
harder-workload result is not available.

## Implementation

The optical spike now supports a dataset selector for real datasets:

```bash
PYTHONPATH=src python -m optical_spike --run-source-coding-calibration-stress --dataset cifar10 ...
```

The `--dataset cifar10` path loads CIFAR-10 Python batches from
`data/cifar10` by default. The legacy `--synthetic` flag still overrides real
dataset loading for smoke tests.

## Verification Runs

Unit and style verification:

```bash
PYTHONPATH=src python -m pytest tests/test_cli.py -q
PYTHONPATH=src python -m ruff check src/optical_spike/data.py src/optical_spike/baseline.py src/optical_spike/quantization.py src/optical_spike/tunable.py src/optical_spike/transformer_mlp.py src/optical_spike/cli.py src/optical_spike/material_training.py src/optical_spike/calibration_cost.py src/optical_spike/material_nonlinearity.py src/optical_spike/constrained_compare.py tests/test_cli.py
```

Result: `40` CLI/data tests passed, and Ruff passed.

The CIFAR-10 loader is covered by a fixture test that writes tiny local
`cifar-10-batches-py` files and verifies image/label shapes, dtype, and scaling
without downloading the real archive.

## Real CIFAR-10 Download

The real CIFAR-10 source was reachable:

```bash
curl -L --head --max-time 20 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

The response redirected to `https://cave.cs.toronto.edu/kriz/` and returned
`200 OK` with `Content-Length: 170498071`.

The download completed after `32m23s` at about `87 KB/s`, producing a
`163M` local archive at `data/cifar10/cifar-10-python.tar.gz`.

## Fixture-Based Transfer Path Check

A tiny CIFAR-10-format local fixture was generated under ignored artifacts and
used to exercise the non-Fashion-MNIST source-coding stress path:

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --dataset cifar10 \
  --data-dir artifacts/cifar10-fixture-2026-09-03 \
  --epochs 1 \
  --seeds 7 \
  --calibration-sample-counts 32 \
  --chain-lengths 4 \
  --max-train-samples 128 \
  --max-test-samples 64 \
  --output-dir artifacts/cifar10-source-coding-fixture-smoke-2026-09-03
```

Result:

| Metric | Value |
| --- | ---: |
| Rows | 192 |
| Summary rows | 24 |
| Passing summary rows | 0 |
| Below-85 rows | 192 |
| Minimum calibrated accuracy | 0.125 |
| Mean calibrated accuracy | 0.125 |
| Trained-chain accuracy | 0.125 |

This fixture run verifies dataset routing and runner compatibility. It is not
scientific evidence about CIFAR-10 transfer quality.

## Real CIFAR-10 Bounded Transfer Check

The real CIFAR-10 archive was then used for the minimum bounded transfer check:

```bash
PYTHONPATH=src python -m optical_spike \
  --run-source-coding-calibration-stress \
  --dataset cifar10 \
  --epochs 1 \
  --seeds 7 \
  --calibration-sample-counts 32 \
  --chain-lengths 4 \
  --max-train-samples 512 \
  --max-test-samples 256 \
  --output-dir artifacts/cifar10-source-coding-smoke-2026-09-03
```

Result:

| Metric | Value |
| --- | ---: |
| Rows | 192 |
| Summary rows | 24 |
| Passing summary rows | 0 |
| Below-85 rows | 192 |
| Minimum calibrated accuracy | 0.0664 |
| Mean calibrated accuracy | 0.0877 |
| Trained-chain accuracy | 0.0781 |

This is a valid harder-workload routing check but not a valid optical-transfer
quality verdict. The one-epoch, 512-sample CIFAR-10 digital target did not
learn the task, landing at chance-level accuracy before the optical stress
question is meaningful.

## Manuscript Implication

The project now has a real CIFAR-10 harder-workload path and a bounded
CIFAR-10 run, but the bounded training recipe is inadequate for CIFAR-10. The
manuscript should remain scoped as Fashion-MNIST-only until a CIFAR-appropriate
digital baseline and optical transfer check exist.

## Follow-Up

Design and run a CIFAR-appropriate baseline or transfer target before using
CIFAR-10 for harder-workload evidence. Minimum next checks should first prove
the digital target learns CIFAR-10 above a credible floor, then rerun the
current 4-block source-coding stress runner against that trained target.
