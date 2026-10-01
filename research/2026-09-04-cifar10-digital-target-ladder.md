# CIFAR-10 Digital Target Ladder

Date: 2026-09-04

## Purpose

Design and run a CIFAR-appropriate digital baseline or transfer target before
using CIFAR-10 as harder-workload optical-transfer evidence.

The previous CIFAR-10 source-coding smoke check verified the data and runner
path, but its one-epoch, 512-sample target stayed at chance-level accuracy.
This pass asks whether the existing project model families can produce a
bounded CIFAR-10 target that learns enough for optical-transfer stress results
to mean anything.

## Commands

Baseline linear-projection ladder:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-baseline \
  --dataset cifar10 \
  --epochs 5 \
  --max-train-samples 5000 \
  --max-test-samples 1000 \
  --output-dir artifacts/cifar10-digital-baseline-ladder-2026-09-04

PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-baseline \
  --dataset cifar10 \
  --epochs 20 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --output-dir artifacts/cifar10-digital-baseline-ladder-2026-09-04/epochs20-10k
```

Transformer-style MLP-up target ladder:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-transformer-mlp-target \
  --dataset cifar10 \
  --epochs 20 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --hidden-dim 128 \
  --mlp-dim 512 \
  --output-dir artifacts/cifar10-transformer-target-ladder-2026-09-04
```

Bounded source-coding transfer check against the stronger target recipe:

```bash
PYTHONPATH=src .venv/bin/python -m optical_spike \
  --run-source-coding-calibration-stress \
  --dataset cifar10 \
  --epochs 20 \
  --seeds 7 \
  --calibration-sample-counts 128 \
  --chain-lengths 4 \
  --max-train-samples 10000 \
  --max-test-samples 2000 \
  --hidden-dim 128 \
  --mlp-dim 512 \
  --output-dir artifacts/cifar10-source-coding-target-ladder-2026-09-04
```

## Results

| Target | Train/Test Slice | Epochs | Key Accuracy |
| --- | ---: | ---: | ---: |
| Linear projection baseline | 5,000 / 1,000 | 5 | 20.90% final test |
| Linear projection baseline | 10,000 / 2,000 | 20 | 35.15% best test, 30.20% final test |
| Transformer-style MLP-up target | 10,000 / 2,000 | 20 | 38.35% best test, 34.65% final test |
| MLP-up moderate fixed + hardware-aware tail fine-tuning | 10,000 / 2,000 | 20 | 40.30% adapted test |
| Unshared 4-block source-coding stress target | 10,000 / 2,000 | 20 | 32.55% trained-chain accuracy |

The bounded source-coding stress run wrote `192` row observations and `24`
summary rows. No summary row passed the old Fashion-MNIST `85%` gate, which is
expected because the CIFAR target itself is far below that floor.

Observed calibrated accuracy across the bounded CIFAR source-coding stress rows:

| Metric | Value |
| --- | ---: |
| Minimum calibrated accuracy | 29.65% |
| Mean calibrated accuracy | 31.39% |
| Maximum calibrated accuracy | 32.60% |

Best summary rows were the 7-bit source-coding cases, with calibrated mean
accuracy around `31.6%` and minimum accuracy around `31.0%` to `31.4%`.
The 6-bit redundant case remained close but slightly lower in this bounded run.

## Interpretation

The existing Fashion-MNIST-oriented dense projection and MLP-up targets can
learn real CIFAR-10 above chance, and they now provide a measurable bounded
transfer path. They are still too weak to support a harder-workload claim.

This result should therefore be treated as a negative target-selection result,
not as optical-transfer evidence. The manuscript should remain
Fashion-MNIST-only until the project has a CIFAR target with a substantially
stronger digital baseline, then reruns the optical/source-coding stress around
that learned target.

## Follow-Up

The next CIFAR step should add a stronger CIFAR target rather than keep turning
the current dense model crank. A small convolutional baseline, or a frozen
feature-extractor plus optical-candidate projection, is the next sensible
target because CIFAR-10 needs spatial inductive bias that the current flattened
projection models largely ignore.
