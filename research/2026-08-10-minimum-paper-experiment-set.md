# Minimum Paper Experiment Set

Date: 2026-08-10

Purpose: define the smallest experiment package needed before drafting the
paper as a credible limits-and-co-design manuscript. This is a planning
artifact, not a claim that the experiments have been run.

Project files loaded for this decision:

- `README.md`
- `research/2026-08-10-paper-track-positioning.md`
- `research/2026-08-10-paper-outline.md`
- `research/2026-08-10-claim-evidence-ledger.md`
- `research/2026-08-10-paper-primary-source-matrix.md`
- `src/optical_spike/cli.py`
- `src/optical_spike/material_training.py`
- `src/optical_spike/transformer_mlp.py`
- `tests/test_cli.py`

## Decision

The paper needs four pre-draft experiment gates:

1. Expanded seed robustness for the current full Fashion-MNIST
   source-coding/calibration stress result.
2. A harder-workload transfer check, or a deliberate written decision to scope
   the paper as a Fashion-MNIST-only controlled transfer benchmark.
3. A 10-epoch calibration-sample sensitivity pass.
4. Ablations that separate source coding, drift, detector/readout noise,
   recalibration policy, and training margin.

The first and third gates are mostly runnable with the current CLI. The harder
workload gate and clean ablation gate need code exposure before the manuscript
can honestly claim they were tested.

Status 2026-10-02:

Gate B has partial diagnostic evidence, not a full pass. The project now has a
CIFAR-10 loader, a bounded spatial-feature target, and a stress matrix across
mild, moderate, and severe signed-tiled projection assumptions. Mild/moderate
preserve the bounded target after calibration; severe remains a failure
boundary. This is enough to avoid saying the project is Fashion-MNIST-only in
the codebase, but not enough to claim broad harder-workload success.

## Gate A: Expanded Seed Robustness

Question:

Does the 10-epoch source-coding confirmation still clear the strict no-below
`85%` gate when the seed set is expanded beyond `7,11,13`?

Why it matters:

The claim ledger currently flags the 3-seed result as too narrow for a strong
empirical robustness claim. The 10-epoch result cleared the modeled stress
matrix, but only over three seeds.

Runnable command:

```bash
python -m optical_spike --run-source-coding-calibration-stress \
  --seeds 7,11,13,17,19,23,29,31,37,41,43,47 \
  --epochs 10 \
  --calibration-sample-counts 128,512,2048 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-full-12seed-10epoch-2026-08-10
```

Minimum pass:

- zero row-level observations below `85%`;
- every summary row passes the strict no-below-`85%` gate;
- worst calibrated accuracy, mean calibrated accuracy, and per-seed minimums
  are recorded in a research note.

If it fails:

- diagnose whether failure clusters by seed, physical scenario, source-coding
  scheme, calibration count, or recalibration policy;
- narrow the paper to a methods/architecture demonstration unless a bounded
  training-margin fix is run and logged.

Claim unlocked if it passes:

The model clears the current full-split source-coding stress gate over a
12-seed Fashion-MNIST confirmation, still under simulation assumptions.

## Gate B: Harder-Workload Transfer Check

Question:

Does the chained optical MLP-up target survive a workload with more demanding
features than Fashion-MNIST?

Current code status:

The original 2026-08-10 blocker has been cleared. The data path now supports
CIFAR-10, and the CLI exposes dataset selection for the relevant baseline and
CIFAR feature-target paths. The remaining blocker is claim quality: the current
bounded CIFAR target is deterministic and spatial-feature based, not a modern
learned vision classifier.

Minimum implementation:

- keep the CIFAR-10 dataset path and Fashion-MNIST behavior covered by tests;
- document the exact bounded workload transformation and why it is harder than
  Fashion-MNIST but still narrow;
- classify each CIFAR result as positive, negative, or diagnostic before using
  it in manuscript positioning;
- add a stronger CIFAR target if the manuscript needs more than a bounded
  architecture diagnostic.

Candidate command shape after implementation:

```bash
python -m optical_spike --run-source-coding-calibration-stress \
  --dataset cifar10-grayscale \
  --seeds 7,11,13 \
  --epochs 10 \
  --calibration-sample-counts 512 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-stress-cifar10-gray-3seed-10epoch-2026-08-10
```

Minimum pass:

- baseline digital model produces a non-trivial, logged accuracy above a
  predeclared floor for the harder workload;
- optical stress result is compared against that workload-specific digital
  baseline, not against the Fashion-MNIST `85%` floor by reflex;
- the research note records whether the harder workload supports, weakens, or
  blocks any generalization language.

If it fails:

- either scope the paper explicitly to a controlled Fashion-MNIST transfer
  benchmark, or treat the failure as a central limitation;
- do not claim task-general photonic inference scaling.

Claim unlocked by the current bounded evidence:

Mild and moderate signed-tiled projection assumptions preserve a bounded
CIFAR-10 spatial-feature target after calibration. Severe signed-tiled
projection remains a failure boundary.

Claim not unlocked yet:

The architecture is not yet shown to generalize beyond a controlled
Fashion-MNIST result plus a bounded CIFAR diagnostic. The paper should still
avoid frontier-model, broad vision, or modern CIFAR-classifier claims.

## Gate C: 10-Epoch Calibration-Sample Sensitivity

Question:

How much does the passing 10-epoch result depend on the number of calibration
samples?

Why it matters:

The manuscript should not report a single calibration count as if calibration
were free. Calibration sample count is part of the system design.

Runnable command:

```bash
python -m optical_spike --run-source-coding-calibration-stress \
  --seeds 7,11,13 \
  --epochs 10 \
  --calibration-sample-counts 32,64,128,256,512,1024,2048,4096 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-calibration-sensitivity-3seed-10epoch-2026-08-10
```

Minimum pass:

- identify the smallest calibration-sample count that clears the strict gate;
- report the accuracy/MSE tradeoff for scheduled-only and
  scheduled-plus-triggered policies;
- state whether the current `512` and `2048` anchor counts are conservative,
  necessary, or overkill.

If it fails:

- the paper must describe calibration as a hard requirement rather than a
  minor implementation detail;
- add a targeted calibration-policy experiment before drafting.

Claim unlocked if it passes:

The paper can state a bounded calibration-sample envelope for the simulated
stress matrix.

## Gate D: Isolation Ablations

Question:

Which factor actually controls pass/fail behavior: source coding, thermal
drift, detector noise, output readout noise, recalibration policy, or training
margin?

Current code status:

The source-coding stress runner already varies source coding, calibration
sample count, and scheduled vs projection-MSE-triggered recalibration. Physical
scenario variation exists, but the CLI does not cleanly expose individual
drift/noise/readout ablation axes. Training-margin variation exists through
`--epochs`, but should be summarized as a designed ablation rather than inferred
from scattered historical runs.

Minimum implementation:

- add or expose ablation controls for:
  - source coding fixed to each source-coding scenario;
  - thermal drift only;
  - detector noise only;
  - output readout noise only;
  - scheduled-only vs projection-MSE-triggered recalibration;
  - 5-epoch vs 10-epoch training margin;
- write a summary table with per-factor deltas against the current best
  combined-stress configuration.

Candidate command shape after implementation:

```bash
python -m optical_spike --run-source-coding-ablation-report \
  --seeds 7,11,13 \
  --epochs 5,10 \
  --calibration-samples 512 \
  --source-coding-projection-mse-threshold 0.18 \
  --output-dir artifacts/source-coding-ablation-report-2026-08-10
```

Minimum pass:

- every ablation row records mean accuracy, worst calibrated accuracy,
  below-floor count, projection MSE, recalibration count, and energy/latency
  implications where applicable;
- the final research note identifies which factor is the dominant design wall.

If it fails:

- the paper can still be drafted as a limits paper, but not with clean causal
  language about which constraint dominates.

Claim unlocked if it passes:

The discussion can say which modeled constraints dominate the chained optical
island's viability, instead of merely listing every possible failure mode and
hoping nobody notices the fog machine.

## Recommended Execution Order

1. Run Gate A, the expanded 12-seed 10-epoch confirmation, because it is
   runnable now and directly tests the strongest existing result.
2. Run Gate C, the calibration-sample sensitivity, because it is also runnable
   now and produces a manuscript-ready table.
3. Implement Gate D's explicit ablation runner.
4. Implement Gate B's harder-workload support and run the transfer check.

This order keeps the first two steps inside current code, then spends
implementation effort only after the existing Fashion-MNIST result has survived
or failed the obvious robustness checks.

## Manuscript Posture After These Gates

If Gates A and C pass, but B and D remain undone:

The paper can be outlined, but not drafted as a final manuscript. It remains a
promising Fashion-MNIST simulation story with unresolved generalization and
causal-isolation gaps.

If Gates A, C, and D pass, but B fails or is deliberately scoped out:

Draft as a controlled-transfer limits paper. Put the Fashion-MNIST boundary in
the abstract/methods, not as a sheepish limitation at the end.

If Gate B remains at the current bounded-CIFAR diagnostic level:

Draft as a controlled-transfer limits paper with a bounded CIFAR diagnostic
section. Put the Fashion-MNIST primary-scope boundary and the CIFAR
mild/moderate-only architecture assumption in the abstract/methods.

If all four gates pass with a stronger CIFAR target:

Draft as a stronger simulation and co-design paper, while still avoiding any
physical-hardware demonstration claim.

## Decision

Use this four-gate set as the minimum pre-draft experiment plan. The next work
item should be Gate A: run the expanded 12-seed 10-epoch source-coding stress
confirmation and summarize the result.
