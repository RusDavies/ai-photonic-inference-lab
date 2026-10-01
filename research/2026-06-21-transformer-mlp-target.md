# Transformer-Style MLP Block Target

Date: 2026-06-21

## Purpose

Test the next optical transfer target after the single Fashion-MNIST input
projection: a transformer-style feed-forward MLP expansion projection.

This is a sequence-free first step toward transformer relevance. Attention
Q/K/V projections need token/sequence structure, but the MLP expansion matrix is
a fixed dense projection and maps cleanly onto the current optical-block
diagnostics.

## Target

Digital model:

- Input: flattened Fashion-MNIST image.
- Embedding projection: 784x64.
- Residual MLP block:
  - MLP up projection: 64x256.
  - ReLU.
  - MLP down projection: 256x64.
  - Residual add and ReLU.
- Linear classifier head: 64x10.

Optical candidate:

- `mlp_up_projection`, matrix shape 64x256.
- This mirrors the expansion projection in a transformer feed-forward block.

Local artifacts:

- `artifacts/transformer-mlp-target-2026-06-21/transformer_mlp_target/transformer_mlp_target.json`
- `artifacts/transformer-mlp-target-2026-06-21/transformer_mlp_target/transformer_mlp_weights.npz`
- `artifacts/transformer-mlp-errors-2026-06-21/transformer_mlp_target/transformer_mlp_target.json`
- `artifacts/transformer-mlp-errors-2026-06-21/transformer_mlp_target/transformer_mlp_weights.npz`
- `artifacts/transformer-mlp-full-multiseed-2026-06-21/multi_seed/transformer_mlp_multi_seed_rows.csv`
- `artifacts/transformer-mlp-full-multiseed-2026-06-21/multi_seed/transformer_mlp_multi_seed_summary.csv`
- `artifacts/transformer-mlp-full-multiseed-2026-06-21/multi_seed/transformer_mlp_multi_seed_summary.md`
- `artifacts/transformer-mlp-full-multiseed-2026-06-21/multi_seed/transformer_mlp_multi_seed_manifest.json`
- `artifacts/transformer-mlp-constrained-fixed-2026-06-21/transformer_mlp_target/transformer_mlp_target.json`
- `artifacts/transformer-mlp-constrained-fixed-2026-06-21/transformer_mlp_target/transformer_mlp_weights.npz`
- `artifacts/constrained-comparison-full-2026-06-21/constrained_comparison/constrained_comparison_rows.csv`
- `artifacts/constrained-comparison-full-2026-06-21/constrained_comparison/constrained_comparison_summary.csv`
- `artifacts/constrained-comparison-full-2026-06-21/constrained_comparison/constrained_comparison_summary.md`
- `artifacts/constrained-comparison-full-2026-06-21/constrained_comparison/constrained_comparison_manifest.json`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_rows.csv`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.md`
- `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_manifest.json`

The artifact directory is intentionally uncommitted.

## Run Context

- Dataset: Fashion-MNIST subset.
- Train samples: 10,000.
- Test samples: 2,000.
- Epochs: 5.
- Hidden dimension: 64.
- MLP dimension: 256.
- Seed: 7.
- Quantization sweep: 8, 6, 4, 3, 2 bits.

Baseline test accuracy: 83.90%.

## Results

Quantized MLP-up projection:

- 8-bit: accuracy 83.90%, no measured drop, projection MSE 0.00016,
  mean cosine 0.99997, prediction-change rate 0.25%.
- 6-bit: accuracy 83.85%, 0.05 point drop, projection MSE 0.00294,
  mean cosine 0.99937, prediction-change rate 0.65%.
- 4-bit: accuracy 83.15%, 0.75 point drop, projection MSE 0.05049,
  mean cosine 0.98949, prediction-change rate 2.10%.
- 3-bit: accuracy 81.85%, 2.05 point drop, projection MSE 0.28968,
  mean cosine 0.94282, prediction-change rate 5.50%.
- 2-bit: accuracy 52.20%, 31.70 point drop, projection MSE 1.85616,
  mean cosine 0.55950, prediction-change rate 40.20%.

## Interpretation

The MLP expansion projection is a better next optical target than attention
Q/K/V for this codebase because it is transformer-relevant while remaining a
fixed dense matrix multiply.

Compared with the earlier 784x64 input projection, this 64x256 MLP-up target
looks more tolerant at 3-bit on the subset run: 3-bit loses about 2 points here,
where the full-split input projection lost about 24 points. That is not enough
to bless 3-bit generally; the run is smaller, the target is different, and no
fabrication/noise model has been applied yet. But it is enough to justify
continuing with this block family.

The immediate lower boundary is still 4-bit for conservative planning. It
stays within 1 point while preserving projection direction well. The 2-bit run
is unusable and changes too many predictions.

## Optical Error and Adaptation Refresh

The 2026-06-21 refresh added tunable optical errors, fixed fabricated optical
errors, affine calibration, and a hardware-aware tail fine-tuning check for the
same MLP-up target.

Tunable mode:

- Mild raw: accuracy 84.15%, 0.25 point improvement, projection MSE 0.00173,
  mean cosine 0.99962, prediction-change rate 0.60%.
- Mild calibrated: accuracy 84.25%, 0.35 point improvement, projection MSE
  0.00072, mean cosine 0.99980, prediction-change rate 0.50%.
- Moderate raw: accuracy 82.95%, 0.95 point drop, projection MSE 0.01987,
  mean cosine 0.99573, prediction-change rate 2.25%.
- Moderate calibrated: accuracy 83.60%, 0.30 point drop, projection MSE 0.00891,
  mean cosine 0.99761, prediction-change rate 1.30%.
- Severe raw: accuracy 80.55%, 3.35 point drop, projection MSE 0.29161,
  mean cosine 0.93947, prediction-change rate 6.35%.
- Severe calibrated: accuracy 82.05%, 1.85 point drop, projection MSE 0.11900,
  mean cosine 0.96953, prediction-change rate 5.10%.

Fixed fabricated mode:

- Mild fixed raw/calibrated stayed at or slightly above baseline.
- Moderate fixed raw: accuracy 82.25%, 1.65 point drop, projection MSE 0.06724,
  mean cosine 0.98541, prediction-change rate 4.10%.
- Moderate fixed calibrated: accuracy 83.10%, 0.80 point drop, projection MSE
  0.02713, mean cosine 0.99266, prediction-change rate 2.20%.
- Severe fixed raw: accuracy 73.10%, 10.80 point drop, projection MSE 0.58740,
  mean cosine 0.88105, prediction-change rate 15.25%.
- Severe fixed calibrated: accuracy 81.50%, 2.40 point drop, projection MSE
  0.21698, mean cosine 0.94508, prediction-change rate 6.30%.

Adaptation check, using the moderate fixed target:

- No adaptation: accuracy 82.25%, 1.65 point drop.
- Calibration-only: accuracy 82.90%, 1.00 point drop.
- Hardware-aware tail fine-tuning: accuracy 85.20%, 1.30 point improvement
  over the digital baseline.

Interpretation: calibration repairs a meaningful fraction of the MLP-up feature
distortion, and tail fine-tuning is the best recovery mode on this subset. This
matches the earlier single-projection result: adaptation around the hardware is
more useful than treating the optical block as a frozen drop-in replacement.

## Full-Split Multi-Seed Refresh

The full-split refresh repeated the transformer MLP target over seeds 7, 11,
and 13, using the full Fashion-MNIST split, 5 epochs, 512 calibration samples,
and 8 adaptation epochs.

Baseline:

- Mean test accuracy: 86.86%.
- Test accuracy standard deviation: 0.34 percentage points.

Quantized MLP-up projection:

- 8-bit: mean accuracy 86.85%, 0.01 point mean drop, projection MSE 0.00027,
  mean cosine 0.99996, prediction-change rate 0.09%.
- 6-bit: mean accuracy 86.86%, effectively no measured drop, projection MSE
  0.00451, mean cosine 0.99931, prediction-change rate 0.59%.
- 4-bit: mean accuracy 86.54%, 0.32 point mean drop, projection MSE 0.08665,
  mean cosine 0.98683, prediction-change rate 2.17%.
- 3-bit: mean accuracy 86.45%, 0.41 point mean drop, projection MSE 0.49384,
  mean cosine 0.93177, prediction-change rate 5.05%.
- 2-bit: mean accuracy 69.21%, 17.65 point mean drop, projection MSE 2.54055,
  mean cosine 0.49986, prediction-change rate 27.99%.

Tunable mode:

- Moderate raw/calibrated stayed within 0.03-0.04 points of baseline.
- Severe raw: mean accuracy 85.04%, 1.82 point drop, projection MSE 0.49288,
  mean cosine 0.93252, prediction-change rate 6.40%.
- Severe calibrated: mean accuracy 86.30%, 0.56 point drop, projection MSE
  0.21213, mean cosine 0.96359, prediction-change rate 4.04%.

Fixed fabricated mode:

- Moderate fixed raw: mean accuracy 86.77%, 0.09 point drop, projection MSE
  0.09314, mean cosine 0.98575, prediction-change rate 2.19%.
- Moderate fixed calibrated: mean accuracy 86.80%, 0.06 point drop, projection
  MSE 0.04526, mean cosine 0.99202, prediction-change rate 1.68%.
- Severe fixed raw: mean accuracy 84.54%, 2.32 point drop, projection MSE
  0.81332, mean cosine 0.88401, prediction-change rate 7.87%.
- Severe fixed calibrated: mean accuracy 85.99%, 0.87 point drop, projection
  MSE 0.35549, mean cosine 0.93932, prediction-change rate 5.29%.

Adaptation check, moderate fixed target:

- No adaptation: mean accuracy 86.73%, 0.13 point drop.
- Calibration-only: mean accuracy 86.81%, 0.05 point drop.
- Hardware-aware tail fine-tuning: mean accuracy 87.93%, 1.07 point improvement
  over the digital baseline.

Interpretation: the full-split result strengthens the MLP-up target. Unlike the
original 784x64 input projection, this 64x256 feed-forward expansion projection
survives 3-bit quantization on the tested task. Conservative planning should
still treat 4-bit as the low-risk boundary until stronger fabrication
constraints are modeled, but the MLP target is clearly more forgiving than the
first input projection.

Severe fixed fabrication remains damaging, but calibration pulls it back inside
the 1-point-ish tolerance band on the full-split mean. Hardware-aware tail
fine-tuning again looks best, which reinforces the pattern that optical blocks
should be integrated with adaptation rather than substituted blindly.

## Stronger Fixed-Fabrication Constraints

The stronger constrained fixed model adds:

- signed differential encoding for weights;
- fan-in and fan-out limits, expressed as routing tile counts;
- optical loss budgets with extra penalty when tile routing exceeds budget;
- geometry-linked crosstalk that scales with fan pressure;
- detector noise and differential path noise.

Subset run context:

- Train samples: 10,000.
- Test samples: 2,000.
- Epochs: 5.
- Calibration samples: 512.
- Target: MLP-up projection, 64x256.
- Baseline accuracy: 83.90%.

Constrained fixed results:

- Signed tiled mild:
  - Routing: 1 fan-in tile, 1 fan-out tile.
  - Effective loss: 2.0%.
  - Raw accuracy: 84.00%, 0.10 point improvement.
  - Calibrated accuracy: 83.90%, no measured drop.
  - Calibrated projection MSE: 0.00077, mean cosine 0.99978.
- Signed tiled moderate:
  - Routing: 1 fan-in tile, 2 fan-out tiles.
  - Effective loss: 6.0%.
  - Raw accuracy: 83.40%, 0.50 point drop.
  - Calibrated accuracy: 83.60%, 0.30 point drop.
  - Calibrated projection MSE: 0.00451, mean cosine 0.99876.
- Signed tiled severe:
  - Routing: 2 fan-in tiles, 4 fan-out tiles.
  - Required loss: 21.0%; effective loss with budget excess: 32.0%.
  - Geometry crosstalk scales to 8.0%.
  - Raw accuracy: 79.85%, 4.05 point drop.
  - Calibrated accuracy: 82.35%, 1.55 point drop.
  - Calibrated projection MSE: 0.04290, mean cosine 0.98883.

Interpretation: adding signed-path and routing constraints makes the fixed path
less magical, as desired. Moderate signed-tiled constraints remain viable on
the subset. Severe routing pressure is the first constrained case that clearly
pushes past the 1-point comfort zone even after calibration, mostly through
loss-budget excess and geometry-linked crosstalk.

## Full-Split Constrained Target Comparison

The full-split comparison repeated the stronger constrained fixed model over
seeds 7, 11, and 13 for two optical targets:

- original input projection: 784x64;
- transformer-style MLP-up projection: 64x256.

Run context:

- Train samples: 60,000.
- Test samples: 10,000.
- Epochs: 5.
- Calibration samples: 512.
- Per-seed rows: 42.
- Aggregate rows: 14.

Baseline mean accuracy:

- Input projection model: 86.72% +/- 0.33 points.
- MLP-up model: 86.86% +/- 0.34 points.

Constrained fixed comparison, mean accuracy drop:

- Signed tiled mild:
  - Input projection raw: 0.04 points; calibrated: 0.03 points.
  - MLP-up raw: 0.07 points; calibrated: 0.05 points.
- Signed tiled moderate:
  - Input projection raw: 3.73 points; calibrated: 0.23 points.
  - MLP-up raw: 0.08 points; calibrated: 0.01 points.
- Signed tiled severe:
  - Input projection raw: 72.33 points; calibrated: 4.69 points.
  - MLP-up raw: 0.58 points; calibrated: 0.18 points.

The routing/loss numbers explain the split:

- Input projection severe case:
  - Fan pressure: 24.5.
  - Effective loss: 95.0%.
  - Effective crosstalk: 49.0%.
  - Calibrated projection MSE: 1.35691.
  - Calibrated prediction changes: 11.74%.
- MLP-up severe case:
  - Fan pressure: 4.0.
  - Effective loss: 32.0%.
  - Effective crosstalk: 8.0%.
  - Calibrated projection MSE: 0.08849.
  - Calibrated prediction changes: 2.61%.

Interpretation: the original 784x64 input projection is too physically wide for
the severe constrained fixed model. Calibration can rescue some moderate damage,
but cannot turn a 95% effective-loss routing mess back into a credible optical
block. The 64x256 MLP-up target remains far more plausible because its fan
pressure and geometry-linked crosstalk are much lower under the same scenario
definitions. This makes the MLP-up block the better first fixed-fabrication
target, not just the better quantization target.

## Next Work

See `research/2026-06-21-calibration-cost.md` for the calibration cost pass.
Next, convert calibration/control counts into energy and latency budget
estimates using candidate hardware assumptions.
