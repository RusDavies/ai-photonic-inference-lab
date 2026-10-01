# Optical Projection Spike Results Summary

Date: 2026-06-20

## Verdict

Decision: **qualified go for deeper simulation; no-go for fabrication planning
yet.**

The first spike says the core idea is not immediately ridiculous, which is a
useful bar to clear. A learned 784x64 projection survived moderate simulated
optical transfer errors with small or recoverable accuracy loss on a small
Fashion-MNIST run. However, this does not yet justify physical implementation
work. The model is tiny, the optical model is simplified, and the baseline is
only strong enough to expose degradation trends.

Proceed with a second simulation pass focused on stronger baselines, repeated
seeds, more realistic optical constraints, and architecture choices that could
scale beyond a single toy projection.

## Run Context

Command family:

- Baseline, quantized, tunable, fixed, nonlinear, adaptation, and report stages.
- Dataset: Fashion-MNIST.
- Train subset: 5,000 samples.
- Test subset: 1,000 samples.
- Epochs: 5.
- Calibration samples: 512.
- Adaptation epochs: 8.
- Hidden optical-candidate projection: 784x64.

Generated local artifacts:

- `artifacts/spike-summary-2026-06-20/reports/summary.csv`
- `artifacts/spike-summary-2026-06-20/reports/summary.md`
- `artifacts/spike-summary-2026-06-20/reports/report_manifest.json`
- Plot PNGs under `artifacts/spike-summary-2026-06-20/reports/`

The artifact directory is intentionally not committed.

## Key Results

Digital baseline:

- Train accuracy: 84.44%.
- Test accuracy: 82.30%.
- Test loss: 0.5010.

Quantized projection:

- 8-bit: 82.20%, 0.10 percentage-point drop.
- 6-bit: 82.40%, no meaningful loss in this run.
- 4-bit: 82.10%, 0.20 point drop.
- 3-bit: 82.10%, 0.20 point drop.
- 2-bit: 58.20%, 24.10 point drop.

Interpretation: the projection is tolerant down to roughly 3-4 bit equivalent
precision in this toy setup, but 2-bit collapses. That is a real cliff and a
useful design boundary.

Tunable optical mode:

- Mild and moderate scenarios stayed within noise of baseline.
- Severe raw scenario dropped 2.6 points.
- Severe calibrated scenario recovered to a 0.9 point drop.

Interpretation: tunable transfer looks viable enough for deeper simulation.
Calibration helps under severe error, but the easy scenarios did not need much
rescue.

Fixed fabricated mode:

- Near-ideal, mild, and moderate scenarios stayed within 0.4 points of baseline.
- Severe raw scenario dropped 2.3 points.
- Severe calibrated scenario recovered to roughly baseline.

Interpretation: fixed fabricated mode is not dead at this tiny scale. That is
the most interesting result, because fixed/replicated optics is the path that
actually supports cheap inference.

Nonlinear variants:

- Mild saturation dropped 1.1 points; compensation recovered to 0.1 point drop.
- Strong saturation dropped 5.2 points; compensation recovered to 0.5 point
  drop.
- Strong detector clipping dropped 3.5 points; compensation recovered to 0.5
  point drop.
- Intensity-dependent distortion was mostly recoverable and sometimes within
  run noise.

Interpretation: nonlinearity is not automatically fatal. Simple polynomial
compensation recovered the worst tested nonlinear cases surprisingly well.
That does not mean material nonlinearity is magic. It means it is worth modeling
more honestly before giving up or celebrating.

Adaptation / combined-stress proxy:

- Moderate fixed adaptation scenario without adaptation: 82.30%, no measured
  drop.
- Calibration-only: 82.30%, no measured drop.
- Retrained downstream head from scratch: 72.70%, worse by 9.6 points.
- Hardware-aware fine-tuning: 82.40%, roughly baseline.
- Distillation: 82.60%, best mode in this run.

Interpretation: do not retrain a tiny head from scratch on limited adapted data
unless we enjoy self-sabotage with logging. Fine-tuning and distillation are the
right adaptation paths to keep.

## Go / No-Go Against Original Gates

Gate 1, digital baseline: pass for spike purposes. The model is not excellent,
but it is good enough to measure degradation.

Gate 2, transfer tolerance: pass. Quantized, tunable, and fixed transfer all
preserved useful behavior under mild/moderate settings.

Gate 3, adaptation value: partial pass. Calibration, fine-tuning, compensation,
and distillation were useful. Retraining the head from scratch failed.

Gate 4, architecture comparison: partial pass. Fixed mode stayed competitive
with tunable mode in this simplified model. That keeps the cheap replicated
hardware path alive.

Gate 5, deeper simulation: pass. Physical implementation: no-go.

## Main Caveats

- The optical model is still a numerical proxy, not a photonics simulator.
- The baseline uses only 5,000 training samples and 1,000 test samples.
- Runs use one seed; robustness over repeated seeds is unknown.
- No confusion matrix, feature cosine, energy, latency, thermal drift, or
  fabrication-layout constraint is modeled yet.
- The projection is only one small layer. This says little about transformer
  attention, MLP blocks, KV-cache behavior, memory bandwidth, or frontier-scale
  model plumbing.

## Next Research Actions

1. Repeat the spike over multiple seeds and full Fashion-MNIST to separate real
   tolerance from lucky noise.
2. Add feature-level metrics: projection MSE, cosine similarity, feature drift,
   and confusion-matrix deltas.
3. Replace the single linear projection toy with a more transformer-relevant
   block: MLP projection, attention Q/K/V projection, or low-rank adapter block.
4. Add stronger fixed-fabrication constraints: nonnegative intensity encoding,
   signed-weight representation, fan-in/fan-out limits, optical loss budgets,
   and routing/crosstalk constraints tied to a plausible geometry.
5. Model calibration cost explicitly: samples required, recalibration frequency,
   drift over time, and whether cheap inference survives the control overhead.
6. Keep material nonlinearity in scope, but treat it as a calibrated distortion
   or activation candidate until a real material model is selected.

## Recommendation

Continue the research spike. The best next question is not "can we 3D print a
frontier model?" It is:

Can a transformer-relevant block be mapped into a constrained optical transfer
model with repeatable accuracy loss under realistic signed-weight encoding,
loss, crosstalk, drift, and calibration overhead?

If the answer stays yes, only then start arguing about actual waveguide geometry
and fabrication. Physics has earned a second interview, not the keys to the lab.
