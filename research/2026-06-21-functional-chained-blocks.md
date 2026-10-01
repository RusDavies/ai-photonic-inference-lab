# Functional Chained Blocks

Date: 2026-06-21

## Purpose

Test whether the earlier energy win from chaining optical blocks survives a
first functional accuracy proxy. This pass repeats a trained material residual
MLP block multiple times, injects constrained fixed-fabrication error into each
MLP-up projection, and evaluates end-to-end Fashion-MNIST accuracy.

This is not yet a properly trained stacked architecture. It deliberately asks a
simpler question first: can a one-block-trained material MLP tolerate being
reused as a 2x or 4x optical chain?

## Local Artifacts

- `artifacts/functional-chain-2026-06-21/functional_chain/functional_chain_rows.csv`
- `artifacts/functional-chain-2026-06-21/functional_chain/functional_chain_summary.csv`
- `artifacts/functional-chain-2026-06-21/functional_chain/functional_chain_summary.md`
- `artifacts/functional-chain-2026-06-21/functional_chain/functional_chain_manifest.json`
- `artifacts/chained-material-training-2026-06-21/chained_material_training/chained_material_training_rows.csv`
- `artifacts/chained-material-training-2026-06-21/chained_material_training/chained_material_training_summary.csv`
- `artifacts/chained-material-training-2026-06-21/chained_material_training/chained_material_training_summary.md`
- `artifacts/chained-material-training-2026-06-21/chained_material_training/chained_material_training_manifest.json`
- `artifacts/unshared-chained-material-training-2026-06-22/unshared_chained_material_training/unshared_chained_material_training_rows.csv`
- `artifacts/unshared-chained-material-training-2026-06-22/unshared_chained_material_training/unshared_chained_material_training_summary.csv`
- `artifacts/unshared-chained-material-training-2026-06-22/unshared_chained_material_training/unshared_chained_material_training_summary.md`
- `artifacts/unshared-chained-material-training-2026-06-22/unshared_chained_material_training/unshared_chained_material_training_manifest.json`

The artifact directory is intentionally uncommitted.

## Run Context

- Dataset: full Fashion-MNIST split.
- Seeds: 7, 11, and 13.
- Epochs: 5.
- Chain lengths: 1, 2, and 4 repeated material blocks.
- Material activations: reverse saturable absorption at strengths 0.1 and 0.2.
- Optical error: `signed_tiled_severe` constrained fixed-fabrication model.
- Calibration samples: 512.
- Rows: 18.
- Aggregate rows: 6.

The model trains the one-block material MLP, then evaluates:

- ideal repeated chain;
- raw constrained MLP-up projections at every chained block;
- per-block affine-calibrated constrained projections.

## Results

Mean test accuracy:

- Reverse saturable strength 0.1:
  - 1 block ideal: 87.00%.
  - 1 block raw constrained: 86.35%.
  - 1 block calibrated constrained: 86.72%.
  - 2 blocks ideal: 66.91%.
  - 2 blocks raw constrained: 73.03%.
  - 2 blocks calibrated constrained: 63.11%.
  - 4 blocks ideal: 15.17%.
  - 4 blocks raw constrained: 21.00%.
  - 4 blocks calibrated constrained: 15.49%.
- Reverse saturable strength 0.2:
  - 1 block ideal: 87.09%.
  - 1 block raw constrained: 86.46%.
  - 1 block calibrated constrained: 86.77%.
  - 2 blocks ideal: 72.10%.
  - 2 blocks raw constrained: 80.28%.
  - 2 blocks calibrated constrained: 70.33%.
  - 4 blocks ideal: 25.91%.
  - 4 blocks raw constrained: 36.61%.
  - 4 blocks calibrated constrained: 23.42%.

## Interpretation

The functional result is harsher than the energy model. A material MLP block
trained for one pass does not tolerate naive repetition. Chain length 2 already
costs roughly 15 to 20 accuracy points in the ideal model, and chain length 4 is
catastrophic.

The raw constrained chain sometimes beats the ideal repeated chain. That is not
a reason to celebrate the fabrication errors. It is a symptom that the repeated
block is over-driving the residual stream, and the constrained/noisy projection
is accidentally damping it. Calibration then pushes the chain back toward the
bad ideal dynamics, so calibrated accuracy can be worse than raw constrained
accuracy for deeper chains.

## Decision

Chaining is still attractive for energy and latency, but it cannot be done by
reusing a single-block-trained MLP as-is. Functional viability now requires
training the chained architecture end-to-end with the material activation and
optical error model in the loop.

## Next Work

Train chained material blocks end-to-end instead of repeating a single-block
model, then rerun constrained optical transfer.

## End-to-End Chained Training

The follow-up trained the repeated material block for the chained objective
instead of training one block and reusing it blindly. The chain still uses
shared MLP-up and MLP-down weights across repeated blocks, but gradients flow
through the whole repeated residual path.

Run context:

- Dataset: full Fashion-MNIST split.
- Seeds: 7, 11, and 13.
- Epochs: 5.
- Chain lengths: 2 and 4.
- Material activations: reverse saturable absorption at strengths 0.1 and 0.2.
- Optical error: `signed_tiled_severe` constrained fixed-fabrication model.
- Calibration samples: 512.
- Rows: 12.
- Aggregate rows: 4.

Mean test accuracy:

- Reverse saturable strength 0.1:
  - 2-block trained chain: 86.03%.
  - 2-block raw constrained: 85.10%.
  - 2-block calibrated constrained: 85.67%.
  - 4-block trained chain: 84.69%.
  - 4-block raw constrained: 82.54%.
  - 4-block calibrated constrained: 83.89%.
- Reverse saturable strength 0.2:
  - 2-block trained chain: 86.26%.
  - 2-block raw constrained: 85.40%.
  - 2-block calibrated constrained: 86.14%.
  - 4-block trained chain: 85.64%.
  - 4-block raw constrained: 83.50%.
  - 4-block calibrated constrained: 84.88%.

## Updated Interpretation

End-to-end chain training changes the answer completely. The naive repeated
one-block model collapsed because it was not trained for repeated residual
dynamics. Once the chain objective is trained directly, 2-block chains land
within roughly 0.6 to 0.8 points of the ReLU baseline, and 4-block chains land
within roughly 1.2 to 2.2 points.

Severe constrained fixed transfer is also tolerable after calibration:

- 2-block strength 0.2 loses only 0.12 points from the trained chain.
- 4-block strength 0.2 loses 0.76 points from the trained chain.
- 4-block strength 0.1 loses 0.80 points from the trained chain.

The strength 0.2 chain is now the better target. It keeps higher trained
accuracy and transfers at least as well after calibration. This rescues the
basic chained-block idea functionally, with the usual caveat that this is still
a small Fashion-MNIST proxy and shared-weight residual chain, not a deployed
model.

## Updated Decision

Chaining is back in the candidate set, but only with hardware-aware chain
training. Naively repeating a trained block is dead. A trained 2-block chain is
plausible; a trained 4-block chain is plausible enough to keep studying if the
energy/latency savings matter.

## Next Work

Train unshared per-block chained material weights and compare against the
shared-weight chained model.

## Unshared Per-Block Chain Training

The next pass gave each chained block its own MLP-up and MLP-down weights,
instead of sharing one trained material block across all chain positions.
The run focused on the best shared-chain material setting: reverse saturable
absorption at strength 0.2.

Run context:

- Dataset: full Fashion-MNIST split.
- Seeds: 7, 11, and 13.
- Epochs: 5.
- Chain lengths: 2 and 4.
- Material activation: reverse saturable absorption at strength 0.2.
- Optical error: `signed_tiled_severe` constrained fixed-fabrication model.
- Calibration samples: 512.
- Rows: 6.
- Aggregate rows: 2.

Mean test accuracy:

- 2-block unshared:
  - trained chain: 85.84%.
  - raw constrained: 85.16%.
  - calibrated constrained: 85.51%.
- 4-block unshared:
  - trained chain: 86.58%.
  - raw constrained: 85.26%.
  - calibrated constrained: 85.67%.

Shared-weight comparison for strength 0.2:

- 2-block shared:
  - trained chain: 86.26%.
  - calibrated constrained: 86.14%.
- 4-block shared:
  - trained chain: 85.64%.
  - calibrated constrained: 84.88%.

## Updated Interpretation

Unshared weights are not automatically better. The 2-block unshared chain
underperforms the shared-weight 2-block chain. The 4-block unshared chain,
however, is the best 4-block result so far: 86.58% trained and 85.67% after
severe constrained transfer plus calibration. That is only about 0.29 points
below the mean ReLU baseline, while getting the chain length needed for stronger
converter amortization.

The practical reading is now:

- 2-block chain: shared weights are better.
- 4-block chain: unshared weights are better.
- Best accuracy: shared 2-block calibrated at 86.14%.
- Best deeper-chain candidate: unshared 4-block calibrated at 85.67%.

The unshared 4-block result is probably the more interesting hardware target,
because it keeps useful accuracy while making ADC/DAC amortization more
credible.

## Updated Decision

Keep both candidates:

- shared 2-block chain for best accuracy;
- unshared 4-block chain for best accuracy/energy-latency trade.

## Next Work

Build an energy-accuracy Pareto report for shared vs unshared 2/4-block chains
with converter/readout assumptions.

## Energy/Accuracy Pareto Report

The next pass joined the shared and unshared trained-chain accuracy summaries
to the chained hardware budget rows. It uses the strength 0.2 reverse-saturable
material setting, because that is the best comparable setting shared by both
the shared-weight and unshared-weight runs.

Run context:

- Accuracy sources:
  - `artifacts/chained-material-training-2026-06-21/chained_material_training/chained_material_training_summary.csv`
  - `artifacts/unshared-chained-material-training-2026-06-22/unshared_chained_material_training/unshared_chained_material_training_summary.csv`
- Hardware budget source:
  - `artifacts/chained-block-budget-2026-06-21/chained_block_budget/chained_block_budget_rows.csv`
- Report output:
  - `artifacts/energy-accuracy-pareto-2026-06-22/energy_accuracy_pareto/energy_accuracy_pareto_rows.csv`
  - `artifacts/energy-accuracy-pareto-2026-06-22/energy_accuracy_pareto/energy_accuracy_pareto_summary.csv`
  - `artifacts/energy-accuracy-pareto-2026-06-22/energy_accuracy_pareto/energy_accuracy_pareto_summary.md`
  - `artifacts/energy-accuracy-pareto-2026-06-22/energy_accuracy_pareto/energy_accuracy_pareto_manifest.json`
- Rows: 20.
- Candidate summaries: 4.
- Pareto-efficient rows: 3.
- Readout assumption: one shared CMOS-like output boundary per chain, matching
  the current chained-block energy model.

Candidate summary:

- `shared_2_block`: 86.14% calibrated accuracy, best 0.290x digital energy,
  best 0.525x digital latency.
- `unshared_4_block`: 85.67% calibrated accuracy, best 0.257x digital energy,
  best 0.3125x digital latency.
- `unshared_2_block`: 85.51% calibrated accuracy, same hardware budget as the
  shared 2-block candidate, but worse accuracy.
- `shared_4_block`: 84.88% calibrated accuracy, same hardware budget as the
  unshared 4-block candidate, but worse accuracy.

Pareto result:

- `shared_2_block` remains the best-accuracy candidate. It is Pareto-efficient
  under the low-I/O hardware scenario because nothing else keeps its accuracy.
- `unshared_4_block` is the best hardware trade candidate. It is
  Pareto-efficient under low-I/O and tensor-core-style hardware assumptions,
  because it gives up 0.47 percentage points of calibrated accuracy versus the
  shared 2-block chain while improving best-case digital energy ratio from
  0.290x to 0.257x and best-case latency ratio from 0.525x to 0.3125x.
- `unshared_2_block` is dominated by `shared_2_block`.
- `shared_4_block` is dominated by `unshared_4_block`.

The hardware scenario still matters. The low-I/O case is the cleanest win:
`unshared_4_block` gives 85.67% calibrated accuracy at 0.281x digital energy
and 0.3125x digital latency. Tensor-core-style hardware gives the best energy
ratio, 0.257x, but still loses slightly on latency at 1.0375x digital. The
nonvolatile estimate remains plausible. Volatile O/E/O and high-precision
ADC-bound assumptions are still disqualifying on energy; the Pareto report does
not launder bad converter assumptions into a good architecture. Shame, really.

Updated decision:

- Keep `shared_2_block` as the accuracy reference.
- Promote `unshared_4_block` as the main hardware target for the next research
  step.
- Drop `unshared_2_block` and `shared_4_block` from the active candidate set
  unless a future training run changes their accuracy.

## Next Work

Stress-test the `unshared_4_block` candidate with stronger drift/noise and a
readout model that explicitly varies output-boundary ADC/TIA assumptions.

## Unshared 4-Block Stress Report

The follow-up stress pass kept the `unshared_4_block` candidate as the only
active hardware target and varied two things explicitly:

- calibrated drift/error drop multiplier: nominal, 2x, and 4x;
- output-boundary readout: CMOS column-parallel reference, 32-lane CCD bucket,
  8-lane CCD bucket, and one-lane serial stress case.

This is a proxy stress report, not a direct retraining run under new physical
noise parameters. It starts from the trained-chain accuracy and calibrated
transfer drop already measured for the unshared 4-block run, then adds explicit
read-noise penalties from the readout model and recomputes output-boundary
energy/latency with the chosen ADC/TIA/readout assumptions.

Run context:

- Accuracy source:
  - `artifacts/unshared-chained-material-training-2026-06-22/unshared_chained_material_training/unshared_chained_material_training_summary.csv`
- Hardware budget source:
  - `artifacts/chained-block-budget-2026-06-21/chained_block_budget/chained_block_budget_rows.csv`
- Report output:
  - `artifacts/unshared-4block-stress-2026-06-22/unshared_4block_stress/unshared_4block_stress_rows.csv`
  - `artifacts/unshared-4block-stress-2026-06-22/unshared_4block_stress/unshared_4block_stress_summary.csv`
  - `artifacts/unshared-4block-stress-2026-06-22/unshared_4block_stress/unshared_4block_stress_summary.md`
  - `artifacts/unshared-4block-stress-2026-06-22/unshared_4block_stress/unshared_4block_stress_manifest.json`
- Rows: 120.
- Summary rows: 24.
- Rows above 85% stressed accuracy: 15.

Key results:

- CMOS column-parallel boundary:
  - nominal/noisy-readout stress: 85.67% accuracy.
  - 2x drift stress: 84.76%.
  - 4x drift stress: 82.95%.
  - best energy ratio: 0.257x digital.
  - best latency ratio: 0.3375x digital.
- 32-lane CCD bucket:
  - nominal stress: 85.21%.
  - 2x readout noise stress: 84.75%.
  - 2x drift stress: 84.30%.
  - best energy ratio: 0.246x digital.
  - best latency ratio: 8.61x digital.
- 8-lane CCD bucket:
  - nominal stress: 82.71%.
  - best latency ratio: 58.04x digital.
- One-lane serial stress:
  - nominal stress: 36.33%.
  - best latency ratio: 871.21x digital.

Interpretation:

The candidate survives only a narrow stress window if the 85% accuracy line
matters. CMOS-style output readout remains the only credible fast-inference
boundary in this model. The 32-lane CCD bucket can keep nominal accuracy above
85%, but its latency is not close to acceptable for the cheap-fast path. The
8-lane and one-lane bucket cases are useful mainly as warnings with numbers
attached, which is at least better than warnings delivered by interpretive
dance.

Drift is now the primary risk. Doubling the calibrated drift/error drop pushes
the unshared 4-block candidate below 85% even before adding CCD read-noise
penalties. That means future work should stop treating the calibrated transfer
drop as a small bookkeeping term and start pinning it to a physical device,
thermal, and recalibration model.

Updated decision:

- Keep `unshared_4_block` as the hardware target only under CMOS-like or very
  fast near-column readout.
- Treat CCD-style bucket readout as non-primary for fast inference.
- Promote drift/noise physical modeling ahead of deeper-chain experiments.

## Next Work

Replace the proxy stress multipliers with a direct physical drift/noise model
for the unshared 4-block chain, then rerun accuracy evaluation against that
model.

## Direct Physical Drift/Noise Evaluation

The next pass replaced the proxy stress multipliers with a direct evaluation
loop. For each seed, it retrained the unshared 4-block reverse-saturable chain,
applied the severe constrained fixed transfer to each block's MLP-up weights,
then evaluated the chain across simulated time steps with:

- thermal drift applied directly to each constrained MLP-up weight matrix;
- detector noise applied inside each optical/material block;
- output-boundary readout noise applied after the residual chain and before the
  classifier head;
- scheduled affine recalibration of each block's MLP-up projection.

Run context:

- Dataset: full Fashion-MNIST split.
- Seeds: 7, 11, and 13.
- Epochs: 5.
- Chain length: 4.
- Material activation: reverse saturable absorption at strength 0.2.
- Optical transfer: `signed_tiled_severe`.
- Calibration samples: 512.
- Physical scenarios: 5.
- Time steps per physical scenario: 8.
- Rows: 120.
- Summary rows: 5.
- Artifacts:
  - `artifacts/unshared-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_rows.csv`
  - `artifacts/unshared-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_summary.csv`
  - `artifacts/unshared-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_summary.md`
  - `artifacts/unshared-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_manifest.json`

Mean calibrated accuracy under direct physical drift/noise:

- `noisy_near_column_boundary`: 85.81%; minimum 85.30%; final step mean 85.89%.
- `stable_cmos_boundary`: 85.78%; minimum 85.26%; final step mean 85.78%.
- `warm_cmos_boundary`: 85.74%; minimum 85.09%; final step mean 85.69%.
- `hot_cmos_sparse_recalibration`: 85.73%; minimum 84.90%; final step mean 85.74%.
- `hot_noisy_boundary`: 85.67%; minimum 84.97%; final step mean 85.63%.

Interpretation:

The direct physical model is less harsh than the proxy multiplier report. All
five scenario means remain above 85%, and only the two hot cases produce any
below-85 rows: one time-step row each out of 24. The proxy report was useful as
a warning siren, but it overestimated how fast the candidate collapses once
drift is applied directly and recalibration is allowed to do its actual job.
Imagine that: measuring the thing beats frowning at a multiplier.

The core risk remains drift, but the candidate is not dead under the first
physical-style model. The most useful current target is a CMOS-like or
near-column boundary with drift around 0.001 to 0.002 and detector noise at or
below 0.02 in this normalized model. Hot cases at drift 0.005 remain marginal:
they average above 85%, but have below-threshold time steps and higher
projection MSE.

Updated decision:

- Keep `unshared_4_block` as the main hardware target.
- Prefer CMOS-like or near-column output boundaries.
- Treat drift 0.005 plus detector noise 0.03 as the current warning edge, not
  an immediate kill condition.
- Do not spend more time on CCD-style low-lane readout for fast inference until
  there is a device reason to believe the latency and read-noise model is wrong.

## Next Work

Add time-series plots for the physical drift/noise evaluation so accuracy,
projection MSE, recalibration points, and hot-case dips are visible instead of
buried in CSV rows.

## Physical Drift/Noise Time-Series Plots

The physical drift/noise evaluation now emits two time-series plots alongside
the CSV and Markdown summaries:

- Accuracy over simulated time:
  - `artifacts/unshared-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_accuracy.png`
- Projection MSE over simulated time:
  - `artifacts/unshared-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_projection_mse.png`

The accuracy plot shows mean calibrated accuracy per physical scenario, seed
min/max bands, recalibration markers, and the 85% threshold. The projection-MSE
plot shows the same time axis and seed bands for calibrated MLP-up projection
error.

What the plots make obvious:

- All mean accuracy lines stay above 85%.
- The below-threshold events are seed-level dips in the hot scenarios, not a
  broad collapse of the average curve.
- `hot_cmos_sparse_recalibration` and `hot_noisy_boundary` separate cleanly in
  projection MSE after step 4.
- Recalibration at step 4 helps, but it does not fully erase the higher-drift
  MSE slope in the hot cases.
- Stable/warm/noisy-near-column scenarios stay visually boring, which is a
  compliment in reliability work.

Updated decision:

- Keep `unshared_4_block` active.
- Use projection-MSE growth, not just accuracy, as the early warning signal for
  the next physical-noise pass.
- Keep the 85% line as a practical threshold until a better success criterion is
  chosen.

## Next Work

Use the physical drift/noise curves to define explicit device tolerance targets
for drift rate, detector noise, output readout noise, and recalibration cadence.

## Device Tolerance Targets

The physical drift/noise summary was converted into explicit near-term device
tolerance targets. The report uses the direct evaluation rows, not the earlier
proxy multipliers.

Artifacts:

- `artifacts/device-tolerance-targets-2026-06-22/device_tolerance_targets/device_tolerance_targets.csv`
- `artifacts/device-tolerance-targets-2026-06-22/device_tolerance_targets/device_tolerance_targets.md`
- `artifacts/device-tolerance-targets-2026-06-22/device_tolerance_targets/device_tolerance_targets_manifest.json`

Recommended operating envelope:

- Thermal drift rate: `<= 0.002` per simulated step in the current sqrt-time
  drift model.
- Detector noise: `<= 0.02` of per-block projection scale.
- Output readout noise: `<= 0.005` of final residual scale.
- Recalibration cadence: every `<= 4` simulated steps.
- Practical accuracy floor: no seed/time-step rows below `85%`.
- Projection-MSE early warning threshold: about `0.191`.

Evidence:

- Passing scenarios:
  - `stable_cmos_boundary`
  - `warm_cmos_boundary`
  - `noisy_near_column_boundary`
- Warning-edge scenarios:
  - `hot_cmos_sparse_recalibration`
  - `hot_noisy_boundary`

The warning edge is now explicit: drift `0.005` with detector noise `0.03` is
marginal. It still has mean accuracy above 85%, but it produces at least one
seed/time-step dip below the floor. Output readout noise `0.01` is tolerable in
the hot/noisy mean, but only in the same marginal sense. In other words, this is
not yet a failure, but it is where the machine starts coughing politely.

Updated decision:

- Treat `drift <= 0.002`, detector noise `<= 0.02`, output readout noise
  `<= 0.005`, recalibration interval `<= 4` as the target envelope for the next
  device-facing pass.
- Treat projection MSE around `0.191` as the early warning line.
- Treat drift `0.005` plus detector noise `0.03` as the current warning edge.

## Next Work

Compare these normalized tolerance targets with plausible physical device
numbers from photonic/nonlinear optical hardware literature, then decide whether
the target envelope is remotely realistic or merely a spreadsheet in costume.

## Physical Device Reality Check

The normalized tolerance envelope was compared against photonic hardware
literature in `research/2026-06-22-device-tolerance-reality-check.md`.

Verdict: the envelope is not obviously fantasy, but it is plausible only with
active stabilization, periodic calibration, and hardware-aware/noise-aware
training. A passive open-loop resonant implementation is not credible.

The biggest missing piece is a calibration bridge from normalized drift/noise
terms to physical units: resonance shift, phase error, temperature excursion,
photon count/SNR, detector/TIA noise, output ADC error, and wall-clock
recalibration interval.

## Next Work

Build the calibration bridge that maps normalized drift/noise targets to
physical device quantities.

## Calibration Bridge to Physical Units

The calibration bridge is now a repeatable CLI report:
`--run-calibration-bridge-report`.

Using the default bridge assumptions, the recommended drift target maps to
about 4.80 pm/step or 0.070 K/step, detector noise maps to a 2500-photon
shot-noise floor, output readout maps to a 40000-photon floor and about
7.35 ENOB equivalent, and the 4-step recalibration cadence maps to a 0.282 K
thermal window.

This is enough to make the next question concrete: replace placeholder anchors
with architecture-specific values, then sweep the bridge sensitivity.

## Next Work

Replace the placeholder calibration-bridge anchors with architecture-specific
device assumptions and sensitivity sweeps.

## Architecture Calibration Sensitivity

The calibration bridge now emits a sensitivity table for MRR-style assumptions.
The tight ring-bank case has only a 0.141 K recalibration window, and the
source-log anchor has a 0.282 K window. Both require active control.

The broader/stabilized and athermal-compensated cases are more plausible:
roughly 1.92 K and 2.82 K recalibration windows, respectively. The readout
constraint remains unchanged: at least 40000 photons/sample and about 7.35 ENOB
equivalent at the output boundary, before ADC/TIA overhead.

## Next Work

Choose the candidate physical architecture profile and feed its constraints
back into the simulation/error model.

## Candidate Architecture Physical Run

The selected working profile is now `athermal_compensated_mrr`.

The physical drift/noise evaluator accepts `--physical-architecture-profile`
and records profile-derived constraints in its rows: resonance shift per step,
thermal excursion per step, photon-count floors, and output ENOB equivalent.

Full run:

- Seeds: 7, 11, 13.
- Chain length: 4.
- Candidate profile: `athermal_compensated_mrr`.
- Rows: 96.
- Summary rows: 4.

Results:

- Nominal: mean accuracy 85.78%, minimum 85.12%, no below-85 rows.
- Sparse refresh: mean accuracy 85.78%, minimum 85.24%, no below-85 rows.
- Thermal edge: mean accuracy 85.76%, minimum 85.19%, no below-85 rows, but
  projection MSE rises to 0.193.
- Readout edge: mean accuracy 85.79%, minimum 85.15%, no below-85 rows.

## Next Work

Connect the selected athermal-compensated MRR profile to the energy/latency
model, especially detector/readout electronics and control overhead.

## Athermal MRR Energy/Latency Overlay

The selected profile is connected to the existing energy/accuracy Pareto rows
through `--run-architecture-energy-overlay-report`.

Results:

- 20 overlay rows.
- 8 rows pass accuracy, energy, and latency gates together.
- Each physical scenario has two passing hardware cases.
- Best energy ratio stays around 0.258 to 0.260 vs digital.
- Best latency ratio is 0.3375 vs digital.
- Median added profile overhead is about 97 to 211 pJ depending on thermal edge
  severity, with thermal control dominating the overhead.

Interpretation: the athermal MRR candidate still has a plausible energy win
after first detector/readout/control overheads, but only under optimistic
low-IO hardware assumptions. The next refinement should stress thermal control
and resonance tracking overheads.

## Next Work

Stress the athermal MRR energy overlay across thermal-control and resonance-
tracking overhead assumptions.

## Thermal-Control / Resonance-Tracking Stress

The athermal MRR energy overlay now has a control-overhead stress report:
`--run-architecture-energy-stress-report`.

Sweep result:

- 320 rows.
- 16 control/tracking combinations.
- Every combination keeps 8 rows passing accuracy, energy, and latency gates.
- Best energy ratio ranges from 0.257x to 0.274x digital.
- Median energy ratio ranges from 0.709x to 0.725x digital.
- Median added profile overhead ranges from about 15 pJ to 1120 pJ.

Interpretation: control overhead matters, but it is not the thing killing the
candidate inside this sweep. The surviving cases are still the low-IO hardware
assumptions. The dead cases remain ADC/readout/latency dominated, because
nature has a fine sense of comedic timing and apparently enjoys converters.

## Next Work

Stress converter placement and ADC/DAC/readout assumptions for the selected
athermal MRR profile.
