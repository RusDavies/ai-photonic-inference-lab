# Device Tolerance Reality Check

Date: 2026-06-22

## Purpose

Compare the normalized tolerance targets from the unshared 4-block optical
chain against plausible physical photonic hardware behavior. This is a
sanity-check pass, not a calibrated device model. The current simulation uses
dimensionless drift and noise terms, so the right question is whether the
targets sound compatible with known device constraints and what bridge is
needed before claiming a physical spec.

## Simulation Targets Being Checked

From the device tolerance target report:

- Drift rate: `<= 0.002` per simulated step in the current sqrt-time drift
  model.
- Detector noise: `<= 0.02` of per-block projection scale.
- Output readout noise: `<= 0.005` of final residual scale.
- Recalibration cadence: every `<= 4` simulated steps.
- Accuracy floor: no seed/time-step rows below `85%`.
- Projection-MSE warning line: about `0.191`.
- Warning edge: drift `0.005` plus detector noise `0.03`.

## Physical Literature Anchors

Thermal drift and crosstalk are the least forgiving part of the target. Silicon
microring systems are highly temperature-sensitive. Recent physical-neural-
network work reports that a 12 pm resonance shift, corresponding to about
0.2 C in their MRR setup, dropped MNIST accuracy from an ideal 99.0% to 67.0%.
Separate programmable-PIC crosstalk work treats crosstalk as a measurable
resonance-wavelength shift and reports sub-pm to low-pm modeling and
compensation errors. Older silicon microring thermal work and sensor literature
put uncompensated SOI ring temperature sensitivity in the rough tens-of-pm/K
range, commonly around 60 to 80 pm/K, while athermalized or compensated designs
can reduce that by more than an order of magnitude.

The implication is blunt: a passive, open-loop resonant implementation is not
compatible with this target envelope. The simulated drift target is only
plausible if it maps to a tightly stabilized operating point: thermal control,
laser or resonance tracking, crosstalk-aware programming, periodic refresh, or
an architecture less sensitive than dense ring weight banks. This lines up with
recent MRR neuron work that relies on TEC control, per-channel laser tuning, and
periodic re-sweeping of the current-weight map to correct slow drift.

Detector and readout noise look more plausible, but not free. Quantum-limited
optical-neural-network work shows that useful inference is possible even near
single-photon activation regimes when the model is trained for the stochastic
physics. It also makes the tradeoff unavoidable: low optical energy pushes the
system toward low SNR, and shot-noise-limited SNR scales like the square root of
photon count. Our detector-noise target of `<= 0.02` sounds plausible for a
moderate-power analog receiver, but it is probably not plausible at the most
aggressive few-photon energy point unless the model is explicitly trained as a
stochastic physical network.

The output readout target, `<= 0.005`, is the stricter readout requirement. It
supports the earlier preference for CMOS-like or near-column readout. It does
not support a slow, low-lane CCD-style output boundary for fast inference unless
a real device provides much better noise/latency behavior than the placeholder
model.

Hardware-aware training is not optional. Control-free/hardware-aware PNN work
reports large accuracy recovery by training into noise-robust regions, and
sharpness-aware physical-neural-network work directly frames thermal drift,
misalignment, fabrication variation, and post-deployment perturbations as core
deployment problems. That matches the local result: the chain only became
interesting after the hardware-style errors and calibration were put in the
loop.

## Target-by-Target Reality Check

- `drift <= 0.002` per simulated step:
  Plausible only after stabilization and recalibration. It is not yet a
  physical spec because one simulated step has not been mapped to resonance
  shift, phase error, temperature change, or wall-clock time.

- detector noise `<= 0.02`:
  Plausible for a receiver operated with enough photons and sane electronics.
  It may collide with the most aggressive energy target. The model needs a
  photon-count/SNR bridge before this can be called realistic.

- output readout noise `<= 0.005`:
  Demanding but plausible for the preferred CMOS-like boundary. It reinforces
  the decision to ignore low-lane CCD-style readout for now.

- recalibration interval `<= 4` simulated steps:
  Plausible if a step represents slow thermal or environmental drift. Useless
  as a physical requirement until the step is mapped to wall-clock time,
  inference count, or temperature excursion.

- projection-MSE warning line around `0.191`:
  Useful as an internal health metric, but not device-facing. It needs to be
  translated into optical transfer error, resonance/phase error, or SNR before
  it can drive hardware requirements.

## Decision

The target envelope is not obviously fantasy, but it is only plausible as an
actively stabilized, periodically calibrated, hardware-aware photonic system.
It is not plausible as a passive optical slab that holds weights accurately
while the room temperature sneezes at it.

The largest unresolved risk is the mapping from normalized drift to physical
thermal/resonance drift. The second-largest risk is whether the detector and
output readout noise targets can be met at the same energy point that makes the
photonic path worthwhile.

## Next Work

Build a calibration bridge from normalized simulation units to physical device
quantities:

- drift -> resonance shift, phase error, heater/current error, and temperature
  excursion;
- detector noise -> photon count, optical power, shot-noise SNR, detector NEP,
  and TIA noise;
- output readout noise -> ADC/TIA/residual-scale error at the block boundary;
- simulated step -> wall-clock time, inference count, thermal time constant, or
  recalibration interval.

Only after that bridge exists can the `0.002 / 0.02 / 0.005 / 4-step` envelope
be judged as a device spec instead of a normalized sim requirement wearing a
lab coat.

## Calibration Bridge Report

The bridge is now implemented as a repeatable report:

```bash
python -m optical_spike --run-calibration-bridge-report --output-dir artifacts/calibration-bridge-2026-06-22
```

Local artifacts:

- `artifacts/calibration-bridge-2026-06-22/calibration_bridge/calibration_bridge_rows.csv`
- `artifacts/calibration-bridge-2026-06-22/calibration_bridge/calibration_bridge.md`
- `artifacts/calibration-bridge-2026-06-22/calibration_bridge/calibration_bridge_sensitivity.csv`
- `artifacts/calibration-bridge-2026-06-22/calibration_bridge/calibration_bridge_sensitivity.md`
- `artifacts/calibration-bridge-2026-06-22/calibration_bridge/calibration_bridge_manifest.json`

Default assumptions:

- Warning-edge normalized drift `0.005` maps to a 12 pm resonance shift.
- Silicon thermal sensitivity is anchored at 68.2 pm/K.
- Resonance FWHM placeholder is 100 pm.
- Shot-noise floors use 1550 nm photons and exclude detector/TIA/ADC overhead.
- Recalibration timing examples use placeholder temperature ramps of 1, 0.1,
  and 0.01 K/min.

First bridge output:

- Recommended drift `0.002` maps to about 4.80 pm per simulated step, or
  0.070 K per step.
- Warning-edge drift `0.005` maps to 12.00 pm per step, or 0.176 K per step.
- Recommended drift is about 0.302 rad/step under the placeholder 100 pm FWHM
  phase-error anchor.
- Detector noise `0.02` implies a shot-noise floor of at least 2500 photons per
  sample, about 34.0 dB amplitude SNR and 0.320 fJ at 1550 nm before electronics.
- Output readout noise `0.005` implies at least 40000 photons per sample, about
  46.0 dB amplitude SNR, 7.35 ENOB equivalent, and 5.126 fJ at 1550 nm before
  ADC/TIA overhead.
- Recalibration every 4 simulated steps corresponds to a 0.282 K thermal window
  under the default drift bridge. That is 16.9 seconds at 1 K/min, 168.9 seconds
  at 0.1 K/min, or 1689.1 seconds at 0.01 K/min.

Updated decision:

- The bridge is useful enough to guide the next device-facing pass.
- It is still an assumption bridge, not a measured device spec.
- The next step is to replace the placeholder anchors with architecture-specific
  measured or sourced values and run a sensitivity sweep.

## Architecture Sensitivity Sweep

The bridge now includes architecture-specific MRR-style sensitivity rows:

- `tight_mrr_weight_bank`: 6 pm warning shift, 68.2 pm/K sensitivity, 80 pm
  FWHM. Recommended drift maps to 2.4 pm/step, 0.035 K/step, and a 0.141 K
  recalibration window. Status: active control needed.
- `source_log_anchor_mrr`: 12 pm warning shift, 68.2 pm/K sensitivity, 100 pm
  FWHM. Recommended drift maps to 4.8 pm/step, 0.070 K/step, and a 0.282 K
  recalibration window. Status: active control needed.
- `broader_stabilized_mrr`: 24 pm warning shift, 20 pm/K effective
  sensitivity, 150 pm FWHM. Recommended drift maps to 9.6 pm/step, 0.480
  K/step, and a 1.92 K recalibration window. Status: plausible with
  stabilization.
- `athermal_compensated_mrr`: 12 pm warning shift, 6.82 pm/K effective
  sensitivity, 100 pm FWHM. Recommended drift maps to 4.8 pm/step, 0.704
  K/step, and a 2.82 K recalibration window. Status: plausible with
  stabilization.

The photon/SNR side does not change by architecture in this pass: detector
noise still implies at least 2500 photons/sample, and output readout still
implies at least 40000 photons/sample plus about 7.35 ENOB equivalent. The
sweep adds rough electronics multipliers so the final output readout floor
becomes roughly 51 fJ/sample for the 10x cases and 103 fJ/sample for the tight
20x case before any broader system overhead.

Updated decision:

- Tight ring banks and the current source-log anchor are thermally unforgiving;
  assume active resonance control.
- Broader/stabilized or athermal-compensated rings are the only MRR-style
  profiles that make the recalibration window look comfortable.
- The next work should pick the candidate architecture profile and put those
  physical constraints back into the simulation/error model.

## Selected Architecture Profile in Simulation

Candidate profile: `athermal_compensated_mrr`.

Rationale:

- It preserves the MRR-weight-bank path instead of switching architectures.
- It gives the widest thermal margin in the sensitivity sweep: about 0.704 K
  per recommended drift step and a 2.82 K window over four simulated steps.
- It still keeps the same readout burden as the normalized target, so it does
  not hide the output-boundary problem.

The physical drift/noise evaluator now accepts `--physical-architecture-profile`
and emits architecture metadata in the row CSV: profile name, resonance shift
per step, thermal excursion per step, photon-count floors, and output ENOB
equivalent.

Full candidate run:

```bash
python -m optical_spike --run-unshared-physical-drift-noise --physical-architecture-profile athermal_compensated_mrr --seeds 7,11,13 --epochs 5 --calibration-samples 512 --output-dir artifacts/athermal-physical-drift-noise-2026-06-22
```

Local artifacts:

- `artifacts/athermal-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_rows.csv`
- `artifacts/athermal-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_summary.csv`
- `artifacts/athermal-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_summary.md`
- `artifacts/athermal-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_accuracy.png`
- `artifacts/athermal-physical-drift-noise-2026-06-22/unshared_physical_drift_noise/unshared_physical_drift_noise_projection_mse.png`

Summary across seeds 7, 11, and 13:

- `athermal_compensated_mrr_nominal`: mean accuracy 85.78%, minimum 85.12%,
  projection MSE 0.176, no below-85 rows.
- `athermal_compensated_mrr_sparse_refresh`: mean accuracy 85.78%, minimum
  85.24%, projection MSE 0.176, no below-85 rows.
- `athermal_compensated_mrr_thermal_edge`: mean accuracy 85.76%, minimum
  85.19%, projection MSE 0.193, no below-85 rows.
- `athermal_compensated_mrr_readout_edge`: mean accuracy 85.79%, minimum
  85.15%, projection MSE 0.176, no below-85 rows.

Updated decision:

- Use `athermal_compensated_mrr` as the working physical architecture profile.
- Sparse refresh is not showing an accuracy penalty yet in this 8-step run, but
  it should not be trusted until longer drift horizons are tested.
- The thermal-edge case raises projection MSE to about 0.193, which confirms the
  earlier MSE warning line even though accuracy stays above the floor.
- The next pass should connect this profile to the energy/latency model,
  especially readout/driver/control overhead.

## Athermal Profile Energy/Latency Overlay

The selected profile is now connected back to the existing energy/accuracy
Pareto rows:

```bash
python -m optical_spike --run-architecture-energy-overlay-report --output-dir artifacts/athermal-energy-overlay-2026-06-22
```

Local artifacts:

- `artifacts/athermal-energy-overlay-2026-06-22/architecture_energy_overlay/architecture_energy_overlay_rows.csv`
- `artifacts/athermal-energy-overlay-2026-06-22/architecture_energy_overlay/architecture_energy_overlay_summary.csv`
- `artifacts/athermal-energy-overlay-2026-06-22/architecture_energy_overlay/architecture_energy_overlay_summary.md`
- `artifacts/athermal-energy-overlay-2026-06-22/architecture_energy_overlay/architecture_energy_overlay_manifest.json`

Overlay assumptions:

- Candidate: `unshared_4_block`.
- Architecture profile: `athermal_compensated_mrr`.
- Detector photon floor gets a 10x electronics multiplier and is applied across
  four 256-wide block outputs.
- Output readout photon floor gets a 10x electronics multiplier and is applied
  to the final 256-wide boundary.
- Thermal control overhead is modeled as 25 pJ per K per block.
- Resonance tracking overhead is modeled as 5 pJ per block.
- Critical-path profile latency overhead is modeled as 0.2 ns.

Summary:

- Rows: 20.
- Rows passing accuracy, energy, and latency gates together: 8.
- Each of the four physical scenarios has two hardware cases passing all gates.
- Best energy ratio remains around 0.258 to 0.260 vs digital.
- Best latency ratio is 0.3375 vs digital.
- Median profile overhead is about 97 pJ for readout edge, 107 pJ for nominal
  and sparse refresh, and 211 pJ for thermal edge.
- Thermal control dominates the profile overhead: about 70 pJ in nominal/sparse
  refresh/readout edge and 176 pJ at thermal edge.

Updated decision:

- The selected athermal MRR profile still has a plausible energy win after a
  first detector/readout/control overlay.
- The win is concentrated in the same optimistic low-IO hardware assumptions;
  ADC/latency-heavy cases remain bad.
- Thermal control overhead is now the control-side risk to refine.

## Thermal-Control / Resonance-Tracking Stress

The athermal profile energy overlay was swept across thermal-control and
resonance-tracking assumptions:

```bash
python -m optical_spike --run-architecture-energy-stress-report --output-dir artifacts/athermal-control-stress-2026-06-23
```

Local artifacts:

- `artifacts/athermal-control-stress-2026-06-23/architecture_energy_stress/architecture_energy_stress_rows.csv`
- `artifacts/athermal-control-stress-2026-06-23/architecture_energy_stress/architecture_energy_stress_summary.csv`
- `artifacts/athermal-control-stress-2026-06-23/architecture_energy_stress/architecture_energy_stress_summary.md`
- `artifacts/athermal-control-stress-2026-06-23/architecture_energy_stress/architecture_energy_stress_manifest.json`

Sweep:

- Thermal control: 0, 25, 100, and 250 pJ/K/block.
- Resonance tracking: 0, 5, 25, and 100 pJ/block.
- Rows: 320.
- Summary rows: 16.

Result:

- Every control/tracking combination still has 8 rows passing accuracy, energy,
  and latency together.
- Best energy ratio ranges from 0.257x to 0.274x digital across the sweep.
- Median energy ratio ranges from 0.709x to 0.725x digital.
- Median profile overhead ranges from about 15 pJ to 1120 pJ.
- The same 12 rows remain below digital energy and the same 8 rows remain below
  digital latency across the sweep.

Updated decision:

- Thermal-control and resonance-tracking overheads are not the immediate kill
  switch under this sweep.
- The viability bottleneck has moved back to converter/readout/IO assumptions:
  low-IO cases survive; ADC-heavy cases are still hopeless.
- Next work should stress converter placement and ADC/DAC/readout assumptions
  for the selected profile.

## Converter Placement / ADC-DAC-Readout Stress

The selected athermal profile was then swept across converter placement and
ADC/DAC/readout energy multipliers:

```bash
python -m optical_spike --run-architecture-converter-stress-report --output-dir artifacts/athermal-converter-stress-2026-06-23
```

Local artifacts:

- `artifacts/athermal-converter-stress-2026-06-23/architecture_converter_stress/architecture_converter_stress_rows.csv`
- `artifacts/athermal-converter-stress-2026-06-23/architecture_converter_stress/architecture_converter_stress_summary.csv`
- `artifacts/athermal-converter-stress-2026-06-23/architecture_converter_stress/architecture_converter_stress_summary.md`
- `artifacts/athermal-converter-stress-2026-06-23/architecture_converter_stress/architecture_converter_stress_manifest.json`

Sweep:

- Converter placement: shared input/output boundary, ADC/readout after each
  block, and full O/E/O between every block.
- ADC energy multipliers: 0.5x, 1x, 5x, and 25x.
- DAC energy multipliers: 0.5x, 1x, and 5x.
- Readout/TIA energy multipliers: 0.5x, 1x, 5x, and 25x.
- Rows: 2880.
- Summary rows: 144.

Result:

- Every summary combination has at least one row passing accuracy, energy, and
  latency together, but the placement changes how many rows survive.
- Shared input/output boundary keeps up to 8 passing rows and reaches a best
  energy ratio of 0.233x digital under the most optimistic converter settings.
- ADC/readout after each block still keeps up to 8 passing rows, but its best
  energy ratio degrades to about 0.288x digital.
- Full O/E/O between blocks tops out at 4 passing rows and a best energy ratio
  of about 0.289x digital; its best latency ratio is 0.900x digital, so there is
  very little latency margin left.
- Under the harshest 25x ADC and 25x readout cases, only 4 rows pass, and median
  energy ratios climb to roughly 4.9x digital even though the best low-IO row can
  still remain below digital energy.

Updated decision:

- The cheap-fast path still requires converter sharing. The shared-boundary
  assumption is not just a nice optimization; it is carrying much of the win.
- Per-block output observation is survivable only in the low-IO hardware rows.
- Full O/E/O between blocks is a warning case: it can barely remain below digital
  latency in the best rows and should not be treated as the target architecture.
- Next work should turn the surviving low-IO rows into concrete ADC/DAC/readout
  target specs instead of leaving them as multipliers.

## Converter Target Specs

The passing low-IO converter stress rows were converted into per-port target
specs:

```bash
python -m optical_spike --run-architecture-converter-target-report --output-dir artifacts/athermal-converter-targets-2026-06-23
```

Local artifacts:

- `artifacts/athermal-converter-targets-2026-06-23/architecture_converter_target/architecture_converter_target_rows.csv`
- `artifacts/athermal-converter-targets-2026-06-23/architecture_converter_target/architecture_converter_target_rows.md`
- `artifacts/athermal-converter-targets-2026-06-23/architecture_converter_target/architecture_converter_target_manifest.json`

Filter:

- Only rows passing accuracy, energy, and latency gates.
- Only `published_273fj_low_io` and `inp_nonvolatile_all_optical_estimate`.
- Energy and latency ratios must both remain below digital.

Derived envelope:

- Broadest passing per-port target: DAC `<= 2.5 pJ/input`, ADC
  `<= 25 pJ/output`, readout/TIA `<= 12.5 pJ/output`.
- Shared input/output boundary passes under that broadest envelope for both
  low-IO scenarios.
- `published_273fj_low_io` also passes under ADC/readout-after-each-block and
  full O/E/O placement at the broadest envelope, but latency margin is much
  thinner: full O/E/O has worst passing latency ratio `0.900x` digital.
- `inp_nonvolatile_all_optical_estimate` passes ADC/readout-after-each-block only
  with ADC `<= 5 pJ/output`; at `25 pJ/output`, its energy margin is gone.

Target rows:

- `published_273fj_low_io`, shared boundary: worst passing energy ratio `0.427x`,
  latency ratio `0.3375x`.
- `inp_nonvolatile_all_optical_estimate`, shared boundary: worst passing energy
  ratio `0.854x`, latency ratio `0.4875x`.
- `published_273fj_low_io`, ADC/readout after each block: worst passing energy
  ratio `0.867x`, latency ratio `0.7125x`.
- `inp_nonvolatile_all_optical_estimate`, ADC/readout after each block: worst
  passing energy ratio `0.981x`, latency ratio `0.8625x`.
- `published_273fj_low_io`, full O/E/O between blocks: worst passing energy
  ratio `0.874x`, latency ratio `0.900x`.

Updated decision:

- For the target architecture, keep the shared-boundary spec as the primary
  requirement: DAC `<= 2.5 pJ/input`, ADC `<= 25 pJ/output`, readout/TIA
  `<= 12.5 pJ/output`, with one input and one output boundary.
- Treat per-block ADC/readout as a fallback only if ADC energy is closer to
  `5 pJ/output` for the all-optical-style row.
- Full O/E/O should stay a stress case, not a design target, until a real
  converter survey proves the latency and energy numbers are not fantasy.
- Next work should compare these targets against primary-source device and
  circuit literature.

## Converter Target Literature Check

The derived converter/readout targets were checked against primary converter
and photonic-system sources:

- `published_273fj_low_io` and `inp_nonvolatile_all_optical_estimate` shared
  boundary target: DAC `<= 2.5 pJ/input`, ADC `<= 25 pJ/output`, readout/TIA
  `<= 12.5 pJ/output`.
- `inp_nonvolatile_all_optical_estimate` with ADC/readout after each block:
  same DAC/readout envelope, but ADC must be closer to `<= 5 pJ/output`.
- Full O/E/O target: only survives in the `published_273fj_low_io` row, with
  very thin latency margin.

Source anchors:

- Tait's silicon photonic power analysis says published 2-6 bit ADC energies
  were above `1 pJ` as of 2017, and warns that ADC energy can exceed
  optoelectronic detection energy by two orders of magnitude.
- Murmann's ADC survey is the right continuing baseline for ADC feasibility; it
  covers ISSCC/VLSI ADC results through 2026.
- A 2022 SAR ADC survey reports best conversion-step FoM improving from about
  `3 pJ/conversion-step` in 2000 to `0.4 fJ/conversion-step` in 2019. That is
  encouraging, but conversion-step FoM is not full sample energy.
- A concrete high-speed 8-bit ADC data point reports `2.0-3.3 pJ/conversion`
  at `24-72 GS/s`, which makes the `<= 5 pJ/output` target plausible but not
  casual.
- The InP all-optical neural-network system model uses `25 mW` DACs, `25 mW`
  ADCs, and `5 mW` receiver PDs at `10 GHz`, corresponding to roughly
  `2.5 pJ/sample` DAC/ADC and `0.5 pJ/sample` receiver power at that rate.
- A 2022 current-steering DAC energy-bound paper supports treating DAC energy
  as a speed/SNR/linearity-bound circuit target rather than one fixed constant.

Comparison:

- DAC `<= 2.5 pJ/input`: supported as an aggressive but literature-backed target
  for a shared-boundary photonic transceiver-style assumption. It is not safe to
  apply blindly to every high-resolution or high-linearity DAC case.
- ADC `<= 25 pJ/output`: loose enough that published high-speed ADC examples can
  beat it by nearly an order of magnitude. This supports the shared-boundary
  architecture.
- ADC `<= 5 pJ/output`: plausible for high-speed low-to-moderate precision, but
  it should be treated as a real design constraint. This is the relevant number
  for any per-block readout design.
- Readout/TIA `<= 12.5 pJ/output`: loose relative to the InP system-level
  receiver-power assumption, but still needs a more specific detector/TIA/source
  model before it can be considered proven.
- Full O/E/O: even if converter energy is feasible, latency is the problem. The
  model's best passing full-O/E/O case has only `0.900x` digital latency margin,
  so any extra serialization, buffering, or DSP overhead can kill it.

Updated decision:

- Shared-boundary conversion is now the backed target architecture, not just a
  modeling convenience.
- Per-block ADC/readout remains possible only if the ADC target is set around
  `<= 5 pJ/output` and the readout path avoids serialization penalties.
- Full O/E/O should be demoted to a negative control / stress case.
- The next model update should replace abstract converter multipliers with
  named literature-backed converter profiles.

## Named Converter Profile Report

The converter placement stress model now has a named profile report instead of
only abstract ADC/DAC/readout multipliers:

```bash
python -m optical_spike --run-architecture-converter-profile-report --output-dir artifacts/athermal-converter-profiles-2026-06-23
```

Local artifacts:

- `artifacts/athermal-converter-profiles-2026-06-23/architecture_converter_profile/architecture_converter_profile_rows.csv`
- `artifacts/athermal-converter-profiles-2026-06-23/architecture_converter_profile/architecture_converter_profile_summary.csv`
- `artifacts/athermal-converter-profiles-2026-06-23/architecture_converter_profile/architecture_converter_profile_summary.md`
- `artifacts/athermal-converter-profiles-2026-06-23/architecture_converter_profile/architecture_converter_profile_manifest.json`

Profiles:

- `inp_2022_10ghz_transceiver`: DAC `2.5 pJ/input`, ADC `2.5 pJ/output`,
  receiver/readout `0.5 pJ/output`, with DAC/ADC latency set to `0.1 ns` from
  the 10 GHz system assumption.
- `kull_2018_8b_ti_sar_adc`: DAC `2.5 pJ/input`, ADC `3.3 pJ/output`,
  receiver/readout `0.5 pJ/output`, using the reported high-speed 8-bit SAR
  ADC point as the ADC anchor.
- `per_block_adc_guardrail`: DAC `2.5 pJ/input`, ADC `5.0 pJ/output`,
  receiver/readout `0.5 pJ/output`, with baseline converter latencies. This is
  the practical per-block readout ceiling from the target/literature pass.
- `shared_boundary_target_ceiling`: DAC `2.5 pJ/input`, ADC `25 pJ/output`,
  readout/TIA `12.5 pJ/output`, representing the broad shared-boundary target
  envelope rather than one demonstrated circuit.

Summary:

- 240 profile rows and 12 placement/profile summaries were generated.
- `inp_2022_10ghz_transceiver` and `kull_2018_8b_ti_sar_adc` pass all gates for
  all three placements in the low-IO passing rows: shared boundary,
  ADC/readout-after-each-block, and full O/E/O.
- `per_block_adc_guardrail` passes shared boundary and ADC/readout-after-each
  block for all low-IO passing rows, but full O/E/O drops to 8 passing rows and
  carries only `0.900x` best latency ratio versus digital.
- `shared_boundary_target_ceiling` passes shared boundary for all low-IO passing
  rows, but per-block placements fall to 8 passing rows and median energy rises
  above digital (`1.29x`), confirming it is not a per-block design point.

Updated decision:

- Use `inp_2022_10ghz_transceiver` as the optimistic named converter profile
  for shared-boundary architecture comparisons.
- Use `kull_2018_8b_ti_sar_adc` as the conservative high-speed ADC profile when
  the model needs a concrete ADC rather than a system-level transceiver budget.
- Keep `per_block_adc_guardrail` as the fallback ceiling for per-block readout.
- Treat `shared_boundary_target_ceiling` as a requirement envelope only. Using
  it as a per-block implementation assumption is just spreadsheet cosplay.
- Next work should split these profiles into separate ADC/DAC/detector/TIA
  bandwidth, ENOB, and source-chain constraints instead of bundling them as one
  converter profile.

## Converter Component Constraint Split

The named converter profiles now split their component assumptions into DAC,
ADC, detector, and TIA constraints. The same CLI flag emits a wider row/summary
schema with component names, per-sample energy, bandwidth, ENOB where known,
source-chain labels, source text, and notes:

```bash
python -m optical_spike --run-architecture-converter-profile-report --output-dir artifacts/athermal-converter-components-2026-06-23
```

Local artifacts:

- `artifacts/athermal-converter-components-2026-06-23/architecture_converter_profile/architecture_converter_profile_rows.csv`
- `artifacts/athermal-converter-components-2026-06-23/architecture_converter_profile/architecture_converter_profile_summary.csv`
- `artifacts/athermal-converter-components-2026-06-23/architecture_converter_profile/architecture_converter_profile_summary.md`
- `artifacts/athermal-converter-components-2026-06-23/architecture_converter_profile/architecture_converter_profile_manifest.json`

Component columns now include:

- DAC: `dac_component`, `dac_pj_per_input`, `dac_bandwidth_gsps`,
  `dac_enob`, `dac_source_chain`.
- ADC: `adc_component`, `adc_pj_per_output`, `adc_bandwidth_gsps`,
  `adc_enob`, `adc_source_chain`.
- Detector: `detector_component`, `detector_pj_per_output`,
  `detector_bandwidth_gsps`, `detector_enob`, `detector_source_chain`.
- TIA: `tia_component`, `tia_pj_per_output`, `tia_bandwidth_gsps`,
  `tia_enob`, `tia_source_chain`.

The pass/fail results did not change from the named-profile report because the
same total per-port energies and converter latencies are preserved:

- 240 profile rows and 12 placement/profile summaries.
- InP 10 GHz transceiver and Kull 2018 ADC profile still pass all gates for all
  low-IO passing placements.
- `per_block_adc_guardrail` still exposes full O/E/O as latency-fragile:
  8 passing rows and `0.900x` best digital latency ratio.
- `shared_boundary_target_ceiling` still confirms that the `25 pJ/output` ADC
  and `12.5 pJ/output` readout allowance are shared-boundary requirements, not
  per-block implementation assumptions.

Updated decision:

- Keep the profile-level report for architecture comparison, but treat ADC,
  DAC, detector, and TIA rows as the audit surface.
- The ADC side is now reasonably anchored: `3.3 pJ`, `72 GS/s`, `8 bit` for
  the concrete Kull profile and `5 pJ/output`, `10 GS/s`, `7.35 ENOB` for the
  per-block guardrail.
- Detector/TIA energy is separated, but still piggybacks on the InP receiver
  power split; this is acceptable as a placeholder, not as a device proof.
- DAC ENOB/linearity remains the obvious hole because the system-level InP DAC
  budget did not specify the resolution/linearity needed by this transfer
  model. Next work should fill that from primary DAC/modulator sources.

## DAC ENOB And Linearity Constraints

The DAC side now carries explicit resolution and linearity constraints instead
of only an energy/rate placeholder. The component schema adds:

- `*_sfdr_db`
- `*_inl_lsb`
- `*_dnl_lsb`

For the DAC rows, the working constraint is now:

- effective precision: `>= 7.35 ENOB`
- dynamic linearity: `>= 50 dBc SFDR`
- static linearity anchor: about `<= 1.22 LSB` INL and `<= 1.21 LSB` DNL

The source chain is deliberately split:

- The InP photonic NN system remains the energy/rate anchor: `25 mW` DACs at
  `10 GHz`, equivalent to `2.5 pJ/sample`.
- Wang et al. provide a primary current-steering DAC linearity anchor: a
  12-bit `3 GS/s` DAC with `> 50 dBc` SFDR across Nyquist, DNL
  `-0.55/+1.21 LSB`, INL `-1.22/+0.64 LSB`, and `495 mW` total power.
- Patel et al. provide a primary photonic modulator route: a 2-bit segmented
  electro-optic MZM DAC for PAM-4 with `48 GHz` measured EO bandwidth on the
  longer segment.

Decision:

- Treat DAC `2.5 pJ/input`, `10 GS/s`, `>= 7.35 ENOB`, and `>= 50 dBc SFDR`
  as a required component target, not as a demonstrated part.
- The low-energy InP-style DAC is still plausible as a system budget, but it is
  not yet a full DAC proof because the source does not report ENOB/linearity.
- The electronic DAC source proves useful linearity but at too much power and
  too little sample rate for the target.
- The segmented photonic modulator source proves speed and optical-level
  generation, but only at 2 bits. Helpful architecture path, not enough
  precision. Tiny annoying detail, naturally.

Updated report output:

```bash
python -m optical_spike --run-architecture-converter-profile-report --output-dir artifacts/athermal-dac-linearity-2026-06-23
```

Local artifacts:

- `artifacts/athermal-dac-linearity-2026-06-23/architecture_converter_profile/architecture_converter_profile_rows.csv`
- `artifacts/athermal-dac-linearity-2026-06-23/architecture_converter_profile/architecture_converter_profile_summary.csv`
- `artifacts/athermal-dac-linearity-2026-06-23/architecture_converter_profile/architecture_converter_profile_summary.md`
- `artifacts/athermal-dac-linearity-2026-06-23/architecture_converter_profile/architecture_converter_profile_manifest.json`

Next work should look for a single source-chain candidate that closes all three
DAC constraints at once: `10 GS/s` class rate, `>= 7.35 ENOB` equivalent
linearity, and near-`2.5 pJ/sample` energy.
