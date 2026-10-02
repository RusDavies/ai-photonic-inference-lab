# Paper-Track Positioning Note

Date: 2026-08-10

## Purpose

Position the current optical inference work as a possible publishable research
track, without overstating what the simulation has proved.

## Short Positioning

The strongest paper direction is a hardware-aware simulation and architecture
limits paper for chained optical interference inference engines.

The claim should not be that optical inference is generally solved, or that this
work demonstrates a buildable cheap frontier-model accelerator. The defensible
claim is narrower:

> Chained optical interference blocks can remain accuracy-viable under modeled
> drift, detector/readout noise, source coding, and calibration constraints, but
> the scaling case depends more on converter placement, source precision,
> calibration policy, and device stabilization than on the raw optical matrix
> multiply alone.

That framing fits the evidence better than a pure positive architecture paper.
The work is most interesting where it exposes the boundary between promising
optical compute and system-level converter reality.

## Current Evidence Spine

- The first useful target moved from a single optical projection to a
  transformer-style MLP-up block, then to chained material-inspired optical MLP
  blocks.
- The unshared 4-block chain became the strongest candidate after training each
  block separately and amortizing converter costs across a longer optical or
  analog island.
- The energy/accuracy Pareto report found an unshared 4-block case with
  calibrated accuracy around `85.67%`, best modeled energy around `0.257x` of
  the digital chain, and best modeled latency around `0.313x` of the digital
  chain.
- Direct physical drift/noise modeling and the calibration bridge narrowed the
  plausible hardware path to stabilized or athermal-compensated MRR-style
  assumptions, not passive open-loop resonant weight banks.
- The DAC/source-chain search found no single demonstrated source-chain
  component meeting the modeled `10 GS/s`, `>= 7.35 ENOB`, `>= 50 dBc SFDR`,
  and near-`2.5 pJ/sample` target.
- Lower-resolution direct DAC relaxation did not clear the modeled accuracy
  gate. Redundant source coding was the first credible relaxation path.
- The initial 6-bit redundant source validation stayed near the threshold but
  had seed/time-step dips below the strict `85%` floor.
- Persistent-failure diagnosis traced those dips mostly to weak training margin
  in seed `11`, not to transient one-off events.
- Stronger 10-epoch training lifted the weak seed enough to clear the stress
  matrix.
- The full 3-seed, 10-epoch source-coding calibration stress confirmation
  cleared the strict no-below-`85%` gate across 72 summary scenarios, with zero
  below-floor row-level observations.

## Meaning For Scaling Optical Interference Engines

The scaling result is conditional but useful: isolated optical accelerators are
unlikely to win if every small matrix operation pays a full electrical/optical
conversion toll. Scaling looks more plausible when multiple optical
interference operations are chained before returning to the digital domain.

This shifts the core research question from "can optics multiply matrices?" to
"how long can an optical or analog island remain accurate, calibrated, and cheap
before source, readout, drift, and control overhead erase the gain?"

For this project, scaling optical interference engines appears to require:

- longer optical compute islands, not one-off optical layers;
- hardware-aware training with enough margin above the task accuracy floor;
- explicit calibration and recalibration policy;
- a stabilized device profile, especially for MRR-style implementations;
- source coding or converter placement that avoids paying high-precision DAC
  cost at every block;
- readout assumptions that remain compatible with the energy and latency target.

The architecture is therefore not primarily a "better optical multiply" story.
It is a co-design story: model training, optical transfer error, converter
placement, source coding, readout, drift control, and calibration all determine
whether the optical engine scales.

## Best Paper Framing

Recommended framing:

**A limits-and-co-design paper for chained photonic neural inference under
converter, calibration, and device-stability constraints.**

Possible title direction:

**Chained Photonic Inference Under Converter and Calibration Constraints**

This is better than a triumphalist accelerator title because the converter wall
is not a side caveat. It is the central design boundary.

The paper can still be constructive: the result says where optical interference
may work, what needs to be amortized, and which assumptions kill the design.
That is valuable even if the final conclusion is "not yet buildable without a
better source/converter stack."

## Likely Venue Fit

Decision update 2026-10-02:

The selected target venue family is **applied photonic-computing systems venues
that accept simulation-heavy architecture and co-design papers**. See
[`research/2026-10-02-target-venue-family.md`](2026-10-02-target-venue-family.md)
for scope, tone, experiment, and citation implications.

Best-fit venues are probably architecture, photonics systems, or applied
photonic-computing venues that accept simulation-heavy co-design work:

- photonic neural-network systems and integrated photonics venues;
- accelerator architecture workshops or conferences, if positioned around
  converter-aware modeling and energy/latency accounting;
- applied optics/photonics venues if the device-bridge and MRR stabilization
  assumptions are made more source-grounded;
- neuromorphic or analog computing workshops if the paper emphasizes
  calibration, drift, and physical transfer rather than a specific MRR device.

Weak venue fit:

- a hardware demonstration venue, unless physical measurements are added;
- a machine-learning venue, unless the workload moves beyond Fashion-MNIST and
  the learning contribution becomes more general;
- a pure device paper, because the current work does not introduce a new device.

## Related-Work Buckets

The literature positioning should be organized by problem boundary rather than
by device taxonomy alone:

- photonic matrix multiplication and optical neural-network architecture
  reviews;
- MZI, MRR, diffractive, wavelength-division, and tensor-processor approaches;
- physical neural-network training, hardware-aware training, sharpness-aware
  training, and analog transfer;
- material or all-optical nonlinear activation mechanisms;
- calibration, drift, thermal crosstalk, resonance control, and MRR
  stabilization;
- ADC/DAC, TIA, detector, source coding, and data-conversion overhead in analog
  or photonic accelerators;
- accelerator-level energy and latency modeling for analog/photonic systems.

The source log already contains useful seeds for these buckets, but it needs a
paper-specific pass before any manuscript outline should cite it as complete.

## CIFAR-10 Result Classification

Status 2026-10-02:

| Result | Classification | Manuscript use |
| --- | --- | --- |
| Dense flattened-pixel and MLP-up CIFAR targets | Negative target-selection result | Use only to explain why a stronger target was needed. |
| CIFAR spatial-feature target | Diagnostic target improvement | Use as a bounded harder-workload target, not as modern CIFAR-10 performance. |
| Mild/moderate signed-tiled CIFAR stress matrix | Positive bounded architecture diagnostic | Use only under the explicit mild/moderate architecture envelope. |
| Severe signed-tiled CIFAR stress matrix | Negative/failure-boundary result | Use as a stress boundary, not as harder-workload success. |

## Claims To Make

- Chaining optical blocks changes the energy/latency trade because converter
  costs can be amortized.
- The simulated unshared 4-block optical MLP-up chain can clear a strict
  accuracy floor under the selected athermal-compensated MRR profile when
  training margin is sufficient.
- Converter/source-chain assumptions dominate the scaling case.
- Direct low-resolution source relaxation is insufficient in this model.
- Redundant source coding plus calibration is a promising but not free escape
  route.
- Hardware-aware training and calibration are not cleanup steps; they are core
  architecture requirements.
- Device-stability assumptions must be mapped into physical drift, resonance,
  photon/SNR, readout, and recalibration quantities before hardware claims are
  credible.
- The bounded CIFAR-10 spatial-feature target is useful as a harder-workload
  diagnostic: mild/moderate signed-tiled projection assumptions survive after
  calibration, while the severe branch is a failure boundary.

## Claims Not To Make Yet

- Do not claim a demonstrated physical accelerator.
- Do not claim general vision or language model applicability.
- Do not claim frontier-model inference economics.
- Do not claim the DAC problem is solved.
- Do not claim passive open-loop optical inference is plausible.
- Do not claim broad transfer beyond Fashion-MNIST. The CIFAR-10 evidence is a
  bounded spatial-feature diagnostic, not a general vision result.
- Do not claim CIFAR-10 harder-workload success under the severe signed-tiled
  projection branch.
- Do not report energy/latency numbers without clearly marking modeled
  assumptions and converter-placement conditions.

## Gap To A Publishable Paper

Minimum next work before drafting the paper as if it were real:

- Build an evidence ledger mapping each claim to research notes, artifact
  directories, commands, summary metrics, and caveats.
- Expand the seed count beyond `7,11,13` or justify why the paper is a methods
  demonstration rather than a robustness claim.
- Add at least one harder workload or a deliberately scoped argument for why
  Fashion-MNIST is only a controlled transfer benchmark. Status 2026-10-02:
  bounded CIFAR-10 spatial-feature evidence exists, but it supports only a
  narrow mild/moderate architecture diagnostic unless a stronger CIFAR target
  or a deliberately narrow manuscript claim is chosen.
- Run ablations that isolate source coding, drift/noise, recalibration policy,
  readout noise, and training margin.
- Convert the source log into a paper-specific related-work matrix with access
  dates and primary-source status.
- Decide the manuscript posture: positive architecture result, limits paper, or
  converter-wall/co-design paper.

## Decision

Proceed as a limits-and-co-design paper track. Treat the current result as a
promising modeled architecture boundary, not a hardware demonstration.

The next artifact should be a paper outline that keeps the converter wall at
the center instead of hiding it in a limitations paragraph at the end, where bad
papers go to pretend they have not noticed their own problem.
