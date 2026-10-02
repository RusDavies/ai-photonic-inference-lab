# Target Venue Family Decision

Date: 2026-10-02

## Decision

Primary target venue family:

**Applied photonic-computing systems venues that accept simulation-heavy
architecture and co-design papers.**

Use this as the target family before polishing introduction prose. The paper
should be written for readers who know photonic neural-network hardware,
integrated photonics, analog/neuromorphic computing, and system-level converter
constraints, but who will not accept unsupported hardware-demonstration claims.

Secondary fit:

- accelerator architecture workshops or special issues, if the framing centers
  converter placement, energy/latency accounting, and analog-island system
  design;
- neuromorphic or analog computing workshops, if the framing centers physical
  transfer, calibration, drift, and hardware-aware training;
- applied optics/photonics venues, if the device bridge and MRR stabilization
  assumptions are strengthened with more source-grounded detail.

Not the target family:

- hardware demonstration venues, unless physical measurements are added;
- mainstream machine-learning venues, unless the workload and learning
  contribution move well beyond the current Fashion-MNIST plus bounded CIFAR
  diagnostic scope;
- pure device venues, because the current work does not introduce or measure a
  new device.

## Scope Implications

The manuscript should present itself as a limits-and-co-design simulation paper:

- primary evidence track: controlled Fashion-MNIST chained optical/material
  MLP-style simulations;
- supporting harder-workload evidence: bounded CIFAR-10 spatial-feature
  diagnostic under mild/moderate signed-tiled architecture assumptions;
- central system result: converter placement, source precision, calibration,
  training margin, and device stabilization dominate the scaling story;
- hardware posture: modeled MRR-style branches and source/converter limits, not
  physical device validation.

This venue family rewards a careful boundary more than a heroic claim. The
paper should show where the idea survives, where it fails, and what assumptions
are doing the work.

## Tone Implications

Use a sober systems tone:

- lead with the converter/calibration/device-stability wall;
- make the optical-compute upside conditional rather than triumphant;
- treat negative source-chain and severe-CIFAR results as evidence, not as
  embarrassing side notes;
- state simulation and benchmark boundaries early, including in the abstract or
  opening methods section.

Avoid:

- "photonic inference is solved" language;
- passive open-loop MRR optimism;
- product, SDK, or accelerator-readiness framing;
- broad vision-model or frontier-model economics language.

## Experiment Expectations

The current evidence package is enough for a scoped systems/co-design argument
only if it stays explicit about its limits.

Minimum experiment expectations for this venue family:

- reproducibility appendix with commands, seeds, artifact paths, assumptions,
  and pass/fail gates;
- claim-to-evidence ledger mapping each manuscript claim to a dated note,
  command, artifact family, metric, and caveat;
- clear split between checked, diagnostic, and exploratory results;
- ablation or explicit limitation around source coding, drift/noise,
  recalibration policy, readout, and training margin;
- cautious treatment of CIFAR-10: bounded diagnostic, not broad generalization;
- source/converter and device-bridge evidence reported as system constraints.

Useful future strengthening before submission:

- expand Fashion-MNIST seed count or justify the paper as a methods/co-design
  demonstration;
- run or explicitly defer the source/drift/readout/recalibration ablations;
- decide whether severe CIFAR remains a failure boundary or gets a separate
  calibration-improvement study;
- add a stronger CIFAR target if the final venue expects more workload breadth.

## Citation Style Implications

Citation metadata should prioritize primary sources and source status:

- photonic neural-network architecture papers for context, not as proof of this
  system's energy claims;
- MRR control, thermal drift, and stabilization sources for device-boundary
  assumptions;
- physical neural-network and analog-transfer papers for hardware-aware
  training/calibration framing;
- ADC/DAC/source-chain papers, surveys, and data sheets for converter limits;
- power-accounting and benchmarking papers for system-level energy framing.

Use reviews sparingly for taxonomy. For hardware-facing claims, cite primary
papers, preprints, data sheets, or author-maintained datasets and preserve
access dates/source status.

## Decision For Next Work

`PIL-006` should build citation metadata for this venue family. It should start
from `research/2026-08-10-paper-primary-source-matrix.md` and preserve source
status, access dates, and caveats rather than flattening every source into a
generic bibliography.

