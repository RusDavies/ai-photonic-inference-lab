# Paper Outline

Date: 2026-08-10

Working title:

**Chained Photonic Inference Under Converter and Calibration Constraints**

## One-Sentence Thesis

Chained optical interference blocks can be accuracy-viable in a hardware-aware
simulation when multiple optical operations are kept inside one analog island,
but the scaling case is governed by converter placement, source precision,
calibration policy, training margin, and stabilized device assumptions rather
than by optical matrix multiplication alone.

## Candidate Abstract

Photonic neural-network accelerators promise high-throughput matrix operations,
but system-level benefits are often limited by electrical/optical conversion,
readout, calibration, and device drift. This paper studies a chained photonic
inference architecture in which multiple optical MLP-up style blocks are
evaluated before returning to the digital domain. Using a controlled
Fashion-MNIST transfer benchmark, the study models trained chained material
blocks, constrained fixed-fabrication transfer, athermal-compensated microring
resonator assumptions, source-coding limits, detector and output-readout noise,
and scheduled or projection-error-triggered recalibration. The strongest
unshared 4-block candidate reaches modeled energy and latency below the digital
chain under selected converter placements, while stronger training clears a
strict no-below-85% accuracy floor across the 3-seed, 10-epoch source-coding
stress confirmation. The results suggest that photonic inference scaling is
most plausible as longer optical or analog compute islands, not as isolated
optical layers. A bounded CIFAR-10 spatial-feature diagnostic further shows
that mild/moderate signed-tiled projection assumptions can preserve a harder
target after calibration, while the severe branch remains a failure boundary.
The results also show that the DAC/source chain and physical stabilization
assumptions remain the dominant risks. The contribution is a limits-and-
co-design analysis, not a physical hardware demonstration.

## Intended Contribution

The paper should make three contributions:

1. A chained optical inference simulation workflow that moves from single-layer
   optical transfer to multi-block material-inspired optical MLP-up chains.
2. A converter-aware energy/latency and accuracy analysis showing when chained
   optical islands remain interesting and when converter/readout costs erase
   the advantage.
3. A device-facing constraint bridge that translates normalized drift/noise
   assumptions into resonance, thermal, photon/SNR, readout, and recalibration
   quantities for stabilized MRR-style profiles.

The fourth, more cautious contribution is the warning: the DAC/source chain is
the design wall. It should be presented as a result, not hidden as fine print.

## Paper Structure

### 1. Introduction

Purpose:

Frame the problem as system-level photonic inference scaling, not another
generic "optics does matrix multiply" story.

Key points:

- Optical interference can perform useful linear operations cheaply in
  principle.
- Practical inference engines pay for modulation, DACs, ADCs, detectors, TIAs,
  drift control, and calibration.
- Single optical layers are a poor scaling target if every layer pays the full
  conversion toll.
- The paper asks whether a longer chained optical island can preserve accuracy
  while amortizing conversion and readout overhead.

Main claim:

Chaining helps only when the optical island, training margin, calibration
policy, and converter placement are co-designed.

### 2. Background And Related Work

Purpose:

Separate this paper from device demonstrations and broad photonic neural-network
reviews.

Buckets:

- photonic matrix multiplication and optical neural-network architectures;
- MZI meshes, MRR weight banks, diffractive optics, wavelength-division
  approaches, and tensor processors;
- physical neural-network transfer, hardware-aware training, and robust
  physical training;
- all-optical and material nonlinear activation mechanisms;
- calibration, thermal drift, resonance control, and MRR stabilization;
- ADC/DAC, TIA, detector, and data-conversion overhead in analog and photonic
  accelerators;
- energy/latency modeling for photonic and analog accelerators.

Needed before manuscript:

Turn `research/source-log.md` into a paper-specific related-work matrix with
primary-source status and access dates.

### 3. Architecture Target

Purpose:

Define the candidate engine as a chained optical MLP-up island.

Describe:

- baseline digital Fashion-MNIST classifier and MLP-up block target;
- material-inspired activation candidates;
- shared vs unshared chained blocks;
- why the unshared 4-block chain became the strongest local candidate;
- converter-placement cases: shared input/output boundary, ADC after each block,
  and full O/E/O between blocks;
- selected `athermal_compensated_mrr` physical profile and why passive open-loop
  dense ring banks are not the claim.

Evidence anchors:

- `research/2026-06-21-transformer-mlp-target.md`
- `research/2026-06-21-material-nonlinearity-candidates.md`
- `research/2026-06-21-functional-chained-blocks.md`
- `artifacts/unshared-chained-material-training-2026-06-22/`
- `research/2026-06-22-device-tolerance-reality-check.md`

### 4. Simulation And Calibration Method

Purpose:

Make the experiment reproducible and expose assumptions.

Describe:

- full Fashion-MNIST split vs bounded smoke runs;
- seed set `7,11,13`;
- 5-epoch runs used while exploring;
- 10-epoch confirmation used after seed `11` exposed weak margin;
- constrained fixed-fabrication transfer with signed/tiled severe assumptions;
- direct physical drift/noise scenarios;
- scheduled recalibration and projection-MSE-triggered recalibration;
- source-coding stress matrix: 6-bit redundant two-phase, 7-bit direct, and
  7-bit redundant two-phase;
- accuracy floor: no seed/time-step row below `85%`.

Evidence anchors:

- `artifacts/source-coding-calibration-stress-full-3seed-10epoch-2026-07-06/`
- `research/2026-07-06-redundant-source-validation.md`
- `artifacts/redundant-source-validation-2026-07-06/`

Reproducibility gap:

The paper needs a compact command/evidence ledger before this section is
manuscript-ready.

### 5. Results: Chaining And Energy/Latency

Purpose:

Show why the chained optical island is the interesting unit of analysis.

Expected content:

- compare shared/unshared and 2/4-block chains;
- report the unshared 4-block candidate: about `85.67%` calibrated accuracy,
  best modeled energy ratio about `0.257x`, and best modeled latency ratio
  about `0.313x` of the digital chain;
- show that converter placement changes the interpretation of the result;
- explain why the shared 4-block chain did not carry the same accuracy result.

Figures/tables:

- energy/accuracy Pareto table;
- energy vs accuracy plot for shared/unshared 2/4-block chains;
- converter-energy fraction by placement;
- latency ratio by placement.

Evidence anchors:

- `artifacts/energy-accuracy-pareto-2026-06-22/`
- `artifacts/athermal-converter-targets-2026-06-23/`

### 6. Results: Device Drift, Readout, And Calibration

Purpose:

Show that the physical tolerance story is conditional on stabilization and
calibration.

Expected content:

- normalized drift/noise target envelope;
- calibration bridge from normalized drift/noise into resonance shift, thermal
  excursion, photon/SNR, ENOB-equivalent readout, and recalibration interval;
- why tight MRR weight banks require active control;
- why broader/stabilized or athermal-compensated MRR profiles are the plausible
  branch;
- scheduled vs projection-MSE-triggered recalibration behavior.

Figures/tables:

- drift/noise accuracy time series;
- projection-MSE time series;
- calibration bridge sensitivity table;
- physical scenario pass/fail table.

Evidence anchors:

- `artifacts/unshared-physical-drift-noise-2026-06-22/`
- `artifacts/calibration-bridge-2026-06-22/`
- `artifacts/athermal-physical-drift-noise-2026-06-22/`

### 7. Results: Source Coding And The Converter Wall

Purpose:

Make the source/DAC problem central.

Expected content:

- no single found DAC/modulator source-chain candidate met `10 GS/s`,
  `>= 7.35 ENOB`, `>= 50 dBc SFDR`, and near-`2.5 pJ/sample`;
- direct 2-6 bit source relaxation failed the modeled accuracy gate;
- calibrated low-resolution drive helped but remained marginal;
- 6-bit redundant two-phase source coding was the first credible branch;
- first validation showed marginal below-floor rows;
- seed `11` diagnosis showed the issue was mostly persistent weak training
  margin;
- 10-epoch confirmation cleared the strict floor across the modeled 3-seed
  stress matrix;
- direct 7-bit and redundant 7-bit remain comparison branches, but not free
  wins because source energy/latency still matters.

Figures/tables:

- DAC/source-chain candidate table;
- source-coding sweep pass/fail table;
- persistent failure by seed table;
- 10-epoch confirmation summary table.

Evidence anchors:

- `research/2026-07-01-dac-source-chain-candidate.md`
- `research/2026-07-06-dac-coding-sweep.md`
- `research/2026-07-06-redundant-source-validation.md`
- `artifacts/source-coding-calibration-stress-full-3seed-10epoch-2026-07-06/`

### 8. Discussion

Purpose:

State what the result means and what it does not mean.

Claims to emphasize:

- optical inference scaling is more plausible as chained analog/optical islands
  than as isolated optical layers;
- converter placement is an architecture decision, not an implementation detail;
- calibration and hardware-aware training are part of the model, not cleanup;
- stabilized device assumptions are required for MRR-style implementations;
- bounded CIFAR-10 evidence is useful as a diagnostic, but only under
  mild/moderate architecture assumptions;
- the converter wall remains the main risk.

Claims to avoid:

- no hardware demonstration;
- no general vision-language or frontier-model inference claim;
- no claim that the DAC/source-chain target has a demonstrated component;
- no claim that Fashion-MNIST or the bounded CIFAR-10 spatial-feature target
  proves task-general scaling;
- no CIFAR-10 success claim under severe signed-tiled projection.

### 9. Limitations

Known limitations:

- controlled Fashion-MNIST benchmark remains the primary evidence track;
- CIFAR-10 evidence is bounded to a spatial-feature target and mild/moderate
  architecture assumptions;
- seed set limited to `7,11,13`;
- no physical measurement;
- device bridge relies on sourced anchors and assumptions, not local hardware;
- converter/source energy model is approximate;
- MRR profile is selected from plausibility analysis, not fabricated evidence;
- the strongest result depends on 10-epoch training margin;
- source-coding and calibration interactions need ablation.

This section should be blunt. A paper like this gains credibility by being
honest about the knife edge instead of pretending all error bars are decorative.

### 10. Conclusion

Expected conclusion:

Chained photonic inference can look attractive in a hardware-aware model, but
only when the system is designed as a calibrated optical island with explicit
converter, source, readout, and drift constraints. The paper's contribution is
identifying that viable design envelope and the remaining wall, not proving a
deployable photonic accelerator.

## Core Figures And Tables

- Figure 1: Chained optical MLP-up island and converter-placement options.
- Figure 2: Simulation workflow from digital baseline to optical transfer,
  physical drift/noise, source coding, and recalibration.
- Figure 3: Energy/accuracy Pareto for shared vs unshared chained blocks.
- Figure 4: Physical drift/noise accuracy and projection-MSE time series.
- Figure 5: Source-coding stress results and seed-margin diagnosis.
- Table 1: Related-work positioning matrix.
- Table 2: Candidate architecture assumptions and device bridge anchors.
- Table 3: Converter/source-chain candidate limits.
- Table 4: Claim-to-evidence ledger.

## Minimum Manuscript-Ready Checklist

- Build the claim-to-evidence ledger.
- Convert the source log into a paper-specific related-work matrix.
- Preserve the bounded CIFAR-10 diagnostic classification unless a stronger
  CIFAR target is added.
- Run ablations for source coding, drift/noise, readout, recalibration, and
  training margin.
- Decide the target venue family before writing introduction prose.
- Create a reproducibility appendix with commands, artifact paths, seeds, and
  exact gates.

## Decision

Use this outline as the paper-track skeleton. The next project artifact should
be the claim-to-evidence ledger, because every section above depends on knowing
which local artifact supports which sentence.
