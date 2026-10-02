# Citation Metadata Note

Date: 2026-10-02

## Purpose

Build draft citation metadata for the selected target venue family:
applied photonic-computing systems venues that accept simulation-heavy
architecture and co-design papers.

Primary artifact:

- [`research/manuscript-primary-sources.bib`](manuscript-primary-sources.bib)

Source control file:

- [`research/2026-08-10-paper-primary-source-matrix.md`](2026-08-10-paper-primary-source-matrix.md)

## Coverage

The BibTeX file covers the primary-source set from the paper matrix:

- photonic neural-network architectures and tensor processors;
- MRR weight banks, control, thermal drift, and stabilization;
- physical neural-network training, analog transfer, and hardware-aware
  robustness;
- all-optical nonlinear activation mechanisms;
- photonic power accounting, O-E-O boundary modeling, ADC/DAC datasets,
  component data sheets, and DAC/source-chain near misses.

The metadata is suitable for manuscript drafting and related-work organization.
It is not final submission metadata.

## Venue-Family Fit

The selected venue family needs citations that support a systems/co-design
argument, not a triumphalist hardware claim. Citation usage should therefore:

- lead with primary sources for demonstrated photonic architectures, MRR
  calibration/control, physical training, converter limits, and power
  accounting;
- preserve source status, including preprint, data-sheet, dataset, and
  weaker-venue caveats;
- use reviews only for taxonomy if later added;
- avoid citing adjacent photonic hardware papers as validation of this
  project's exact energy, latency, or device assumptions.

## Drafting Priorities

High-priority citation groups:

- photonic power and system accounting: `tait2022quantifyingPowerSiliconPhotonicNeuralNetworks`;
- physical training and analog transfer: `wright2022deepPhysicalNeuralNetworks`,
  `vadlamani2023transferableLearningAnalogHardware`,
  `xu2024controlFreeIntegratedPhotonicNeuralNetworks`,
  `xu2026physicalNeuralNetworksSharpnessAware`;
- MRR control and drift: `tait2016multiChannelControlMicroringWeightBanks`,
  `dwivedi2020photonicThermalMicrorings`,
  `arbabi2024thermalCrosstalkProgrammablePhotonic`;
- converter/source limits: `wang2022inpPhotonicIntegratedMultilayerNeuralNetworks`,
  `murmann2026adcPerformanceSurvey`, `kull2018timeInterleavedSarAdc`,
  `analogDevicesAd9172`, `pelgrom2022energyConsumptionBoundsDac`;
- nonlinear activation context: `miscuglio2018allOpticalNonlinearActivation`,
  `jha2020reconfigurableAllOpticalActivation`,
  `he2026twoPhotonAbsorptionActivation`.

Medium-priority citation groups:

- optical neural-network architecture context:
  `shen2017coherentNanophotonicCircuits`,
  `xu2021elevenTopsPhotonicConvolutional`,
  `feldmann2021parallelConvolutionalPhotonicTensorCore`,
  `meyer2026integratedPhotonicTensorProcessor`,
  `largeScalePhotonicAccelerator2025`;
- compact MRR neuron context:
  `tait2019siliconPhotonicModulatorNeuron`,
  `zhang2026compactReconfigurablePhotonicNeurons`.

Lower-priority or caution-use entries:

- `huang2022currentSteeringDac10gs` is an energy/rate near miss from a weaker
  publication/source class. Use only as a near-miss example, not as a primary
  feasibility anchor.
- Entries with `others` or placeholder author groups need final author-list
  verification before submission.

## Known Metadata Gaps

- Several entries need full author lists before submission.
- The 2025/2026 MRR scalability and large-scale accelerator entries should be
  checked for final venue metadata.
- The Arbabi thermal-crosstalk source should be replaced with final venue
  metadata if available.
- Data sheets and author-maintained datasets should keep access dates in the
  final manuscript or supplement.

## Verification

The BibTeX file is intentionally conservative: every entry includes a key,
title, year, URL or DOI, and source-status note. The notes preserve the caveats
from the source matrix so drafting does not flatten weak and strong sources into
the same evidential tier.

