# Paper Primary-Source Literature Matrix

Date: 2026-08-10

Purpose: collect paper-specific primary literature for the related-work and
positioning section. This file complements `research/source-log.md`; it filters
that broader source log into manuscript-facing buckets, marks source status, and
records how each paper supports or constrains the local chained-photonic
inference claim.

Access date for all entries: 2026-08-10.

## Source-Use Rules

- Use primary papers, preprints, data sheets, or author-maintained datasets for
  claims about demonstrated devices, circuits, training methods, or measured
  performance.
- Use reviews only for taxonomy and broad framing, not for paper-specific
  evidence claims.
- Keep local claims conditional where the cited paper demonstrates a different
  workload, device class, scale, or measurement regime.

## Matrix

| Bucket | Primary source | Status | Use in this paper | Supports | Complicates or limits |
| --- | --- | --- | --- | --- | --- |
| Photonic neural networks / MZI meshes | Y. Shen et al., "Deep learning with coherent nanophotonic circuits," Nature Photonics 2017. URL: https://www.nature.com/articles/nphoton.2017.93 | Primary journal paper | Foundational integrated coherent nanophotonic neural-network demonstration; cite when introducing MZI-mesh PNNs and optical matrix multiplication. | Shows on-chip coherent photonic circuits can implement trained linear transforms for neural inference. | Small network/demo scale; not a converter-amortized multi-block system; not evidence for this project's energy model. |
| Photonic tensor cores / convolution | X. Xu et al., "11 TOPS photonic convolutional accelerator for optical neural networks," Nature 2021. URL: https://www.nature.com/articles/s41586-020-03063-0 | Primary journal paper | Cite as a demonstrated optical convolution / tensor-core branch. | Gives a concrete high-throughput optical neural-network accelerator reference. | Convolution/tensor-core framing differs from the local MLP-up chained optical island; system-level converter accounting must be compared carefully. |
| Photonic tensor cores / convolution | J. Feldmann et al., "Parallel convolutional processing using an integrated photonic tensor core," Nature 2021. URL: https://www.nature.com/articles/s41586-020-03070-1 | Primary journal paper | Cite as a frequency-comb / WDM tensor-core demonstration. | Supports the point that optical parallelism can be real in integrated photonic processors. | Mostly linear convolutional processing; nonlinear activation and full-system data movement remain separate constraints. |
| Photonic tensor processors | L. Meyer et al., "Deep neural network inference on an integrated, reconfigurable photonic tensor processor," Nature Communications 2026. URL: https://www.nature.com/articles/s41467-026-71599-2 | Primary journal paper | Cite as a recent rack-integrated photonic inference processor with an electronic/PyTorch interface. | Helps position the paper against modern hardware-backed photonic inference, not only toy optical components. | Reported processor scale and interface are not equivalent to the local simulated chained MLP-up island. |
| Large-scale photonic accelerators | "An integrated large-scale photonic accelerator with ultralow latency," Nature 2025 / open PMC mirror. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC11981923/ | Primary journal paper | Cite as a current large-scale integrated photonic MAC accelerator example. | Useful for latency and integration context around >10k-component photonic systems. | Does not remove this project's DAC/source-chain and calibration burden; treat as adjacent architecture evidence. |
| MRR weight banks / calibration | A. N. Tait et al., "Multi-channel control for microring weight banks," Optics Express 2016. URL: https://opg.optica.org/oe/abstract.cfm?uri=oe-24-8-8895 | Primary journal paper | Cite for MRR weight-bank control, thermal crosstalk, and calibration burden. | Directly supports the claim that MRR banks require calibration/control rather than passive optimism. | The paper is control-method evidence, not a proof that the local athermal-compensated profile is sufficient. |
| MRR weight banks / neuromorphic photonics | A. N. Tait et al., "Silicon photonic modulator neuron," Physical Review Applied 2019. URL: https://journals.aps.org/prapplied/abstract/10.1103/PhysRevApplied.11.064043 | Primary journal paper | Cite for broadcast-and-weight / MRR-modulator neuron architecture context. | Supports the relevance of microring modulator/weighting structures to neuromorphic photonics. | Architecture and neuron model differ from the local fixed-fabrication transfer model. |
| MRR weight banks / scalability | W. Zhang et al., "Compact, reconfigurable, and scalable photonic neurons by modulation-and-weighting microring resonators," 2025/2026. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12886301/ | Primary journal paper | Cite as a recent MRR modulation-and-weighting bank design with scalability motivation. | Useful for explaining why compact MRR banks are attractive implementation candidates. | Recent, specific design assumptions; source status and final citation metadata should be checked before manuscript submission. |
| MRR drift / thermal modeling | A. Arbabi et al. / IEEE, "Thermal Crosstalk Modelling and Compensation Methods for Programmable Photonic Integrated Circuits," 2024. URL: https://arxiv.org/html/2404.10589v1 | Primary preprint / conference-style manuscript | Cite for thermal-crosstalk compensation and resonance-shift modeling in programmable PICs. | Supports translating normalized drift into resonance-shift and compensation budgets. | Preprint source; should be replaced with final venue metadata if published. |
| MRR thermal sensitivity | S. Dwivedi et al., "Photonic and Thermal Modelling of Microrings in Silicon, Diamond and GaN," 2020. URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7279479/ | Primary journal paper | Cite for order-of-magnitude thermal resonance-shift anchors. | Supports the project bridge from temperature excursion to resonance shift. | Device/material geometry differs from the selected local MRR profile; do not overfit its numbers. |
| Physical neural-network training | L. G. Wright et al., "Deep physical neural networks trained with backpropagation," Nature 2022. URL: https://www.nature.com/articles/s41586-021-04223-6 | Primary journal paper | Cite for physics-aware training of physical systems with imperfections and model mismatch. | Supports treating training/calibration as core to physical inference, not post-hoc cleanup. | General physical-neural-network result; not specific to this project's MRR/source-coding assumptions. |
| Analog transfer / hardware-aware learning | S. K. Vadlamani, D. Englund, and R. Hamerly, "Transferable learning on analog hardware," Science Advances 2023. URL: https://www.science.org/doi/10.1126/sciadv.adh3436 | Primary journal paper | Cite for transfer learning to analog hardware and robustness to non-ideal deployment. | Supports the model-to-physical-transfer framing used in the simulation pipeline. | Analog hardware setting is broader than photonic MRR inference; local evidence still needs its own ablations. |
| Physical robustness training | T. Xu et al., "Physical neural networks using sharpness-aware training," Nature Communications 2026. URL: https://www.nature.com/articles/s41467-026-68470-9 | Primary journal paper | Cite for robustness-focused physical neural-network training across post-deployment perturbations. | Reinforces the local finding that training margin controls whether stress cases clear the accuracy floor. | Uses its own physical platforms and tasks; not evidence that 3 local seeds are enough. |
| Control-free / hardware-aware PNNs | T. Xu et al., "Control-free and efficient integrated photonic neural networks via hardware-aware training and pruning," Optica 2024. URL: https://opg.optica.org/optica/fulltext.cfm?uri=optica-11-8-1039 | Primary journal paper | Cite for hardware-aware training/pruning in integrated PNNs, including MRR-based cases. | Useful contrast to active-control-heavy MRR assumptions. | "Control-free" does not automatically validate the local athermal-compensated profile; compare disturbance scales explicitly. |
| All-optical nonlinear activation | M. Miscuglio et al., "All-optical nonlinear activation function for photonic neural networks," Optical Materials Express 2018. URL: https://opg.optica.org/ome/fulltext.cfm?uri=ome-8-12-3851 | Primary journal paper | Cite for induced-transparency and reverse-saturable-absorption activation concepts. | Supports material-inspired nonlinear activation as a legitimate PNN design branch. | Reported MNIST-style inference assumes simplified network/noise treatment; not a direct hardware validation of this project's nonlinear block. |
| All-optical nonlinear activation | A. Jha, C. Huang, and P. R. Prucnal, "Reconfigurable all-optical nonlinear activation functions for neuromorphic photonics," Optics Letters 2020. URL: https://doi.org/10.1364/OL.398234 | Primary journal paper | Cite for reconfigurable all-optical activation device behavior. | Supports programmable optical nonlinearities as more than a software abstraction. | Device response, insertion loss, and operating point must be reconciled with chained-block energy/accuracy assumptions. |
| All-optical nonlinear activation | M. He et al., "All-optical nonlinear activation function based on two-photon absorption for photonic neural networks," APL Photonics 2026. URL: https://pubs.aip.org/aip/app/article/11/1/016101/3375966/All-optical-nonlinear-activation-function-based-on | Primary journal paper | Cite as recent experimental TPA activation work in SOI and Ge-on-Si waveguides. | Supports broad-band, high-speed all-optical activation motivation and O-E-O avoidance. | Demonstrated response frequencies and MNIST CNN use do not prove suitability for the local MLP-up chain. |
| Power / benchmarking | A. N. Tait, "Quantifying Power in Silicon Photonic Neural Networks," Physical Review Applied 2022. URL: https://doi.org/10.1103/PhysRevApplied.17.054029 | Primary journal paper | Cite for system-level power accounting, scaling laws, and photonic performance metrics. | Directly supports the paper's argument that photonic accelerator energy cannot be summarized by optical MACs alone. | The local energy ratios remain model-specific; do not cite this as validation of the exact 0.257x figure. |
| Power / O-E-O boundaries | Y. Wang et al., "InP photonic integrated multi-layer neural networks," APL Photonics 2022. URL: https://pubs.aip.org/aip/app/article/7/1/010801/2835098/InP-photonic-integrated-multi-layer-neural | Primary journal paper | Cite for all-optical vs O/E/O multilayer energy estimates and converter assumptions. | Gives a primary anchor for DAC/ADC/receiver power at 10 GHz-class assumptions. | InP platform and modeling assumptions are not the same as the selected MRR branch. |
| ADC targets | B. Murmann, "ADC Performance Survey 1997-2026." URL: https://github.com/bmurmann/ADC-survey | Author-maintained primary dataset | Cite for ADC performance landscape and energy/precision feasibility checks. | Useful for judging whether ADC energy/ENOB targets are ordinary, aggressive, or fantasy. | Survey data is not a single device demonstration; filter by bandwidth/ENOB before comparison. |
| ADC targets | P. Kull et al., "A 24-72-GS/s 8-b Time-Interleaved SAR ADC With 2.0-3.3-pJ/Conversion and >30 dB SNDR at Nyquist in 14-nm CMOS FinFET," IEEE JSSC 2018. URL: https://doi.org/10.1109/JSSC.2017.2775482 | Primary journal paper | Cite as a concrete high-speed ADC energy/rate data point. | Supports `<=5 pJ/sample`-class output conversion as plausible but demanding. | SNDR/ENOB and bandwidth conditions differ from the local readout specification; do not overclaim. |
| DAC/source chain | Analog Devices, "AD9172: Dual, 16-Bit, 12.6 GSPS RF DAC with Channelizers." URL: https://www.analog.com/media/en/technical-documentation/data-sheets/ad9172.pdf | Primary data sheet | Cite as the nearest commercial rate/SFDR anchor found in the local search. | Clears rate/SFDR reference needs while showing the energy miss is huge. | Product data sheet lacks local ENOB fit and is far above the modeled energy target. |
| DAC/source chain | R. Pelgrom et al., "Analysis of energy consumption bounds in CMOS current-steering digital-to-analog converters," Analog Integrated Circuits and Signal Processing 2022. URL: https://link.springer.com/article/10.1007/s10470-022-02013-2 | Primary journal paper | Cite for DAC energy bounds under noise, speed, and linearity constraints. | Supports keeping SFDR/INL/DNL/ENOB in the DAC table instead of rate and pJ/sample only. | Bounds analysis is not a passing source-chain implementation. |
| Electro-optic DAC / modulator | A. Samani et al., "Silicon Photonic Segmented Modulator-Based Electro-Optic DAC for 100 Gb/s PAM-4 Generation," IEEE Photonics Technology Letters 2015. URL: https://ieeexplore.ieee.org/document/7185365 | Primary journal paper | Cite for segmented electro-optic DAC / modulator route. | Supports optical-DAC rate feasibility for low-bit optical communication style modulation. | PAM-4 / 2-bit style precision does not satisfy this project's 7-8-bit effective source-chain burden. |
| DAC near-miss | T. Huang et al., "A 10GS/s 8-bit Current Steering DAC in 65nm CMOS Technology," WCSE 2022. URL: https://www.wcse.org/content-14-499-1.html | Primary paper / simulation report | Cite only as an energy/rate near-miss. | Shows a simulated 10 GS/s, 8-bit, few-pJ/sample-class DAC point. | SFDR is below the project target; publication venue and simulation status make it weaker than JSSC/data-sheet anchors. |

## Related-Work Positioning Buckets

### Photonic Neural Networks And Architectures

Use Shen 2017, Xu 2021, Feldmann 2021, Meyer 2026, and the large-scale
photonic accelerator paper to show that optical matrix/tensor processing is a
live hardware direction. The local paper should then pivot quickly: the missing
question is not whether optics can multiply, but whether a chained optical
island remains accurate and cheap after converters, source drive, readout,
calibration, and drift are included.

### MRR Weight Banks, Drift, And Stabilization

Use Tait's microring weight-bank work, recent modulation-and-weighting MRR
neurons, and thermal/resonance modeling sources to justify why the selected
`athermal_compensated_mrr` profile is a cautious branch. This bucket should also
prevent passive open-loop language from sneaking into the manuscript wearing a
fake mustache.

### Physical Training And Calibration

Use Wright 2022, Vadlamani 2023, Xu 2024, and Xu 2026 to place the local
training-margin and calibration results in the physical-learning literature.
The local paper's novelty is not "physical training exists"; it is using
training margin and calibration policy as architecture gates for a chained
photonic inference island.

### Nonlinear Activation

Use Miscuglio 2018, Jha 2020, and He 2026 to motivate material or all-optical
nonlinear activation options. Keep the local claims modest: the project's
reverse-saturable/material-inspired functions are simulation candidates anchored
by the existence of optical nonlinear activation mechanisms, not direct device
measurements.

### Converter, Source, And Benchmarking Limits

Use Tait 2022, Wang 2022, Murmann's ADC survey, Kull 2018, AD9172, Pelgrom
2022, and Samani 2015 to make the converter wall central. The result section
should treat source-chain failure as evidence, not as an inconvenient errand
for some future graduate student with more caffeine than self-preservation.

## Citation Priorities For A First Manuscript Draft

High priority:

- Tait 2022 for photonic power accounting.
- Wright 2022, Vadlamani 2023, Xu 2024, and Xu 2026 for physical training and
  robustness.
- Tait 2016 plus thermal microring modeling for MRR calibration and drift.
- Wang 2022, Murmann, Kull 2018, AD9172, and Pelgrom 2022 for converter limits.
- Miscuglio 2018 / Jha 2020 / He 2026 for nonlinear activation context.

Medium priority:

- Shen 2017, Xu 2021, Feldmann 2021, Meyer 2026, and the large-scale photonic
  accelerator paper for architecture positioning.
- Recent compact MRR neuron papers if the final manuscript leans harder into
  MRR implementation.

Lower priority / taxonomy only:

- Broad PNN reviews and photonic matrix multiplication reviews. Useful for
  organizing the introduction, but not enough for device or energy claims.

## Manuscript Gaps Exposed By The Source Pass

- The athermal-compensated MRR profile still needs device-specific primary
  anchors if it becomes a central hardware claim.
- The DAC/source chain still has no passing primary component candidate meeting
  the local rate, precision, SFDR, and energy target together.
- The nonlinear activation sources support plausibility, but the local
  material-inspired activation model needs explicit caveats because it is not a
  measured transfer function.
- The physical-training literature strengthens the argument for training margin
  and calibration, but it also raises the bar for ablations separating drift,
  readout, source coding, recalibration policy, and training margin.

## Decision

Use this matrix as the paper-specific related-work control file. It is enough to
start drafting the background section, but not enough to make a strong hardware
claim about a fabricated MRR implementation or a solved source/converter stack.
