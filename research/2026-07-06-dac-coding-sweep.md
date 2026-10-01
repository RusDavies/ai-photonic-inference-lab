# DAC Source Coding Sweep

Date: 2026-07-06

## Question

The previous DAC source-chain search did not find a single demonstrated
DAC/modulator candidate that simultaneously met:

- 10 GS/s
- >= 7.35 ENOB
- >= 50 dBc SFDR
- near 2.5 pJ/sample

The follow-up question was whether the architecture can survive by relaxing DAC
precision, reducing electrical DAC count, or using lower-resolution optical
source coding plus calibration.

## Method

Added `--run-architecture-dac-coding-sweep-report`.

The report reuses the selected `athermal_compensated_mrr` physical-drift/noise
rows and the `unshared_4_block` energy/accuracy Pareto rows. It sweeps:

- source precision from 2 to 6 bits;
- direct lower-resolution drive;
- calibrated lower-resolution drive;
- redundant two-phase source coding;
- shared input/output boundary, ADC-after-each-block, and full O/E/O converter
  placements.

The report keeps the Kull 2018 8-bit SAR ADC profile as the output ADC anchor
and models the DAC precision relaxation as an explicit accuracy penalty. This
is a pruning model, not proof that the surviving source coding will work in the
trained optical transfer simulation.

Artifacts:

- `artifacts/athermal-dac-coding-sweep-2026-07-06/architecture_dac_coding_sweep/architecture_dac_coding_sweep_rows.csv`
- `artifacts/athermal-dac-coding-sweep-2026-07-06/architecture_dac_coding_sweep/architecture_dac_coding_sweep_summary.csv`
- `artifacts/athermal-dac-coding-sweep-2026-07-06/architecture_dac_coding_sweep/architecture_dac_coding_sweep_summary.md`

## Result

The sweep generated 900 rows and 45 summary rows.

The lowest source precision that passed all gates was 6 bits, and only with
redundant two-phase coding. It produced 12 passing rows across the modeled
placements and hardware cases. The best energy ratio was about 0.226x digital
for the shared input/output boundary, with a median effective minimum accuracy
of about 85.05%.

Direct lower-resolution drive did not pass the 85% effective minimum-accuracy
floor for 2-6 bits. Calibrated lower-resolution drive at 6 bits produced a few
passing rows but had median effective minimum accuracy below 85%, so it is not
robust enough to call a decision.

## Decision

Do not declare the DAC/source-chain problem solved. The current evidence says:

- Below 6 bits is not credible for this candidate under the current penalty
  model.
- Direct low-resolution DAC relaxation is not enough.
- Calibration helps but is marginal unless paired with redundant coding.
- Redundant 6-bit source coding is the only surviving branch worth validating.

The DAC is not yet a hard blocker, but it remains the main architecture risk.

## Next Step

Validate the 6-bit redundant source-coding case in the actual transfer
simulation, including explicit source quantization/noise, calibration, and
energy/latency accounting. If that fails, treat the DAC/source-chain as a
blocker for this cheap-inference architecture.
