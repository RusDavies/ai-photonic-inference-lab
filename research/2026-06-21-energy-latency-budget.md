# Energy and Latency Budget

Date: 2026-06-21

## Purpose

Convert the calibration/control count model into a coarse energy and latency
budget for the transformer-style MLP-up optical target.

This is not a device claim. It is a sensitivity model over plausible component
assumptions so the project can see which overheads are fatal and which are just
accounting noise.

## Source Basis

The assumptions are grounded by the project source log:

- photonic accelerator reviews reporting sub-pJ/op to low-pJ/op operating
  regimes;
- a silicon-photonic accelerator estimate around 273 fJ/op;
- reviews warning that ADC, DAC, TIA, data movement, and control electronics can
  dominate practical system energy.

See `research/source-log.md` for links.

## Local Artifacts

- `artifacts/energy-budget-2026-06-21/energy_budget/energy_budget_rows.csv`
- `artifacts/energy-budget-2026-06-21/energy_budget/energy_budget_summary.csv`
- `artifacts/energy-budget-2026-06-21/energy_budget/energy_budget_summary.md`
- `artifacts/energy-budget-2026-06-21/energy_budget/energy_budget_manifest.json`
- `artifacts/platform-energy-budget-2026-06-21/energy_budget/energy_budget_rows.csv`
- `artifacts/platform-energy-budget-2026-06-21/energy_budget/energy_budget_summary.csv`
- `artifacts/platform-energy-budget-2026-06-21/energy_budget/energy_budget_summary.md`
- `artifacts/platform-energy-budget-2026-06-21/energy_budget/energy_budget_manifest.json`
- `artifacts/chained-block-budget-2026-06-21/chained_block_budget/chained_block_budget_rows.csv`
- `artifacts/chained-block-budget-2026-06-21/chained_block_budget/chained_block_budget_summary.csv`
- `artifacts/chained-block-budget-2026-06-21/chained_block_budget/chained_block_budget_summary.md`
- `artifacts/chained-block-budget-2026-06-21/chained_block_budget/chained_block_budget_manifest.json`
- `artifacts/charge-readout-budget-2026-06-21/charge_readout_budget/charge_readout_budget_rows.csv`
- `artifacts/charge-readout-budget-2026-06-21/charge_readout_budget/charge_readout_budget_summary.csv`
- `artifacts/charge-readout-budget-2026-06-21/charge_readout_budget/charge_readout_budget_summary.md`
- `artifacts/charge-readout-budget-2026-06-21/charge_readout_budget/charge_readout_budget_manifest.json`

The artifact directory is intentionally uncommitted.

## Model

Target:

- MLP-up projection: 64x256.
- Optical operation count: 16,384 multiply-accumulate-equivalent operations.
- Digital correction: 512 simple operations per inference.
- Calibration source: `artifacts/calibration-cost-full-2026-06-21/...`.
- Amortization assumption: 10,000 inferences per drift step.

Hardware scenarios:

- `optimistic_low_converter`:
  - Optical op: 0.273 pJ.
  - DAC per input: 0.5 pJ.
  - ADC per output: 1.0 pJ.
  - Digital correction op: 0.03 pJ.
  - Control write: 5 pJ.
- `converter_bound`:
  - Optical op: 0.273 pJ.
  - DAC per input: 5 pJ.
  - ADC per output: 10 pJ.
  - Digital correction op: 0.1 pJ.
  - Control write: 50 pJ.
- `conservative_system`:
  - Optical op: 1.0 pJ.
  - DAC per input: 10 pJ.
  - ADC per output: 20 pJ.
  - Digital correction op: 0.2 pJ.
  - Control write: 100 pJ.

Digital comparison:

- Digital MLP-up energy: 1.0 pJ per MAC-equivalent operation.
- Digital MLP-up latency: 2 ns.

## Results

Best energy ratios versus the digital MLP-up block:

- Optimistic low-converter case: 0.29x digital energy.
- Converter-bound case: 0.45x digital energy.
- Conservative system case: 1.36x digital energy.

Best total energy per inference:

- Optimistic low-converter: 4,777 pJ.
- Converter-bound: 7,406 pJ.
- Conservative system: 22,251 pJ.
- Digital comparison point: 16,384 pJ.

Latency ratio versus the digital MLP-up block:

- Optimistic low-converter: 0.95x digital latency.
- Converter-bound: 3.85x digital latency.
- Conservative system: 8.25x digital latency.

The lowest-energy calibration setting remained the sparse setting:

- 32 calibration samples.
- Recalibrate every 16 drift steps.
- Accuracy drop stayed around 0.11 to 0.13 points across the tested drift rates.

## Interpretation

The calibration loop itself is not the dominant energy cost under the current
amortization assumption. Sparse recalibration makes calibration/control writes
small compared with per-inference optical compute and conversion.

The important question is converter overhead. With good ADC/DAC assumptions, the
MLP-up optical block can beat a 1 pJ/MAC digital comparison by roughly 2x to 3x.
With mediocre converters and control electronics, the energy advantage shrinks
or disappears. With conservative system assumptions, the optical block loses on
both energy and latency.

Latency is less forgiving than energy. The optical compute itself may be fast,
but data conversion dominates unless the interface is excellent or the design
keeps data optical across multiple blocks.

Current decision:

- Energy: qualified go under optimistic and converter-bound assumptions.
- Latency: go only under optimistic converter/interface assumptions.
- System risk: converter, detector/TIA, laser, and control electronics dominate
  the next uncertainty. Annoying, because physics did the fun part and then
  electronics walked in with a clipboard.

## First-Pass Follow-Up

Refine the model with platform-specific measured ADC/DAC, laser, detector, and
control electronics data.

## Platform-Specific Refinement

The follow-up split the hardware assumptions into explicit component buckets:

- optical operation energy;
- DAC input energy;
- ADC output energy;
- detector/TIA output energy;
- laser energy per inference;
- static control energy;
- calibration/control-write amortization.

Five platform scenarios were evaluated:

- `published_273fj_low_io`: silicon-photonic-style 273 fJ/op with low converter
  overhead.
- `photonic_tensor_core_5topsw_io`: 5.1 TOPS/W-style optical core with
  more realistic converter and detector/TIA overhead.
- `inp_nonvolatile_all_optical_estimate`: InP/nonvolatile-style all-optical
  estimate around 0.7 pJ/op plus low I/O overhead.
- `inp_volatile_oeo_effective`: volatile O/E/O InP-style effective energy around
  8 pJ/op.
- `high_precision_adc_bound`: intentionally ugly high-precision ADC-bound case.

Best energy and latency ratios versus the same 1 pJ/MAC digital MLP-up block:

- `published_273fj_low_io`: 0.31x energy, 0.95x latency.
- `photonic_tensor_core_5topsw_io`: 0.44x energy, 3.85x latency.
- `inp_nonvolatile_all_optical_estimate`: 0.73x energy, 1.10x latency.
- `inp_volatile_oeo_effective`: 8.00x energy, 4.25x latency.
- `high_precision_adc_bound`: 16.87x energy, 15.75x latency.

Component reading:

- Low-I/O silicon-style assumptions keep the optical block attractive.
- Tensor-core-style assumptions still win on energy, but converter latency
  erases the latency advantage.
- Nonvolatile/all-optical InP-style assumptions are plausible on energy and
  close on latency.
- Volatile/current-biased O/E/O assumptions are not credible for cheap
  replicated inference at this block size.
- High-precision ADC is fatal. The ADC alone contributes 256,000 pJ per
  inference in the stress case. Yes, the converter ate the accelerator. Rude,
  but predictable.

Updated decision:

- Energy: still a qualified go for low-I/O silicon, tensor-core-style, and
  nonvolatile/all-optical assumptions.
- Latency: only a go if converter latency is very low or if multiple optical
  blocks can be chained before conversion.
- Main architecture pressure: avoid paying ADC/DAC for every tiny block.

## Next Platform Work

Explore chaining multiple optical blocks before ADC to amortize converter
latency and energy.

## Chained Optical Block Budget

The follow-up modeled multiple MLP-up-sized optical blocks kept in the optical
domain before a single ADC/DAC boundary. This is an architecture budget, not a
functional stacked-analog-error simulation. It assumes optical compute scales
with chain length, converter/detector/digital correction cost is paid once at
the chain boundary, and per-block control writes still scale with the number of
optical blocks.

Run context:

- Calibration source: `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv`.
- Chain lengths: 1, 2, 4, and 8 blocks.
- Hardware scenarios: same five platform-specific scenarios.
- Rows: 720.
- Aggregate rows: 20.
- Error-growth proxy: single-block calibrated accuracy drop scaled by square
  root of chain length, used only as a warning flag.

Best energy and latency ratios versus a digital chain of the same number of
MLP-up blocks:

- `published_273fj_low_io`, 8-block chain: 0.28x energy, 0.21x latency.
- `photonic_tensor_core_5topsw_io`, 8-block chain: 0.23x energy, 0.57x latency.
- `inp_nonvolatile_all_optical_estimate`, 8-block chain: 0.70x energy,
  0.36x latency.
- `inp_volatile_oeo_effective`, 8-block chain: 8.00x energy, 0.97x latency.
- `high_precision_adc_bound`, 8-block chain: 2.98x energy, 2.19x latency.

Chaining materially changes the converter-bound cases. The tensor-core-style
case previously won on energy but lost badly on latency; at 8 blocks it wins on
both in this model. The low-I/O silicon-style case becomes stronger still.
The nonvolatile/all-optical estimate also becomes a clear latency win while
staying under digital energy.

The ugly cases remain ugly. Volatile O/E/O still loses on energy because the
effective optical operation cost dominates. High-precision ADC remains above
digital even after 8-block amortization. If the converter is that bad, chaining
helps but does not rescue the architecture. The spreadsheet has spoken, and
for once it is not being subtle.

Updated decision:

- Chaining before ADC is a go as an architectural requirement for
  converter-bound platforms.
- It is not optional for tensor-core-style or nonvolatile-style claims if
  latency matters.
- High-precision ADC is still disqualifying unless the block count rises far
  beyond this toy chain or the converter assumptions improve.

## Next Platform Work

Build a functional chained-block simulation with accumulated analog error,
material activation, and end-to-end accuracy checks.

## CCD-Style Charge Readout

The CCD question was modeled as charge-domain readout rather than magic ADC
removal. The model compares the existing CMOS-like column-parallel reference
against shared charge-bucket readout lanes. CCD-style cases still perform ADC
conversions at the output edge; they reduce parallel readout hardware and some
readout energy, but pay charge-transfer latency and read-noise accumulation.

Readout scenarios:

- `cmos_column_parallel_reference`: 256 parallel lanes.
- `ccd_32_lane_fast_bucket`: 32 shared lanes.
- `ccd_8_lane_low_power_bucket`: 8 shared lanes.
- `ccd_1_lane_serial_stress`: one serial lane.

Run context:

- Calibration source: `artifacts/calibration-cost-full-2026-06-21/calibration_cost/calibration_cost_summary.csv`.
- Hardware scenarios: same five platform-specific scenarios.
- Rows: 720.
- Aggregate rows: 20.
- Output dimension: 256 values.

Representative best ratios versus the digital MLP-up block:

- Low-I/O silicon-style:
  - CMOS reference: 0.31x energy, 1.05x latency.
  - 32-lane CCD bucket: 0.30x energy, 34.15x latency.
  - 8-lane CCD bucket: 0.30x energy, 231.85x latency.
- Photonic tensor-core-style:
  - CMOS reference: 0.44x energy, 3.95x latency.
  - 32-lane CCD bucket: 0.40x energy, 52.65x latency.
  - 8-lane CCD bucket: 0.37x energy, 309.55x latency.
- High-precision ADC-bound:
  - CMOS reference: 16.87x energy, 15.85x latency.
  - 32-lane CCD bucket: 15.15x energy, 142.55x latency.
  - 8-lane CCD bucket: 12.81x energy, 695.45x latency.
  - one-lane serial stress: 10.47x energy, 8095.35x latency.

Interpretation: CCD-style readout can cut readout energy a little when ADC
hardware is expensive, but it is a latency disaster for fast inference unless
the lane count is high and charge-transfer latency is extraordinarily low. It
also adds a read-noise proxy that grows with transfer count: the 32-lane case
is tolerable as a first approximation, the 8-lane case is already suspicious,
and the one-lane serial case is basically a science-fair exhibit wearing a
trench coat.

Decision:

- CCD-style charge buckets are not a replacement for ADC.
- They are worth keeping as a low-throughput/calibration/readout-multiplexing
  option.
- For the main cheap-fast inference path, prefer CMOS column-parallel or
  near-column charge-domain ADC unless a future device model gives much faster
  transfer and much lower read-noise accumulation.
