"""Energy and latency budget estimates for calibrated optical MLP-up inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import pandas as pd

from optical_spike.reporting import markdown_table


@dataclass(frozen=True)
class HardwareBudgetScenario:
    name: str
    optical_op_pj: float
    dac_per_input_pj: float
    adc_per_output_pj: float
    digital_correction_op_pj: float
    control_write_pj: float
    digital_mac_pj: float
    optical_compute_latency_ns: float
    dac_latency_ns: float
    adc_latency_ns: float
    correction_latency_ns: float
    digital_matmul_latency_ns: float
    laser_per_inference_pj: float = 0.0
    detector_tia_per_output_pj: float = 0.0
    static_control_per_inference_pj: float = 0.0


DEFAULT_HARDWARE_BUDGETS: tuple[HardwareBudgetScenario, ...] = (
    HardwareBudgetScenario(
        "published_273fj_low_io",
        optical_op_pj=0.273,
        dac_per_input_pj=0.5,
        adc_per_output_pj=1.0,
        digital_correction_op_pj=0.03,
        control_write_pj=5.0,
        digital_mac_pj=1.0,
        optical_compute_latency_ns=0.2,
        dac_latency_ns=0.5,
        adc_latency_ns=1.0,
        correction_latency_ns=0.2,
        digital_matmul_latency_ns=2.0,
        laser_per_inference_pj=100.0,
        detector_tia_per_output_pj=0.5,
        static_control_per_inference_pj=10.0,
    ),
    HardwareBudgetScenario(
        "photonic_tensor_core_5topsw_io",
        optical_op_pj=0.196,
        dac_per_input_pj=5.0,
        adc_per_output_pj=10.0,
        digital_correction_op_pj=0.1,
        control_write_pj=50.0,
        digital_mac_pj=1.0,
        optical_compute_latency_ns=0.2,
        dac_latency_ns=2.0,
        adc_latency_ns=5.0,
        correction_latency_ns=0.5,
        digital_matmul_latency_ns=2.0,
        laser_per_inference_pj=500.0,
        detector_tia_per_output_pj=2.0,
        static_control_per_inference_pj=50.0,
    ),
    HardwareBudgetScenario(
        "inp_nonvolatile_all_optical_estimate",
        optical_op_pj=0.7,
        dac_per_input_pj=0.5,
        adc_per_output_pj=1.0,
        digital_correction_op_pj=0.03,
        control_write_pj=5.0,
        digital_mac_pj=1.0,
        optical_compute_latency_ns=0.5,
        dac_latency_ns=0.5,
        adc_latency_ns=1.0,
        correction_latency_ns=0.2,
        digital_matmul_latency_ns=2.0,
        laser_per_inference_pj=100.0,
        detector_tia_per_output_pj=0.5,
        static_control_per_inference_pj=10.0,
    ),
    HardwareBudgetScenario(
        "inp_volatile_oeo_effective",
        optical_op_pj=8.0,
        dac_per_input_pj=0.0,
        adc_per_output_pj=0.0,
        digital_correction_op_pj=0.1,
        control_write_pj=50.0,
        digital_mac_pj=1.0,
        optical_compute_latency_ns=1.0,
        dac_latency_ns=2.0,
        adc_latency_ns=5.0,
        correction_latency_ns=0.5,
        digital_matmul_latency_ns=2.0,
    ),
    HardwareBudgetScenario(
        "high_precision_adc_bound",
        optical_op_pj=1.0,
        dac_per_input_pj=10.0,
        adc_per_output_pj=1000.0,
        digital_correction_op_pj=0.2,
        control_write_pj=100.0,
        digital_mac_pj=1.0,
        optical_compute_latency_ns=0.5,
        dac_latency_ns=5.0,
        adc_latency_ns=25.0,
        correction_latency_ns=1.0,
        digital_matmul_latency_ns=2.0,
        laser_per_inference_pj=500.0,
        detector_tia_per_output_pj=10.0,
        static_control_per_inference_pj=100.0,
    ),
)


@dataclass(frozen=True)
class EnergyBudgetConfig:
    calibration_summary: Path
    output_dir: Path
    input_dim: int = 64
    output_dim: int = 256
    inferences_per_drift_step: int = 10_000
    scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS


@dataclass(frozen=True)
class ChainedBlockBudgetConfig:
    calibration_summary: Path
    output_dir: Path
    input_dim: int = 64
    output_dim: int = 256
    inferences_per_drift_step: int = 10_000
    chain_lengths: tuple[int, ...] = (1, 2, 4, 8)
    scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS


@dataclass(frozen=True)
class ChargeReadoutScenario:
    name: str
    lanes: int
    integration_time_ns: float
    transfer_energy_pj: float
    transfer_latency_ns: float
    output_amp_per_sample_pj: float
    adc_energy_scale: float
    adc_latency_scale: float
    read_noise_drop_per_100_transfers: float


DEFAULT_CHARGE_READOUTS: tuple[ChargeReadoutScenario, ...] = (
    ChargeReadoutScenario(
        "cmos_column_parallel_reference",
        lanes=256,
        integration_time_ns=0.2,
        transfer_energy_pj=0.0,
        transfer_latency_ns=0.0,
        output_amp_per_sample_pj=0.0,
        adc_energy_scale=1.0,
        adc_latency_scale=1.0,
        read_noise_drop_per_100_transfers=0.0,
    ),
    ChargeReadoutScenario(
        "ccd_32_lane_fast_bucket",
        lanes=32,
        integration_time_ns=1.0,
        transfer_energy_pj=0.02,
        transfer_latency_ns=0.05,
        output_amp_per_sample_pj=0.2,
        adc_energy_scale=0.9,
        adc_latency_scale=1.1,
        read_noise_drop_per_100_transfers=0.0004,
    ),
    ChargeReadoutScenario(
        "ccd_8_lane_low_power_bucket",
        lanes=8,
        integration_time_ns=2.0,
        transfer_energy_pj=0.01,
        transfer_latency_ns=0.10,
        output_amp_per_sample_pj=0.1,
        adc_energy_scale=0.75,
        adc_latency_scale=1.2,
        read_noise_drop_per_100_transfers=0.0007,
    ),
    ChargeReadoutScenario(
        "ccd_1_lane_serial_stress",
        lanes=1,
        integration_time_ns=5.0,
        transfer_energy_pj=0.005,
        transfer_latency_ns=0.20,
        output_amp_per_sample_pj=0.05,
        adc_energy_scale=0.6,
        adc_latency_scale=1.5,
        read_noise_drop_per_100_transfers=0.0015,
    ),
)


@dataclass(frozen=True)
class ChargeReadoutBudgetConfig:
    calibration_summary: Path
    output_dir: Path
    input_dim: int = 64
    output_dim: int = 256
    inferences_per_drift_step: int = 10_000
    scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS
    readouts: tuple[ChargeReadoutScenario, ...] = DEFAULT_CHARGE_READOUTS


@dataclass(frozen=True)
class EnergyAccuracyParetoConfig:
    chained_budget_rows: Path
    shared_training_summary: Path
    unshared_training_summary: Path
    output_dir: Path
    material_scenario: str = "reverse_saturable_strength_0p2"


@dataclass(frozen=True)
class ChainStressProfile:
    name: str
    drift_drop_multiplier: float
    readout_noise_multiplier: float


DEFAULT_CHAIN_STRESS_PROFILES: tuple[ChainStressProfile, ...] = (
    ChainStressProfile("nominal_calibrated", 1.0, 1.0),
    ChainStressProfile("strong_drift_2x", 2.0, 1.0),
    ChainStressProfile("strong_drift_4x", 4.0, 1.0),
    ChainStressProfile("noisy_readout_2x", 1.0, 2.0),
    ChainStressProfile("combined_drift2x_readout2x", 2.0, 2.0),
    ChainStressProfile("combined_drift4x_readout2x", 4.0, 2.0),
)


@dataclass(frozen=True)
class ConverterPlacementScenario:
    name: str
    input_dac_boundaries: int
    output_adc_boundaries: int
    detector_readout_boundaries: int
    latency_dac_boundaries: int
    latency_adc_boundaries: int
    note: str


@dataclass(frozen=True)
class ConverterComponentConstraint:
    name: str
    pj_per_sample: float
    bandwidth_gsps: float | None
    enob: float | None
    source_chain: str
    source: str
    note: str
    sfdr_db: float | None = None
    inl_lsb: float | None = None
    dnl_lsb: float | None = None


@dataclass(frozen=True)
class ConverterCircuitProfile:
    name: str
    dac: ConverterComponentConstraint
    adc: ConverterComponentConstraint
    detector: ConverterComponentConstraint
    tia: ConverterComponentConstraint
    dac_latency_ns: float
    adc_latency_ns: float
    note: str


@dataclass(frozen=True)
class DacSourceCodingScenario:
    name: str
    source_bits: int
    energy_overhead_multiplier: float
    latency_overhead_multiplier: float
    accuracy_penalty_multiplier: float
    calibration_note: str


DEFAULT_CONVERTER_PLACEMENTS: tuple[ConverterPlacementScenario, ...] = (
    ConverterPlacementScenario(
        "shared_input_output_boundary",
        input_dac_boundaries=1,
        output_adc_boundaries=1,
        detector_readout_boundaries=1,
        latency_dac_boundaries=1,
        latency_adc_boundaries=1,
        note="one input DAC and one final ADC/TIA readout for the optical chain",
    ),
    ConverterPlacementScenario(
        "adc_after_each_block",
        input_dac_boundaries=1,
        output_adc_boundaries=4,
        detector_readout_boundaries=4,
        latency_dac_boundaries=1,
        latency_adc_boundaries=4,
        note="digital observation after every optical block, input DAC still shared",
    ),
    ConverterPlacementScenario(
        "full_oeo_between_blocks",
        input_dac_boundaries=4,
        output_adc_boundaries=4,
        detector_readout_boundaries=4,
        latency_dac_boundaries=4,
        latency_adc_boundaries=4,
        note="DAC/ADC/TIA at every block boundary, approximating repeated OEO hops",
    ),
)


DEFAULT_DAC_SOURCE_CODING_SCENARIOS: tuple[DacSourceCodingScenario, ...] = tuple(
    scenario
    for bits in (2, 3, 4, 5, 6)
    for scenario in (
        DacSourceCodingScenario(
            name="direct_low_resolution",
            source_bits=bits,
            energy_overhead_multiplier=1.0,
            latency_overhead_multiplier=1.0,
            accuracy_penalty_multiplier=1.0,
            calibration_note="direct lower-resolution drive; no special recovery assumed",
        ),
        DacSourceCodingScenario(
            name="calibrated_low_resolution",
            source_bits=bits,
            energy_overhead_multiplier=1.15,
            latency_overhead_multiplier=1.0,
            accuracy_penalty_multiplier=0.35,
            calibration_note="lower-resolution drive with hardware-aware calibration/adaptation",
        ),
        DacSourceCodingScenario(
            name="redundant_2phase_coding",
            source_bits=bits,
            energy_overhead_multiplier=2.0,
            latency_overhead_multiplier=2.0,
            accuracy_penalty_multiplier=0.2,
            calibration_note="two-phase redundant source coding; lower error but higher I/O work",
        ),
    )
)


DEFAULT_CONVERTER_CIRCUIT_PROFILES: tuple[ConverterCircuitProfile, ...] = (
    ConverterCircuitProfile(
        "inp_2022_10ghz_transceiver",
        dac=ConverterComponentConstraint(
            "inp_2022_system_dac",
            pj_per_sample=2.5,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="laser_modulator_driver_to_optical_input",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022); Wang et al. 3-GS/s 12-bit current-steering DAC (Electronics 2019)",
            note=(
                "25 mW DAC budget at 10 GHz, converted to per-sample energy; "
                "ENOB/linearity set as a transfer-model constraint, not proven by the InP system budget"
            ),
            sfdr_db=50.0,
            inl_lsb=1.22,
            dnl_lsb=1.21,
        ),
        adc=ConverterComponentConstraint(
            "inp_2022_system_adc",
            pj_per_sample=2.5,
            bandwidth_gsps=10.0,
            enob=None,
            source_chain="photodetector_receiver_to_digital_output",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="25 mW ADC budget at 10 GHz, converted to per-sample energy",
        ),
        detector=ConverterComponentConstraint(
            "inp_2022_receiver_pd",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="photonic_output_to_receiver_frontend",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="Half of the 5 mW receiver budget assigned to detector/front-end collection",
        ),
        tia=ConverterComponentConstraint(
            "inp_2022_receiver_tia",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="receiver_frontend_to_adc_input",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="Half of the 5 mW receiver budget assigned to TIA/readout conditioning",
        ),
        dac_latency_ns=0.1,
        adc_latency_ns=0.1,
        note="25 mW DAC/ADC and 5 mW receiver at 10 GHz, converted to per-sample energy",
    ),
    ConverterCircuitProfile(
        "kull_2018_8b_ti_sar_adc",
        dac=ConverterComponentConstraint(
            "inp_2022_system_dac",
            pj_per_sample=2.5,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="laser_modulator_driver_to_optical_input",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022); Wang et al. 3-GS/s 12-bit current-steering DAC (Electronics 2019)",
            note=(
                "System-level DAC energy anchor retained while swapping in a concrete ADC; "
                "DAC ENOB/linearity is now an explicit transfer-model constraint"
            ),
            sfdr_db=50.0,
            inl_lsb=1.22,
            dnl_lsb=1.21,
        ),
        adc=ConverterComponentConstraint(
            "kull_2018_8b_ti_sar_adc",
            pj_per_sample=3.3,
            bandwidth_gsps=72.0,
            enob=8.0,
            source_chain="time_interleaved_sar_adc_output_boundary",
            source="Kull et al. 24-72 GS/s 8-bit time-interleaved SAR ADC (JSSC 2018)",
            note="Uses the high end of the reported 2.0-3.3 pJ/conversion range",
        ),
        detector=ConverterComponentConstraint(
            "inp_2022_receiver_pd",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="photonic_output_to_receiver_frontend",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="Detector half of the system-level receiver anchor",
        ),
        tia=ConverterComponentConstraint(
            "inp_2022_receiver_tia",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="receiver_frontend_to_adc_input",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="TIA half of the system-level receiver anchor",
        ),
        dac_latency_ns=0.1,
        adc_latency_ns=0.1,
        note="Uses the reported 3.3 pJ/conversion high-speed ADC point with InP-style DAC/readout anchors",
    ),
    ConverterCircuitProfile(
        "per_block_adc_guardrail",
        dac=ConverterComponentConstraint(
            "target_dac_guardrail",
            pj_per_sample=2.5,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="shared_or_per_block_input_dac",
            source="Derived from converter target report plus DAC energy-bound and current-steering DAC literature",
            note=(
                "Keeps the literature-backed DAC target while testing per-block readout; "
                "requires high-speed DAC linearity at the same effective precision"
            ),
            sfdr_db=50.0,
            inl_lsb=1.22,
            dnl_lsb=1.21,
        ),
        adc=ConverterComponentConstraint(
            "per_block_adc_guardrail",
            pj_per_sample=5.0,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="per_block_adc_readout_boundary",
            source="Derived from converter target report plus high-speed ADC literature check",
            note="Practical per-block ADC ceiling before energy margin collapses",
        ),
        detector=ConverterComponentConstraint(
            "inp_2022_receiver_pd",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="photonic_output_to_receiver_frontend",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="Detector half of the system-level receiver anchor",
        ),
        tia=ConverterComponentConstraint(
            "inp_2022_receiver_tia",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="receiver_frontend_to_adc_input",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="TIA half of the system-level receiver anchor",
        ),
        dac_latency_ns=0.5,
        adc_latency_ns=1.0,
        note="Conservative per-block fallback ceiling; above this, repeated ADC/readout erodes the margin",
    ),
    ConverterCircuitProfile(
        "shared_boundary_target_ceiling",
        dac=ConverterComponentConstraint(
            "shared_boundary_dac_ceiling",
            pj_per_sample=2.5,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="single_input_boundary_dac",
            source="Derived shared-boundary converter target envelope plus DAC/modulator literature",
            note=(
                "Requirement ceiling, not one demonstrated DAC; 2-bit segmented EO-DAC "
                "modulators show the optical route but not the required precision"
            ),
            sfdr_db=50.0,
            inl_lsb=1.22,
            dnl_lsb=1.21,
        ),
        adc=ConverterComponentConstraint(
            "shared_boundary_adc_ceiling",
            pj_per_sample=25.0,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="single_output_boundary_adc",
            source="Derived shared-boundary converter target envelope",
            note="Requirement ceiling, deliberately looser than demonstrated high-speed ADC points",
        ),
        detector=ConverterComponentConstraint(
            "shared_boundary_detector_ceiling",
            pj_per_sample=5.0,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="single_output_boundary_detector",
            source="Derived shared-boundary converter target envelope",
            note="Detector allowance split out from the 12.5 pJ/output readout ceiling",
        ),
        tia=ConverterComponentConstraint(
            "shared_boundary_tia_ceiling",
            pj_per_sample=7.5,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="single_output_boundary_tia",
            source="Derived shared-boundary converter target envelope",
            note="TIA/readout allowance split out from the 12.5 pJ/output readout ceiling",
        ),
        dac_latency_ns=0.5,
        adc_latency_ns=1.0,
        note="Broadest passing shared-boundary envelope, not a single demonstrated circuit",
    ),
    ConverterCircuitProfile(
        "ad9172_12gsps_rf_dac_reference",
        dac=ConverterComponentConstraint(
            "adi_ad9172_dual_16b_rf_dac",
            pj_per_sample=125.0,
            bandwidth_gsps=12.0,
            enob=None,
            source_chain="direct_rf_dac_to_modulator_driver_reference",
            source="Analog Devices AD9172 data sheet and high-speed converter product brief",
            note=(
                "Closest single commercial DAC candidate found for rate and SFDR: "
                "12 GSPS, nominal 16 bit, 70 dBc SFDR at 1.8 GHz, but about "
                "1.5 W/channel or 125 pJ/sample at 12 GSPS; ENOB is not specified"
            ),
            sfdr_db=70.0,
            inl_lsb=7.0,
            dnl_lsb=7.0,
        ),
        adc=ConverterComponentConstraint(
            "kull_2018_8b_ti_sar_adc",
            pj_per_sample=3.3,
            bandwidth_gsps=72.0,
            enob=8.0,
            source_chain="time_interleaved_sar_adc_output_boundary",
            source="Kull et al. 24-72 GS/s 8-bit time-interleaved SAR ADC (JSSC 2018)",
            note="Uses the high end of the reported 2.0-3.3 pJ/conversion range",
        ),
        detector=ConverterComponentConstraint(
            "inp_2022_receiver_pd",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="photonic_output_to_receiver_frontend",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="Detector half of the system-level receiver anchor",
        ),
        tia=ConverterComponentConstraint(
            "inp_2022_receiver_tia",
            pj_per_sample=0.25,
            bandwidth_gsps=10.0,
            enob=7.35,
            source_chain="receiver_frontend_to_adc_input",
            source="InP photonic integrated multi-layer neural networks (APL Photonics 2022)",
            note="TIA half of the system-level receiver anchor",
        ),
        dac_latency_ns=0.1,
        adc_latency_ns=0.1,
        note=(
            "Failing commercial-reference profile: useful for bounding the DAC "
            "hole, not a passing low-energy source chain"
        ),
    ),
)


@dataclass(frozen=True)
class UnsharedFourBlockStressConfig:
    chained_budget_rows: Path
    unshared_training_summary: Path
    output_dir: Path
    material_scenario: str = "reverse_saturable_strength_0p2"
    chain_length: int = 4
    input_dim: int = 64
    output_dim: int = 256
    stress_profiles: tuple[ChainStressProfile, ...] = DEFAULT_CHAIN_STRESS_PROFILES
    hardware_scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS
    readouts: tuple[ChargeReadoutScenario, ...] = DEFAULT_CHARGE_READOUTS


@dataclass(frozen=True)
class ArchitectureEnergyOverlayConfig:
    pareto_rows: Path
    physical_rows: Path
    output_dir: Path
    candidate: str = "unshared_4_block"
    architecture_profile: str = "athermal_compensated_mrr"
    output_dim: int = 256
    chain_length: int = 4
    detector_energy_multiplier: float = 10.0
    output_readout_energy_multiplier: float = 10.0
    thermal_control_pj_per_k_per_block: float = 25.0
    resonance_tracking_pj_per_block: float = 5.0
    profile_latency_overhead_ns: float = 0.2


@dataclass(frozen=True)
class ArchitectureEnergyStressConfig:
    pareto_rows: Path
    physical_rows: Path
    output_dir: Path
    candidate: str = "unshared_4_block"
    architecture_profile: str = "athermal_compensated_mrr"
    output_dim: int = 256
    chain_length: int = 4
    detector_energy_multiplier: float = 10.0
    output_readout_energy_multiplier: float = 10.0
    thermal_control_pj_per_k_per_block_values: tuple[float, ...] = (
        0.0,
        25.0,
        100.0,
        250.0,
    )
    resonance_tracking_pj_per_block_values: tuple[float, ...] = (
        0.0,
        5.0,
        25.0,
        100.0,
    )
    profile_latency_overhead_ns: float = 0.2


@dataclass(frozen=True)
class ArchitectureConverterStressConfig:
    pareto_rows: Path
    physical_rows: Path
    output_dir: Path
    candidate: str = "unshared_4_block"
    architecture_profile: str = "athermal_compensated_mrr"
    input_dim: int = 64
    output_dim: int = 256
    chain_length: int = 4
    detector_energy_multiplier: float = 10.0
    output_readout_energy_multiplier: float = 10.0
    thermal_control_pj_per_k_per_block: float = 25.0
    resonance_tracking_pj_per_block: float = 5.0
    profile_latency_overhead_ns: float = 0.2
    adc_energy_multipliers: tuple[float, ...] = (0.5, 1.0, 5.0, 25.0)
    dac_energy_multipliers: tuple[float, ...] = (0.5, 1.0, 5.0)
    readout_energy_multipliers: tuple[float, ...] = (0.5, 1.0, 5.0, 25.0)
    placements: tuple[ConverterPlacementScenario, ...] = DEFAULT_CONVERTER_PLACEMENTS
    hardware_scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS


@dataclass(frozen=True)
class ArchitectureConverterTargetConfig:
    converter_stress_rows: Path
    output_dir: Path
    input_dim: int = 64
    output_dim: int = 256
    allowed_hardware_scenarios: tuple[str, ...] = (
        "published_273fj_low_io",
        "inp_nonvolatile_all_optical_estimate",
    )
    max_energy_ratio_vs_digital: float = 1.0
    max_latency_ratio_vs_digital: float = 1.0


@dataclass(frozen=True)
class ArchitectureConverterProfileConfig:
    pareto_rows: Path
    physical_rows: Path
    output_dir: Path
    candidate: str = "unshared_4_block"
    architecture_profile: str = "athermal_compensated_mrr"
    input_dim: int = 64
    output_dim: int = 256
    chain_length: int = 4
    detector_energy_multiplier: float = 10.0
    output_readout_energy_multiplier: float = 10.0
    thermal_control_pj_per_k_per_block: float = 25.0
    resonance_tracking_pj_per_block: float = 5.0
    profile_latency_overhead_ns: float = 0.2
    placements: tuple[ConverterPlacementScenario, ...] = DEFAULT_CONVERTER_PLACEMENTS
    converter_profiles: tuple[
        ConverterCircuitProfile, ...
    ] = DEFAULT_CONVERTER_CIRCUIT_PROFILES
    hardware_scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS


@dataclass(frozen=True)
class ArchitectureDacCodingSweepConfig:
    pareto_rows: Path
    physical_rows: Path
    output_dir: Path
    candidate: str = "unshared_4_block"
    architecture_profile: str = "athermal_compensated_mrr"
    reference_converter_profile: str = "kull_2018_8b_ti_sar_adc"
    target_dac_enob: float = 7.35
    target_dac_pj_per_sample: float = 2.5
    input_dim: int = 64
    output_dim: int = 256
    chain_length: int = 4
    detector_energy_multiplier: float = 10.0
    output_readout_energy_multiplier: float = 10.0
    thermal_control_pj_per_k_per_block: float = 25.0
    resonance_tracking_pj_per_block: float = 5.0
    profile_latency_overhead_ns: float = 0.2
    min_accuracy_percent: float = 85.0
    placements: tuple[ConverterPlacementScenario, ...] = DEFAULT_CONVERTER_PLACEMENTS
    coding_scenarios: tuple[
        DacSourceCodingScenario, ...
    ] = DEFAULT_DAC_SOURCE_CODING_SCENARIOS
    converter_profiles: tuple[
        ConverterCircuitProfile, ...
    ] = DEFAULT_CONVERTER_CIRCUIT_PROFILES
    hardware_scenarios: tuple[HardwareBudgetScenario, ...] = DEFAULT_HARDWARE_BUDGETS


def estimate_budget_rows(config: EnergyBudgetConfig) -> pd.DataFrame:
    calibration = pd.read_csv(config.calibration_summary)
    optical_ops = config.input_dim * config.output_dim
    correction_ops = 2 * config.output_dim
    rows: list[dict[str, float | int | str]] = []
    for calibration_row in calibration.to_dict(orient="records"):
        calibration_examples_per_step = float(
            calibration_row["calibration_examples_per_step_mean"]
        )
        affine_values_per_step = float(calibration_row["affine_values_per_step_mean"])
        for scenario in config.scenarios:
            optical_compute_energy = optical_ops * scenario.optical_op_pj
            dac_energy = config.input_dim * scenario.dac_per_input_pj
            adc_energy = config.output_dim * scenario.adc_per_output_pj
            detector_tia_energy = config.output_dim * scenario.detector_tia_per_output_pj
            correction_energy = correction_ops * scenario.digital_correction_op_pj
            control_energy_per_step = affine_values_per_step * scenario.control_write_pj
            control_energy_per_inference = control_energy_per_step / config.inferences_per_drift_step

            base_inference_energy = (
                optical_compute_energy
                + dac_energy
                + adc_energy
                + detector_tia_energy
                + scenario.laser_per_inference_pj
                + scenario.static_control_per_inference_pj
                + correction_energy
            )
            calibration_energy_per_example = base_inference_energy
            calibration_energy_per_step = (
                calibration_examples_per_step * calibration_energy_per_example
            )
            calibration_energy_per_inference = (
                calibration_energy_per_step / config.inferences_per_drift_step
            )
            total_energy = (
                base_inference_energy
                + control_energy_per_inference
                + calibration_energy_per_inference
            )
            digital_energy = optical_ops * scenario.digital_mac_pj
            latency = (
                scenario.optical_compute_latency_ns
                + scenario.dac_latency_ns
                + scenario.adc_latency_ns
                + scenario.correction_latency_ns
            )

            rows.append(
                {
                    "hardware_scenario": scenario.name,
                    "calibration_samples": int(calibration_row["calibration_samples"]),
                    "recalibration_interval_steps": int(
                        calibration_row["recalibration_interval_steps"]
                    ),
                    "drift_rate": float(calibration_row["drift_rate"]),
                    "accuracy_drop_mean": float(
                        calibration_row["calibrated_accuracy_drop_mean"]
                    ),
                    "accuracy_drop_max_mean": float(
                        calibration_row["calibrated_accuracy_drop_max_mean"]
                    ),
                    "optical_compute_energy_pj": optical_compute_energy,
                    "dac_energy_pj": dac_energy,
                    "adc_energy_pj": adc_energy,
                    "detector_tia_energy_pj": detector_tia_energy,
                    "laser_energy_pj": scenario.laser_per_inference_pj,
                    "digital_correction_energy_pj": correction_energy,
                    "static_control_energy_pj": scenario.static_control_per_inference_pj,
                    "control_energy_per_inference_pj": control_energy_per_inference,
                    "calibration_energy_per_inference_pj": calibration_energy_per_inference,
                    "total_energy_per_inference_pj": total_energy,
                    "digital_mlp_up_energy_pj": digital_energy,
                    "energy_ratio_vs_digital": total_energy / digital_energy,
                    "estimated_latency_ns": latency,
                    "digital_mlp_up_latency_ns": scenario.digital_matmul_latency_ns,
                    "latency_ratio_vs_digital": latency / scenario.digital_matmul_latency_ns,
                    "inferences_per_drift_step": config.inferences_per_drift_step,
                }
            )
    return pd.DataFrame(rows)


def charge_transfer_count(output_dim: int, lanes: int) -> int:
    lane_count = max(1, min(lanes, output_dim))
    base = output_dim // lane_count
    remainder = output_dim % lane_count
    total = 0
    for lane_index in range(lane_count):
        lane_outputs = base + (1 if lane_index < remainder else 0)
        total += lane_outputs * (lane_outputs + 1) // 2
    return total


def estimate_charge_readout_rows(config: ChargeReadoutBudgetConfig) -> pd.DataFrame:
    calibration = pd.read_csv(config.calibration_summary)
    optical_ops = config.input_dim * config.output_dim
    correction_ops = 2 * config.output_dim
    rows: list[dict[str, float | int | str]] = []
    for calibration_row in calibration.to_dict(orient="records"):
        calibration_examples_per_step = float(
            calibration_row["calibration_examples_per_step_mean"]
        )
        affine_values_per_step = float(calibration_row["affine_values_per_step_mean"])
        calibrated_accuracy_drop = float(calibration_row["calibrated_accuracy_drop_mean"])
        for scenario in config.scenarios:
            optical_compute_energy = optical_ops * scenario.optical_op_pj
            dac_energy = config.input_dim * scenario.dac_per_input_pj
            correction_energy = correction_ops * scenario.digital_correction_op_pj
            control_energy_per_step = affine_values_per_step * scenario.control_write_pj
            control_energy_per_inference = (
                control_energy_per_step / config.inferences_per_drift_step
            )
            digital_energy = optical_ops * scenario.digital_mac_pj
            digital_latency = scenario.digital_matmul_latency_ns
            for readout in config.readouts:
                lanes = max(1, min(readout.lanes, config.output_dim))
                serial_groups = math.ceil(config.output_dim / lanes)
                transfer_count = charge_transfer_count(config.output_dim, lanes)
                transfer_energy = transfer_count * readout.transfer_energy_pj
                adc_energy = (
                    config.output_dim
                    * scenario.adc_per_output_pj
                    * readout.adc_energy_scale
                )
                output_amp_energy = config.output_dim * readout.output_amp_per_sample_pj
                detector_tia_energy = (
                    config.output_dim * scenario.detector_tia_per_output_pj
                    if readout.transfer_energy_pj == 0.0
                    else output_amp_energy
                )
                readout_energy = adc_energy + detector_tia_energy + transfer_energy
                base_inference_energy = (
                    optical_compute_energy
                    + dac_energy
                    + readout_energy
                    + scenario.laser_per_inference_pj
                    + scenario.static_control_per_inference_pj
                    + correction_energy
                )
                calibration_energy_per_step = (
                    calibration_examples_per_step * base_inference_energy
                )
                calibration_energy_per_inference = (
                    calibration_energy_per_step / config.inferences_per_drift_step
                )
                total_energy = (
                    base_inference_energy
                    + control_energy_per_inference
                    + calibration_energy_per_inference
                )
                readout_latency = (
                    readout.integration_time_ns
                    + transfer_count * readout.transfer_latency_ns
                    + serial_groups * scenario.adc_latency_ns * readout.adc_latency_scale
                )
                total_latency = (
                    scenario.optical_compute_latency_ns
                    + scenario.dac_latency_ns
                    + readout_latency
                    + scenario.correction_latency_ns
                )
                read_noise_drop = (
                    transfer_count / 100.0 * readout.read_noise_drop_per_100_transfers
                )
                rows.append(
                    {
                        "hardware_scenario": scenario.name,
                        "readout_scenario": readout.name,
                        "lanes": lanes,
                        "serial_groups": serial_groups,
                        "charge_transfer_count": transfer_count,
                        "calibration_samples": int(calibration_row["calibration_samples"]),
                        "recalibration_interval_steps": int(
                            calibration_row["recalibration_interval_steps"]
                        ),
                        "drift_rate": float(calibration_row["drift_rate"]),
                        "single_block_accuracy_drop_mean": calibrated_accuracy_drop,
                        "read_noise_accuracy_drop_proxy": read_noise_drop,
                        "total_accuracy_drop_proxy": calibrated_accuracy_drop + read_noise_drop,
                        "optical_compute_energy_pj": optical_compute_energy,
                        "dac_energy_pj": dac_energy,
                        "adc_energy_pj": adc_energy,
                        "detector_or_output_amp_energy_pj": detector_tia_energy,
                        "charge_transfer_energy_pj": transfer_energy,
                        "readout_energy_pj": readout_energy,
                        "laser_energy_pj": scenario.laser_per_inference_pj,
                        "digital_correction_energy_pj": correction_energy,
                        "static_control_energy_pj": scenario.static_control_per_inference_pj,
                        "control_energy_per_inference_pj": control_energy_per_inference,
                        "calibration_energy_per_inference_pj": calibration_energy_per_inference,
                        "total_energy_per_inference_pj": total_energy,
                        "digital_mlp_up_energy_pj": digital_energy,
                        "energy_ratio_vs_digital": total_energy / digital_energy,
                        "readout_latency_ns": readout_latency,
                        "estimated_latency_ns": total_latency,
                        "digital_mlp_up_latency_ns": digital_latency,
                        "latency_ratio_vs_digital": total_latency / digital_latency,
                        "inferences_per_drift_step": config.inferences_per_drift_step,
                    }
                )
    return pd.DataFrame(rows)


def estimate_chained_block_rows(config: ChainedBlockBudgetConfig) -> pd.DataFrame:
    calibration = pd.read_csv(config.calibration_summary)
    optical_ops_per_block = config.input_dim * config.output_dim
    correction_ops = 2 * config.output_dim
    rows: list[dict[str, float | int | str]] = []
    for calibration_row in calibration.to_dict(orient="records"):
        calibration_examples_per_step = float(
            calibration_row["calibration_examples_per_step_mean"]
        )
        affine_values_per_step = float(calibration_row["affine_values_per_step_mean"])
        calibrated_accuracy_drop = float(calibration_row["calibrated_accuracy_drop_mean"])
        for scenario in config.scenarios:
            single_block_optical_compute_energy = (
                optical_ops_per_block * scenario.optical_op_pj
            )
            single_block_dac_energy = config.input_dim * scenario.dac_per_input_pj
            single_block_adc_energy = config.output_dim * scenario.adc_per_output_pj
            single_block_detector_tia_energy = (
                config.output_dim * scenario.detector_tia_per_output_pj
            )
            single_block_correction_energy = correction_ops * scenario.digital_correction_op_pj
            single_block_energy = (
                single_block_optical_compute_energy
                + single_block_dac_energy
                + single_block_adc_energy
                + single_block_detector_tia_energy
                + scenario.laser_per_inference_pj
                + scenario.static_control_per_inference_pj
                + single_block_correction_energy
            )
            for chain_length in config.chain_lengths:
                optical_compute_energy = (
                    chain_length * single_block_optical_compute_energy
                )
                converter_energy = single_block_dac_energy + single_block_adc_energy
                detector_tia_energy = single_block_detector_tia_energy
                correction_energy = single_block_correction_energy
                chained_base_energy = (
                    optical_compute_energy
                    + converter_energy
                    + detector_tia_energy
                    + scenario.laser_per_inference_pj
                    + scenario.static_control_per_inference_pj
                    + correction_energy
                )
                isolated_optical_energy = chain_length * single_block_energy
                control_energy_per_step = (
                    chain_length * affine_values_per_step * scenario.control_write_pj
                )
                control_energy_per_inference = (
                    control_energy_per_step / config.inferences_per_drift_step
                )
                calibration_energy_per_step = (
                    calibration_examples_per_step * chained_base_energy
                )
                calibration_energy_per_inference = (
                    calibration_energy_per_step / config.inferences_per_drift_step
                )
                total_energy = (
                    chained_base_energy
                    + control_energy_per_inference
                    + calibration_energy_per_inference
                )
                digital_energy = (
                    chain_length * optical_ops_per_block * scenario.digital_mac_pj
                )
                chained_latency = (
                    scenario.dac_latency_ns
                    + chain_length * scenario.optical_compute_latency_ns
                    + scenario.adc_latency_ns
                    + scenario.correction_latency_ns
                )
                isolated_optical_latency = chain_length * (
                    scenario.optical_compute_latency_ns
                    + scenario.dac_latency_ns
                    + scenario.adc_latency_ns
                    + scenario.correction_latency_ns
                )
                digital_latency = chain_length * scenario.digital_matmul_latency_ns
                converter_energy_saved = (
                    chain_length - 1
                ) * (
                    single_block_dac_energy
                    + single_block_adc_energy
                    + single_block_detector_tia_energy
                    + single_block_correction_energy
                    + scenario.static_control_per_inference_pj
                )
                converter_latency_saved = isolated_optical_latency - chained_latency
                rows.append(
                    {
                        "hardware_scenario": scenario.name,
                        "chain_length": chain_length,
                        "calibration_samples": int(calibration_row["calibration_samples"]),
                        "recalibration_interval_steps": int(
                            calibration_row["recalibration_interval_steps"]
                        ),
                        "drift_rate": float(calibration_row["drift_rate"]),
                        "single_block_accuracy_drop_mean": calibrated_accuracy_drop,
                        "chain_error_growth_proxy": calibrated_accuracy_drop
                        * (chain_length ** 0.5),
                        "optical_compute_energy_pj": optical_compute_energy,
                        "converter_energy_pj": converter_energy,
                        "detector_tia_energy_pj": detector_tia_energy,
                        "laser_energy_pj": scenario.laser_per_inference_pj,
                        "digital_correction_energy_pj": correction_energy,
                        "static_control_energy_pj": scenario.static_control_per_inference_pj,
                        "control_energy_per_inference_pj": control_energy_per_inference,
                        "calibration_energy_per_inference_pj": calibration_energy_per_inference,
                        "total_energy_per_chain_pj": total_energy,
                        "energy_per_block_pj": total_energy / chain_length,
                        "digital_chain_energy_pj": digital_energy,
                        "energy_ratio_vs_digital_chain": total_energy / digital_energy,
                        "isolated_optical_energy_pj": isolated_optical_energy,
                        "energy_ratio_vs_isolated_optical": total_energy
                        / isolated_optical_energy,
                        "converter_energy_saved_pj": converter_energy_saved,
                        "estimated_latency_ns": chained_latency,
                        "latency_per_block_ns": chained_latency / chain_length,
                        "digital_chain_latency_ns": digital_latency,
                        "latency_ratio_vs_digital_chain": chained_latency / digital_latency,
                        "isolated_optical_latency_ns": isolated_optical_latency,
                        "latency_ratio_vs_isolated_optical": chained_latency
                        / isolated_optical_latency,
                        "converter_latency_saved_ns": converter_latency_saved,
                        "inferences_per_drift_step": config.inferences_per_drift_step,
                    }
                )
    return pd.DataFrame(rows)


def summarize_budget_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["hardware_scenario", "drift_rate"], dropna=False)
    summary = grouped.agg(
        best_energy_ratio_vs_digital=("energy_ratio_vs_digital", "min"),
        median_energy_ratio_vs_digital=("energy_ratio_vs_digital", "median"),
        best_total_energy_pj=("total_energy_per_inference_pj", "min"),
        median_total_energy_pj=("total_energy_per_inference_pj", "median"),
        best_accuracy_drop_mean=("accuracy_drop_mean", "min"),
        worst_accuracy_drop_max_mean=("accuracy_drop_max_mean", "max"),
        latency_ratio_vs_digital=("latency_ratio_vs_digital", "median"),
        rows=("hardware_scenario", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def summarize_charge_readout_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["hardware_scenario", "readout_scenario"], dropna=False)
    summary = grouped.agg(
        lanes=("lanes", "median"),
        median_charge_transfer_count=("charge_transfer_count", "median"),
        best_energy_ratio_vs_digital=("energy_ratio_vs_digital", "min"),
        median_energy_ratio_vs_digital=("energy_ratio_vs_digital", "median"),
        best_latency_ratio_vs_digital=("latency_ratio_vs_digital", "min"),
        median_latency_ratio_vs_digital=("latency_ratio_vs_digital", "median"),
        best_readout_energy_pj=("readout_energy_pj", "min"),
        median_readout_latency_ns=("readout_latency_ns", "median"),
        median_total_accuracy_drop_proxy=("total_accuracy_drop_proxy", "median"),
        rows=("hardware_scenario", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def summarize_chained_block_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["hardware_scenario", "chain_length"], dropna=False)
    summary = grouped.agg(
        best_energy_ratio_vs_digital_chain=("energy_ratio_vs_digital_chain", "min"),
        median_energy_ratio_vs_digital_chain=("energy_ratio_vs_digital_chain", "median"),
        best_latency_ratio_vs_digital_chain=("latency_ratio_vs_digital_chain", "min"),
        median_latency_ratio_vs_digital_chain=("latency_ratio_vs_digital_chain", "median"),
        best_energy_ratio_vs_isolated_optical=("energy_ratio_vs_isolated_optical", "min"),
        median_energy_ratio_vs_isolated_optical=("energy_ratio_vs_isolated_optical", "median"),
        best_latency_ratio_vs_isolated_optical=("latency_ratio_vs_isolated_optical", "min"),
        median_latency_ratio_vs_isolated_optical=("latency_ratio_vs_isolated_optical", "median"),
        best_energy_per_block_pj=("energy_per_block_pj", "min"),
        best_latency_per_block_ns=("latency_per_block_ns", "min"),
        median_chain_error_growth_proxy=("chain_error_growth_proxy", "median"),
        rows=("hardware_scenario", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def _load_chain_accuracy_rows(config: EnergyAccuracyParetoConfig) -> pd.DataFrame:
    shared = pd.read_csv(config.shared_training_summary)
    shared = shared[shared["scenario"] == config.material_scenario].copy()
    shared["weight_sharing"] = "shared"

    unshared = pd.read_csv(config.unshared_training_summary)
    unshared = unshared[unshared["scenario"] == config.material_scenario].copy()
    unshared["weight_sharing"] = "unshared"

    accuracy = pd.concat([shared, unshared], ignore_index=True)
    return accuracy[
        [
            "weight_sharing",
            "scenario",
            "material_family",
            "kind",
            "strength",
            "chain_length",
            "trained_chain_accuracy_mean",
            "raw_constrained_accuracy_mean",
            "calibrated_constrained_accuracy_mean",
            "trained_delta_vs_relu_mean",
            "raw_drop_vs_trained_mean",
            "calibrated_drop_vs_trained_mean",
        ]
    ]


def _best_budget_rows(chained_budget_rows: pd.DataFrame) -> pd.DataFrame:
    ordered = chained_budget_rows.sort_values(
        [
            "hardware_scenario",
            "chain_length",
            "energy_ratio_vs_digital_chain",
            "latency_ratio_vs_digital_chain",
        ]
    )
    return ordered.groupby(["hardware_scenario", "chain_length"], as_index=False).first()


def _mark_pareto_efficient(rows: pd.DataFrame) -> pd.Series:
    efficient: list[bool] = []
    for _, candidate in rows.iterrows():
        dominated = (
            (
                rows["calibrated_constrained_accuracy_mean"]
                >= candidate["calibrated_constrained_accuracy_mean"]
            )
            & (rows["energy_ratio_vs_digital_chain"] <= candidate["energy_ratio_vs_digital_chain"])
            & (rows["latency_ratio_vs_digital_chain"] <= candidate["latency_ratio_vs_digital_chain"])
            & (
                (
                    rows["calibrated_constrained_accuracy_mean"]
                    > candidate["calibrated_constrained_accuracy_mean"]
                )
                | (
                    rows["energy_ratio_vs_digital_chain"]
                    < candidate["energy_ratio_vs_digital_chain"]
                )
                | (
                    rows["latency_ratio_vs_digital_chain"]
                    < candidate["latency_ratio_vs_digital_chain"]
                )
            )
        ).any()
        efficient.append(not bool(dominated))
    return pd.Series(efficient, index=rows.index)


def build_energy_accuracy_pareto_rows(config: EnergyAccuracyParetoConfig) -> pd.DataFrame:
    accuracy = _load_chain_accuracy_rows(config)
    budget = _best_budget_rows(pd.read_csv(config.chained_budget_rows))
    rows = accuracy.merge(budget, on="chain_length", how="inner")
    rows["candidate"] = rows["weight_sharing"] + "_" + rows["chain_length"].astype(str) + "_block"
    rows["readout_assumption"] = "single shared CMOS-like output boundary"
    rows["converter_energy_fraction"] = (
        rows["converter_energy_pj"] + rows["detector_tia_energy_pj"]
    ) / rows["total_energy_per_chain_pj"]
    rows["accuracy_percent"] = rows["calibrated_constrained_accuracy_mean"] * 100.0
    rows["energy_saving_vs_digital_percent"] = (
        1.0 - rows["energy_ratio_vs_digital_chain"]
    ) * 100.0
    rows["latency_saving_vs_digital_percent"] = (
        1.0 - rows["latency_ratio_vs_digital_chain"]
    ) * 100.0
    rows["pareto_efficient"] = _mark_pareto_efficient(rows)
    return rows[
        [
            "candidate",
            "weight_sharing",
            "chain_length",
            "hardware_scenario",
            "readout_assumption",
            "accuracy_percent",
            "calibrated_constrained_accuracy_mean",
            "trained_chain_accuracy_mean",
            "trained_delta_vs_relu_mean",
            "calibrated_drop_vs_trained_mean",
            "total_energy_per_chain_pj",
            "energy_ratio_vs_digital_chain",
            "energy_saving_vs_digital_percent",
            "converter_energy_pj",
            "detector_tia_energy_pj",
            "converter_energy_fraction",
            "estimated_latency_ns",
            "latency_ratio_vs_digital_chain",
            "latency_saving_vs_digital_percent",
            "pareto_efficient",
            "calibration_samples",
            "recalibration_interval_steps",
            "drift_rate",
        ]
    ].sort_values(
        [
            "pareto_efficient",
            "calibrated_constrained_accuracy_mean",
            "energy_ratio_vs_digital_chain",
        ],
        ascending=[False, False, True],
    )


def summarize_energy_accuracy_pareto_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["candidate", "chain_length", "weight_sharing"], dropna=False)
    summary = grouped.agg(
        calibrated_accuracy_percent=("accuracy_percent", "median"),
        best_energy_ratio_vs_digital_chain=("energy_ratio_vs_digital_chain", "min"),
        best_latency_ratio_vs_digital_chain=("latency_ratio_vs_digital_chain", "min"),
        best_energy_saving_percent=("energy_saving_vs_digital_percent", "max"),
        best_latency_saving_percent=("latency_saving_vs_digital_percent", "max"),
        median_converter_energy_fraction=("converter_energy_fraction", "median"),
        pareto_efficient_rows=("pareto_efficient", "sum"),
        rows=("candidate", "count"),
    ).reset_index()
    return summary.sort_values(
        [
            "pareto_efficient_rows",
            "calibrated_accuracy_percent",
            "best_energy_ratio_vs_digital_chain",
        ],
        ascending=[False, False, True],
    )


def _photon_energy_fj(wavelength_nm: float = 1550.0) -> float:
    planck_j_s = 6.62607015e-34
    speed_of_light_m_s = 299_792_458
    energy_j = planck_j_s * speed_of_light_m_s / (wavelength_nm * 1e-9)
    return energy_j * 1e15


def _physical_scenario_summary(physical_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        physical_rows.groupby(
            ["physical_scenario", "architecture_profile"],
            dropna=False,
        )
        .agg(
            accuracy_percent=("calibrated_accuracy", lambda values: values.mean() * 100.0),
            min_accuracy_percent=("calibrated_accuracy", lambda values: values.min() * 100.0),
            projection_mse_mean=("projection_mse_mean", "mean"),
            steps_below_85_percent=("below_85_percent", "sum"),
            thermal_drift_rate=("thermal_drift_rate", "first"),
            detector_noise=("detector_noise", "first"),
            output_readout_noise=("output_readout_noise", "first"),
            recalibration_interval_steps=("recalibration_interval_steps", "first"),
            resonance_shift_pm_per_step=("resonance_shift_pm_per_step", "first"),
            thermal_excursion_k_per_step=("thermal_excursion_k_per_step", "first"),
            detector_photons_per_sample=("detector_photons_per_sample", "first"),
            output_photons_per_sample=("output_photons_per_sample", "first"),
            output_enob_equivalent=("output_enob_equivalent", "first"),
            rows=("seed", "count"),
        )
        .reset_index()
    )


def build_architecture_energy_overlay_rows(
    config: ArchitectureEnergyOverlayConfig,
) -> pd.DataFrame:
    pareto = pd.read_csv(config.pareto_rows)
    pareto = pareto[pareto["candidate"] == config.candidate].copy()
    physical = _physical_scenario_summary(pd.read_csv(config.physical_rows))
    physical = physical[
        physical["architecture_profile"] == config.architecture_profile
    ].copy()
    if pareto.empty:
        raise ValueError(f"candidate {config.candidate!r} not found in {config.pareto_rows}")
    if physical.empty:
        raise ValueError(
            f"architecture profile {config.architecture_profile!r} not found in "
            f"{config.physical_rows}"
        )

    photon_energy_fj = _photon_energy_fj()
    rows: list[dict[str, object]] = []
    for physical_row in physical.to_dict(orient="records"):
        detector_energy_floor_fj = (
            float(physical_row["detector_photons_per_sample"]) * photon_energy_fj
        )
        output_energy_floor_fj = (
            float(physical_row["output_photons_per_sample"]) * photon_energy_fj
        )
        detector_energy_with_overhead_pj = (
            detector_energy_floor_fj
            * config.detector_energy_multiplier
            * config.output_dim
            * config.chain_length
            / 1000.0
        )
        output_energy_with_overhead_pj = (
            output_energy_floor_fj
            * config.output_readout_energy_multiplier
            * config.output_dim
            / 1000.0
        )
        thermal_control_energy_pj = (
            float(physical_row["thermal_excursion_k_per_step"])
            * config.thermal_control_pj_per_k_per_block
            * config.chain_length
        )
        resonance_tracking_energy_pj = (
            config.resonance_tracking_pj_per_block * config.chain_length
        )
        profile_energy_overhead_pj = (
            detector_energy_with_overhead_pj
            + output_energy_with_overhead_pj
            + thermal_control_energy_pj
            + resonance_tracking_energy_pj
        )

        for budget_row in pareto.to_dict(orient="records"):
            base_energy_pj = float(budget_row["total_energy_per_chain_pj"])
            digital_energy_pj = base_energy_pj / float(
                budget_row["energy_ratio_vs_digital_chain"]
            )
            total_energy_pj = base_energy_pj + profile_energy_overhead_pj
            estimated_latency_ns = (
                float(budget_row["estimated_latency_ns"])
                + config.profile_latency_overhead_ns
            )
            digital_latency_ns = float(budget_row["estimated_latency_ns"]) / float(
                budget_row["latency_ratio_vs_digital_chain"]
            )
            energy_ratio = total_energy_pj / digital_energy_pj
            latency_ratio = estimated_latency_ns / digital_latency_ns
            passes_85_percent_floor = (
                float(physical_row["min_accuracy_percent"]) >= 85.0
                and int(physical_row["steps_below_85_percent"]) == 0
            )
            energy_below_digital = energy_ratio < 1.0
            latency_below_digital = latency_ratio < 1.0
            rows.append(
                {
                    "candidate": config.candidate,
                    "architecture_profile": config.architecture_profile,
                    "physical_scenario": physical_row["physical_scenario"],
                    "hardware_scenario": budget_row["hardware_scenario"],
                    "accuracy_percent": physical_row["accuracy_percent"],
                    "min_accuracy_percent": physical_row["min_accuracy_percent"],
                    "projection_mse_mean": physical_row["projection_mse_mean"],
                    "steps_below_85_percent": int(
                        physical_row["steps_below_85_percent"]
                    ),
                    "base_total_energy_pj": base_energy_pj,
                    "profile_detector_energy_pj": detector_energy_with_overhead_pj,
                    "profile_output_readout_energy_pj": output_energy_with_overhead_pj,
                    "profile_thermal_control_energy_pj": thermal_control_energy_pj,
                    "profile_resonance_tracking_energy_pj": resonance_tracking_energy_pj,
                    "profile_energy_overhead_pj": profile_energy_overhead_pj,
                    "profile_total_energy_pj": total_energy_pj,
                    "profile_energy_ratio_vs_digital": energy_ratio,
                    "base_energy_ratio_vs_digital": budget_row[
                        "energy_ratio_vs_digital_chain"
                    ],
                    "profile_estimated_latency_ns": estimated_latency_ns,
                    "profile_latency_ratio_vs_digital": latency_ratio,
                    "base_latency_ratio_vs_digital": budget_row[
                        "latency_ratio_vs_digital_chain"
                    ],
                    "resonance_shift_pm_per_step": physical_row[
                        "resonance_shift_pm_per_step"
                    ],
                    "thermal_excursion_k_per_step": physical_row[
                        "thermal_excursion_k_per_step"
                    ],
                    "detector_photons_per_sample": physical_row[
                        "detector_photons_per_sample"
                    ],
                    "output_photons_per_sample": physical_row[
                        "output_photons_per_sample"
                    ],
                    "output_enob_equivalent": physical_row["output_enob_equivalent"],
                    "passes_85_percent_floor": passes_85_percent_floor,
                    "energy_below_digital": energy_below_digital,
                    "latency_below_digital": latency_below_digital,
                    "passes_all_gates": (
                        passes_85_percent_floor
                        and energy_below_digital
                        and latency_below_digital
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        [
            "passes_85_percent_floor",
            "energy_below_digital",
            "latency_below_digital",
            "accuracy_percent",
            "profile_energy_ratio_vs_digital",
        ],
        ascending=[False, False, False, False, True],
    )


def summarize_architecture_energy_overlay_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["architecture_profile", "physical_scenario"], dropna=False)
        .agg(
            mean_accuracy_percent=("accuracy_percent", "median"),
            min_accuracy_percent=("min_accuracy_percent", "median"),
            best_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "min"),
            best_latency_ratio_vs_digital=("profile_latency_ratio_vs_digital", "min"),
            median_profile_energy_overhead_pj=("profile_energy_overhead_pj", "median"),
            median_detector_energy_pj=("profile_detector_energy_pj", "median"),
            median_output_readout_energy_pj=(
                "profile_output_readout_energy_pj",
                "median",
            ),
            median_thermal_control_energy_pj=(
                "profile_thermal_control_energy_pj",
                "median",
            ),
            hardware_cases_passing_all=("passes_all_gates", "sum"),
            rows=("candidate", "count"),
        )
        .reset_index()
        .sort_values(
            ["min_accuracy_percent", "best_energy_ratio_vs_digital"],
            ascending=[False, True],
        )
    )


def run_architecture_energy_overlay_report(
    config: ArchitectureEnergyOverlayConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_architecture_energy_overlay_rows(config)
    summary = summarize_architecture_energy_overlay_rows(rows)
    rows_csv = config.output_dir / "architecture_energy_overlay_rows.csv"
    summary_csv = config.output_dir / "architecture_energy_overlay_summary.csv"
    summary_md = config.output_dir / "architecture_energy_overlay_summary.md"
    manifest_path = config.output_dir / "architecture_energy_overlay_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    passing_all = (
        rows["passes_85_percent_floor"]
        & rows["energy_below_digital"]
        & rows["latency_below_digital"]
    )
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "passing_all_rows": int(passing_all.sum()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_architecture_energy_stress_rows(
    config: ArchitectureEnergyStressConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for thermal_control in config.thermal_control_pj_per_k_per_block_values:
        for resonance_tracking in config.resonance_tracking_pj_per_block_values:
            rows = build_architecture_energy_overlay_rows(
                ArchitectureEnergyOverlayConfig(
                    pareto_rows=config.pareto_rows,
                    physical_rows=config.physical_rows,
                    output_dir=config.output_dir,
                    candidate=config.candidate,
                    architecture_profile=config.architecture_profile,
                    output_dim=config.output_dim,
                    chain_length=config.chain_length,
                    detector_energy_multiplier=config.detector_energy_multiplier,
                    output_readout_energy_multiplier=(
                        config.output_readout_energy_multiplier
                    ),
                    thermal_control_pj_per_k_per_block=thermal_control,
                    resonance_tracking_pj_per_block=resonance_tracking,
                    profile_latency_overhead_ns=config.profile_latency_overhead_ns,
                )
            )
            rows["thermal_control_pj_per_k_per_block"] = thermal_control
            rows["resonance_tracking_pj_per_block"] = resonance_tracking
            frames.append(rows)
    return pd.concat(frames, ignore_index=True).sort_values(
        [
            "passes_all_gates",
            "physical_scenario",
            "thermal_control_pj_per_k_per_block",
            "resonance_tracking_pj_per_block",
            "profile_energy_ratio_vs_digital",
        ],
        ascending=[False, True, True, True, True],
    )


def summarize_architecture_energy_stress_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(
            [
                "thermal_control_pj_per_k_per_block",
                "resonance_tracking_pj_per_block",
            ],
            dropna=False,
        )
        .agg(
            passing_all_rows=("passes_all_gates", "sum"),
            passing_accuracy_rows=("passes_85_percent_floor", "sum"),
            energy_below_digital_rows=("energy_below_digital", "sum"),
            latency_below_digital_rows=("latency_below_digital", "sum"),
            best_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "min"),
            median_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "median"),
            worst_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "max"),
            median_profile_energy_overhead_pj=("profile_energy_overhead_pj", "median"),
            rows=("candidate", "count"),
        )
        .reset_index()
    )
    return summary.sort_values(
        [
            "passing_all_rows",
            "best_energy_ratio_vs_digital",
            "median_profile_energy_overhead_pj",
        ],
        ascending=[False, True, True],
    )


def run_architecture_energy_stress_report(
    config: ArchitectureEnergyStressConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_architecture_energy_stress_rows(config)
    summary = summarize_architecture_energy_stress_rows(rows)
    rows_csv = config.output_dir / "architecture_energy_stress_rows.csv"
    summary_csv = config.output_dir / "architecture_energy_stress_summary.csv"
    summary_md = config.output_dir / "architecture_energy_stress_summary.md"
    manifest_path = config.output_dir / "architecture_energy_stress_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "max_passing_all_rows": int(summary["passing_all_rows"].max()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_architecture_converter_stress_rows(
    config: ArchitectureConverterStressConfig,
) -> pd.DataFrame:
    pareto = pd.read_csv(config.pareto_rows)
    pareto = pareto[pareto["candidate"] == config.candidate].copy()
    physical = _physical_scenario_summary(pd.read_csv(config.physical_rows))
    physical = physical[
        physical["architecture_profile"] == config.architecture_profile
    ].copy()
    if pareto.empty:
        raise ValueError(f"candidate {config.candidate!r} not found in {config.pareto_rows}")
    if physical.empty:
        raise ValueError(
            f"architecture profile {config.architecture_profile!r} not found in "
            f"{config.physical_rows}"
        )

    scenario_by_name = {scenario.name: scenario for scenario in config.hardware_scenarios}
    photon_energy_fj = _photon_energy_fj()
    rows: list[dict[str, object]] = []
    for physical_row in physical.to_dict(orient="records"):
        detector_energy_floor_fj = (
            float(physical_row["detector_photons_per_sample"]) * photon_energy_fj
        )
        output_energy_floor_fj = (
            float(physical_row["output_photons_per_sample"]) * photon_energy_fj
        )
        profile_detector_energy_pj = (
            detector_energy_floor_fj
            * config.detector_energy_multiplier
            * config.output_dim
            * config.chain_length
            / 1000.0
        )
        profile_output_readout_energy_pj = (
            output_energy_floor_fj
            * config.output_readout_energy_multiplier
            * config.output_dim
            / 1000.0
        )
        thermal_control_energy_pj = (
            float(physical_row["thermal_excursion_k_per_step"])
            * config.thermal_control_pj_per_k_per_block
            * config.chain_length
        )
        resonance_tracking_energy_pj = (
            config.resonance_tracking_pj_per_block * config.chain_length
        )
        profile_energy_overhead_pj = (
            profile_detector_energy_pj
            + profile_output_readout_energy_pj
            + thermal_control_energy_pj
            + resonance_tracking_energy_pj
        )
        passes_85_percent_floor = (
            float(physical_row["min_accuracy_percent"]) >= 85.0
            and int(physical_row["steps_below_85_percent"]) == 0
        )

        for budget_row in pareto.to_dict(orient="records"):
            hardware_name = str(budget_row["hardware_scenario"])
            if hardware_name not in scenario_by_name:
                continue
            scenario = scenario_by_name[hardware_name]
            base_energy_pj = float(budget_row["total_energy_per_chain_pj"])
            digital_energy_pj = base_energy_pj / float(
                budget_row["energy_ratio_vs_digital_chain"]
            )
            base_without_converter_readout_pj = (
                base_energy_pj
                - float(budget_row["converter_energy_pj"])
                - float(budget_row["detector_tia_energy_pj"])
            )
            base_latency_ns = float(budget_row["estimated_latency_ns"])
            digital_latency_ns = base_latency_ns / float(
                budget_row["latency_ratio_vs_digital_chain"]
            )
            latency_without_converter_ns = (
                base_latency_ns - scenario.dac_latency_ns - scenario.adc_latency_ns
            )

            for placement in config.placements:
                for adc_multiplier in config.adc_energy_multipliers:
                    for dac_multiplier in config.dac_energy_multipliers:
                        for readout_multiplier in config.readout_energy_multipliers:
                            stressed_dac_energy_pj = (
                                config.input_dim
                                * scenario.dac_per_input_pj
                                * placement.input_dac_boundaries
                                * dac_multiplier
                            )
                            stressed_adc_energy_pj = (
                                config.output_dim
                                * scenario.adc_per_output_pj
                                * placement.output_adc_boundaries
                                * adc_multiplier
                            )
                            stressed_readout_energy_pj = (
                                config.output_dim
                                * scenario.detector_tia_per_output_pj
                                * placement.detector_readout_boundaries
                                * readout_multiplier
                            )
                            stressed_converter_readout_pj = (
                                stressed_dac_energy_pj
                                + stressed_adc_energy_pj
                                + stressed_readout_energy_pj
                            )
                            total_energy_pj = (
                                base_without_converter_readout_pj
                                + stressed_converter_readout_pj
                                + profile_energy_overhead_pj
                            )
                            estimated_latency_ns = (
                                latency_without_converter_ns
                                + placement.latency_dac_boundaries * scenario.dac_latency_ns
                                + placement.latency_adc_boundaries * scenario.adc_latency_ns
                                + config.profile_latency_overhead_ns
                            )
                            energy_ratio = total_energy_pj / digital_energy_pj
                            latency_ratio = estimated_latency_ns / digital_latency_ns
                            energy_below_digital = energy_ratio < 1.0
                            latency_below_digital = latency_ratio < 1.0
                            rows.append(
                                {
                                    "candidate": config.candidate,
                                    "architecture_profile": config.architecture_profile,
                                    "physical_scenario": physical_row[
                                        "physical_scenario"
                                    ],
                                    "hardware_scenario": hardware_name,
                                    "converter_placement": placement.name,
                                    "converter_placement_note": placement.note,
                                    "adc_energy_multiplier": adc_multiplier,
                                    "dac_energy_multiplier": dac_multiplier,
                                    "readout_energy_multiplier": readout_multiplier,
                                    "input_dac_boundaries": placement.input_dac_boundaries,
                                    "output_adc_boundaries": placement.output_adc_boundaries,
                                    "detector_readout_boundaries": (
                                        placement.detector_readout_boundaries
                                    ),
                                    "base_without_converter_readout_pj": (
                                        base_without_converter_readout_pj
                                    ),
                                    "stressed_dac_energy_pj": stressed_dac_energy_pj,
                                    "stressed_adc_energy_pj": stressed_adc_energy_pj,
                                    "stressed_readout_energy_pj": (
                                        stressed_readout_energy_pj
                                    ),
                                    "stressed_converter_readout_pj": (
                                        stressed_converter_readout_pj
                                    ),
                                    "profile_energy_overhead_pj": (
                                        profile_energy_overhead_pj
                                    ),
                                    "profile_total_energy_pj": total_energy_pj,
                                    "profile_energy_ratio_vs_digital": energy_ratio,
                                    "profile_estimated_latency_ns": (
                                        estimated_latency_ns
                                    ),
                                    "profile_latency_ratio_vs_digital": latency_ratio,
                                    "accuracy_percent": physical_row["accuracy_percent"],
                                    "min_accuracy_percent": physical_row[
                                        "min_accuracy_percent"
                                    ],
                                    "projection_mse_mean": physical_row[
                                        "projection_mse_mean"
                                    ],
                                    "passes_85_percent_floor": (
                                        passes_85_percent_floor
                                    ),
                                    "energy_below_digital": energy_below_digital,
                                    "latency_below_digital": latency_below_digital,
                                    "passes_all_gates": (
                                        passes_85_percent_floor
                                        and energy_below_digital
                                        and latency_below_digital
                                    ),
                                }
                            )
    return pd.DataFrame(rows).sort_values(
        [
            "passes_all_gates",
            "converter_placement",
            "adc_energy_multiplier",
            "dac_energy_multiplier",
            "readout_energy_multiplier",
            "profile_energy_ratio_vs_digital",
        ],
        ascending=[False, True, True, True, True, True],
    )


def summarize_architecture_converter_stress_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(
            [
                "converter_placement",
                "adc_energy_multiplier",
                "dac_energy_multiplier",
                "readout_energy_multiplier",
            ],
            dropna=False,
        )
        .agg(
            passing_all_rows=("passes_all_gates", "sum"),
            passing_accuracy_rows=("passes_85_percent_floor", "sum"),
            energy_below_digital_rows=("energy_below_digital", "sum"),
            latency_below_digital_rows=("latency_below_digital", "sum"),
            best_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "min"),
            median_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "median"),
            worst_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "max"),
            best_latency_ratio_vs_digital=("profile_latency_ratio_vs_digital", "min"),
            median_converter_readout_pj=("stressed_converter_readout_pj", "median"),
            rows=("candidate", "count"),
        )
        .reset_index()
    )
    return summary.sort_values(
        [
            "passing_all_rows",
            "best_energy_ratio_vs_digital",
            "median_converter_readout_pj",
        ],
        ascending=[False, True, True],
    )


def run_architecture_converter_stress_report(
    config: ArchitectureConverterStressConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_architecture_converter_stress_rows(config)
    summary = summarize_architecture_converter_stress_rows(rows)
    rows_csv = config.output_dir / "architecture_converter_stress_rows.csv"
    summary_csv = config.output_dir / "architecture_converter_stress_summary.csv"
    summary_md = config.output_dir / "architecture_converter_stress_summary.md"
    manifest_path = config.output_dir / "architecture_converter_stress_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "max_passing_all_rows": int(summary["passing_all_rows"].max()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_architecture_converter_target_rows(
    config: ArchitectureConverterTargetConfig,
) -> pd.DataFrame:
    rows = pd.read_csv(config.converter_stress_rows)
    passing = rows[
        (rows["passes_all_gates"])
        & (rows["profile_energy_ratio_vs_digital"] <= config.max_energy_ratio_vs_digital)
        & (rows["profile_latency_ratio_vs_digital"] <= config.max_latency_ratio_vs_digital)
        & (rows["hardware_scenario"].isin(config.allowed_hardware_scenarios))
    ].copy()
    if passing.empty:
        raise ValueError("no passing converter stress rows match the target criteria")

    passing["dac_pj_per_input"] = passing["stressed_dac_energy_pj"] / (
        passing["input_dac_boundaries"] * config.input_dim
    )
    passing["adc_pj_per_output"] = passing["stressed_adc_energy_pj"] / (
        passing["output_adc_boundaries"] * config.output_dim
    )
    passing["readout_pj_per_output"] = passing["stressed_readout_energy_pj"] / (
        passing["detector_readout_boundaries"] * config.output_dim
    )

    grouped = passing.groupby(["hardware_scenario", "converter_placement"], dropna=False)
    target_rows: list[dict[str, object]] = []
    for (hardware_scenario, placement), group in grouped:
        worst_energy = group.sort_values(
            ["profile_energy_ratio_vs_digital", "profile_latency_ratio_vs_digital"],
            ascending=[False, False],
        ).iloc[0]
        max_converter = group.sort_values(
            [
                "stressed_converter_readout_pj",
                "profile_energy_ratio_vs_digital",
            ],
            ascending=[False, False],
        ).iloc[0]
        target_rows.append(
            {
                "hardware_scenario": hardware_scenario,
                "converter_placement": placement,
                "passing_rows": int(len(group)),
                "physical_scenarios_passing": int(group["physical_scenario"].nunique()),
                "max_dac_pj_per_input": float(group["dac_pj_per_input"].max()),
                "max_adc_pj_per_output": float(group["adc_pj_per_output"].max()),
                "max_readout_tia_pj_per_output": float(
                    group["readout_pj_per_output"].max()
                ),
                "max_total_converter_readout_pj": float(
                    group["stressed_converter_readout_pj"].max()
                ),
                "best_energy_ratio_vs_digital": float(
                    group["profile_energy_ratio_vs_digital"].min()
                ),
                "worst_passing_energy_ratio_vs_digital": float(
                    worst_energy["profile_energy_ratio_vs_digital"]
                ),
                "best_latency_ratio_vs_digital": float(
                    group["profile_latency_ratio_vs_digital"].min()
                ),
                "worst_passing_latency_ratio_vs_digital": float(
                    worst_energy["profile_latency_ratio_vs_digital"]
                ),
                "widest_converter_corner_adc_multiplier": float(
                    max_converter["adc_energy_multiplier"]
                ),
                "widest_converter_corner_dac_multiplier": float(
                    max_converter["dac_energy_multiplier"]
                ),
                "widest_converter_corner_readout_multiplier": float(
                    max_converter["readout_energy_multiplier"]
                ),
            }
        )
    return pd.DataFrame(target_rows).sort_values(
        [
            "worst_passing_latency_ratio_vs_digital",
            "worst_passing_energy_ratio_vs_digital",
            "max_total_converter_readout_pj",
        ],
        ascending=[True, True, False],
    )


def run_architecture_converter_target_report(
    config: ArchitectureConverterTargetConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_architecture_converter_target_rows(config)
    rows_csv = config.output_dir / "architecture_converter_target_rows.csv"
    rows_md = config.output_dir / "architecture_converter_target_rows.md"
    manifest_path = config.output_dir / "architecture_converter_target_manifest.json"
    rows.to_csv(rows_csv, index=False)
    rows_md.write_text(markdown_table(rows), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "max_dac_pj_per_input": float(rows["max_dac_pj_per_input"].max()),
        "max_adc_pj_per_output": float(rows["max_adc_pj_per_output"].max()),
        "max_readout_tia_pj_per_output": float(
            rows["max_readout_tia_pj_per_output"].max()
        ),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "rows_md": str(rows_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_architecture_converter_profile_rows(
    config: ArchitectureConverterProfileConfig,
) -> pd.DataFrame:
    pareto = pd.read_csv(config.pareto_rows)
    pareto = pareto[pareto["candidate"] == config.candidate].copy()
    physical = _physical_scenario_summary(pd.read_csv(config.physical_rows))
    physical = physical[
        physical["architecture_profile"] == config.architecture_profile
    ].copy()
    if pareto.empty:
        raise ValueError(f"candidate {config.candidate!r} not found in {config.pareto_rows}")
    if physical.empty:
        raise ValueError(
            f"architecture profile {config.architecture_profile!r} not found in "
            f"{config.physical_rows}"
        )

    scenario_by_name = {scenario.name: scenario for scenario in config.hardware_scenarios}
    photon_energy_fj = _photon_energy_fj()
    rows: list[dict[str, object]] = []
    for physical_row in physical.to_dict(orient="records"):
        detector_energy_floor_fj = (
            float(physical_row["detector_photons_per_sample"]) * photon_energy_fj
        )
        output_energy_floor_fj = (
            float(physical_row["output_photons_per_sample"]) * photon_energy_fj
        )
        profile_detector_energy_pj = (
            detector_energy_floor_fj
            * config.detector_energy_multiplier
            * config.output_dim
            * config.chain_length
            / 1000.0
        )
        profile_output_readout_energy_pj = (
            output_energy_floor_fj
            * config.output_readout_energy_multiplier
            * config.output_dim
            / 1000.0
        )
        thermal_control_energy_pj = (
            float(physical_row["thermal_excursion_k_per_step"])
            * config.thermal_control_pj_per_k_per_block
            * config.chain_length
        )
        resonance_tracking_energy_pj = (
            config.resonance_tracking_pj_per_block * config.chain_length
        )
        profile_energy_overhead_pj = (
            profile_detector_energy_pj
            + profile_output_readout_energy_pj
            + thermal_control_energy_pj
            + resonance_tracking_energy_pj
        )
        passes_85_percent_floor = (
            float(physical_row["min_accuracy_percent"]) >= 85.0
            and int(physical_row["steps_below_85_percent"]) == 0
        )

        for budget_row in pareto.to_dict(orient="records"):
            hardware_name = str(budget_row["hardware_scenario"])
            if hardware_name not in scenario_by_name:
                continue
            base_energy_pj = float(budget_row["total_energy_per_chain_pj"])
            digital_energy_pj = base_energy_pj / float(
                budget_row["energy_ratio_vs_digital_chain"]
            )
            base_without_converter_readout_pj = (
                base_energy_pj
                - float(budget_row["converter_energy_pj"])
                - float(budget_row["detector_tia_energy_pj"])
            )
            base_latency_ns = float(budget_row["estimated_latency_ns"])
            digital_latency_ns = base_latency_ns / float(
                budget_row["latency_ratio_vs_digital_chain"]
            )
            scenario = scenario_by_name[hardware_name]
            latency_without_converter_ns = (
                base_latency_ns - scenario.dac_latency_ns - scenario.adc_latency_ns
            )

            for placement in config.placements:
                for converter_profile in config.converter_profiles:
                    dac_meets_source_chain_target = (
                        converter_profile.dac.pj_per_sample <= 2.5
                        and converter_profile.dac.bandwidth_gsps is not None
                        and converter_profile.dac.bandwidth_gsps >= 10.0
                        and converter_profile.dac.enob is not None
                        and converter_profile.dac.enob >= 7.35
                        and converter_profile.dac.sfdr_db is not None
                        and converter_profile.dac.sfdr_db >= 50.0
                    )
                    profiled_dac_energy_pj = (
                        config.input_dim
                        * converter_profile.dac.pj_per_sample
                        * placement.input_dac_boundaries
                    )
                    profiled_adc_energy_pj = (
                        config.output_dim
                        * converter_profile.adc.pj_per_sample
                        * placement.output_adc_boundaries
                    )
                    profiled_readout_energy_pj = (
                        config.output_dim
                        * (
                            converter_profile.detector.pj_per_sample
                            + converter_profile.tia.pj_per_sample
                        )
                        * placement.detector_readout_boundaries
                    )
                    profiled_converter_readout_pj = (
                        profiled_dac_energy_pj
                        + profiled_adc_energy_pj
                        + profiled_readout_energy_pj
                    )
                    total_energy_pj = (
                        base_without_converter_readout_pj
                        + profiled_converter_readout_pj
                        + profile_energy_overhead_pj
                    )
                    estimated_latency_ns = (
                        latency_without_converter_ns
                        + placement.latency_dac_boundaries
                        * converter_profile.dac_latency_ns
                        + placement.latency_adc_boundaries
                        * converter_profile.adc_latency_ns
                        + config.profile_latency_overhead_ns
                    )
                    energy_ratio = total_energy_pj / digital_energy_pj
                    latency_ratio = estimated_latency_ns / digital_latency_ns
                    rows.append(
                        {
                            "candidate": config.candidate,
                            "architecture_profile": config.architecture_profile,
                            "physical_scenario": physical_row["physical_scenario"],
                            "hardware_scenario": hardware_name,
                            "converter_placement": placement.name,
                            "converter_placement_note": placement.note,
                            "converter_profile": converter_profile.name,
                            "converter_profile_note": converter_profile.note,
                            "dac_component": converter_profile.dac.name,
                            "dac_pj_per_input": converter_profile.dac.pj_per_sample,
                            "dac_bandwidth_gsps": (
                                converter_profile.dac.bandwidth_gsps
                            ),
                            "dac_enob": converter_profile.dac.enob,
                            "dac_sfdr_db": converter_profile.dac.sfdr_db,
                            "dac_inl_lsb": converter_profile.dac.inl_lsb,
                            "dac_dnl_lsb": converter_profile.dac.dnl_lsb,
                            "dac_meets_source_chain_target": (
                                dac_meets_source_chain_target
                            ),
                            "dac_source_chain": converter_profile.dac.source_chain,
                            "dac_source": converter_profile.dac.source,
                            "dac_note": converter_profile.dac.note,
                            "adc_component": converter_profile.adc.name,
                            "adc_pj_per_output": converter_profile.adc.pj_per_sample,
                            "adc_bandwidth_gsps": (
                                converter_profile.adc.bandwidth_gsps
                            ),
                            "adc_enob": converter_profile.adc.enob,
                            "adc_sfdr_db": converter_profile.adc.sfdr_db,
                            "adc_inl_lsb": converter_profile.adc.inl_lsb,
                            "adc_dnl_lsb": converter_profile.adc.dnl_lsb,
                            "adc_source_chain": converter_profile.adc.source_chain,
                            "adc_source": converter_profile.adc.source,
                            "adc_note": converter_profile.adc.note,
                            "detector_component": converter_profile.detector.name,
                            "detector_pj_per_output": (
                                converter_profile.detector.pj_per_sample
                            ),
                            "detector_bandwidth_gsps": (
                                converter_profile.detector.bandwidth_gsps
                            ),
                            "detector_enob": converter_profile.detector.enob,
                            "detector_sfdr_db": converter_profile.detector.sfdr_db,
                            "detector_inl_lsb": converter_profile.detector.inl_lsb,
                            "detector_dnl_lsb": converter_profile.detector.dnl_lsb,
                            "detector_source_chain": (
                                converter_profile.detector.source_chain
                            ),
                            "detector_source": converter_profile.detector.source,
                            "detector_note": converter_profile.detector.note,
                            "tia_component": converter_profile.tia.name,
                            "tia_pj_per_output": converter_profile.tia.pj_per_sample,
                            "tia_bandwidth_gsps": (
                                converter_profile.tia.bandwidth_gsps
                            ),
                            "tia_enob": converter_profile.tia.enob,
                            "tia_sfdr_db": converter_profile.tia.sfdr_db,
                            "tia_inl_lsb": converter_profile.tia.inl_lsb,
                            "tia_dnl_lsb": converter_profile.tia.dnl_lsb,
                            "tia_source_chain": converter_profile.tia.source_chain,
                            "tia_source": converter_profile.tia.source,
                            "tia_note": converter_profile.tia.note,
                            "readout_tia_pj_per_output": (
                                converter_profile.detector.pj_per_sample
                                + converter_profile.tia.pj_per_sample
                            ),
                            "dac_latency_ns": converter_profile.dac_latency_ns,
                            "adc_latency_ns": converter_profile.adc_latency_ns,
                            "input_dac_boundaries": placement.input_dac_boundaries,
                            "output_adc_boundaries": placement.output_adc_boundaries,
                            "detector_readout_boundaries": (
                                placement.detector_readout_boundaries
                            ),
                            "profiled_dac_energy_pj": profiled_dac_energy_pj,
                            "profiled_adc_energy_pj": profiled_adc_energy_pj,
                            "profiled_readout_energy_pj": profiled_readout_energy_pj,
                            "profiled_converter_readout_pj": (
                                profiled_converter_readout_pj
                            ),
                            "profile_energy_overhead_pj": profile_energy_overhead_pj,
                            "profile_total_energy_pj": total_energy_pj,
                            "profile_energy_ratio_vs_digital": energy_ratio,
                            "profile_estimated_latency_ns": estimated_latency_ns,
                            "profile_latency_ratio_vs_digital": latency_ratio,
                            "accuracy_percent": physical_row["accuracy_percent"],
                            "min_accuracy_percent": physical_row["min_accuracy_percent"],
                            "projection_mse_mean": physical_row["projection_mse_mean"],
                            "passes_85_percent_floor": passes_85_percent_floor,
                            "energy_below_digital": energy_ratio < 1.0,
                            "latency_below_digital": latency_ratio < 1.0,
                            "passes_all_gates": (
                                passes_85_percent_floor
                                and energy_ratio < 1.0
                                and latency_ratio < 1.0
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        [
            "passes_all_gates",
            "converter_placement",
            "converter_profile",
            "profile_energy_ratio_vs_digital",
        ],
        ascending=[False, True, True, True],
    )


def summarize_architecture_converter_profile_rows(rows: pd.DataFrame) -> pd.DataFrame:
    summary = (
        rows.groupby(["converter_placement", "converter_profile"], dropna=False)
        .agg(
            passing_all_rows=("passes_all_gates", "sum"),
            passing_accuracy_rows=("passes_85_percent_floor", "sum"),
            energy_below_digital_rows=("energy_below_digital", "sum"),
            latency_below_digital_rows=("latency_below_digital", "sum"),
            best_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "min"),
            median_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "median"),
            worst_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "max"),
            best_latency_ratio_vs_digital=("profile_latency_ratio_vs_digital", "min"),
            median_converter_readout_pj=("profiled_converter_readout_pj", "median"),
            dac_pj_per_input=("dac_pj_per_input", "first"),
            dac_bandwidth_gsps=("dac_bandwidth_gsps", "first"),
            dac_enob=("dac_enob", "first"),
            dac_sfdr_db=("dac_sfdr_db", "first"),
            dac_inl_lsb=("dac_inl_lsb", "first"),
            dac_dnl_lsb=("dac_dnl_lsb", "first"),
            dac_target_rows=("dac_meets_source_chain_target", "sum"),
            dac_meets_source_chain_target=(
                "dac_meets_source_chain_target",
                "first",
            ),
            dac_source_chain=("dac_source_chain", "first"),
            adc_pj_per_output=("adc_pj_per_output", "first"),
            adc_bandwidth_gsps=("adc_bandwidth_gsps", "first"),
            adc_enob=("adc_enob", "first"),
            adc_sfdr_db=("adc_sfdr_db", "first"),
            adc_source_chain=("adc_source_chain", "first"),
            detector_pj_per_output=("detector_pj_per_output", "first"),
            detector_bandwidth_gsps=("detector_bandwidth_gsps", "first"),
            detector_enob=("detector_enob", "first"),
            detector_sfdr_db=("detector_sfdr_db", "first"),
            detector_source_chain=("detector_source_chain", "first"),
            tia_pj_per_output=("tia_pj_per_output", "first"),
            tia_bandwidth_gsps=("tia_bandwidth_gsps", "first"),
            tia_enob=("tia_enob", "first"),
            tia_sfdr_db=("tia_sfdr_db", "first"),
            tia_source_chain=("tia_source_chain", "first"),
            readout_tia_pj_per_output=("readout_tia_pj_per_output", "first"),
            rows=("candidate", "count"),
        )
        .reset_index()
    )
    return summary.sort_values(
        [
            "passing_all_rows",
            "best_energy_ratio_vs_digital",
            "median_converter_readout_pj",
        ],
        ascending=[False, True, True],
    )


def run_architecture_converter_profile_report(
    config: ArchitectureConverterProfileConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_architecture_converter_profile_rows(config)
    summary = summarize_architecture_converter_profile_rows(rows)
    rows_csv = config.output_dir / "architecture_converter_profile_rows.csv"
    summary_csv = config.output_dir / "architecture_converter_profile_summary.csv"
    summary_md = config.output_dir / "architecture_converter_profile_summary.md"
    manifest_path = config.output_dir / "architecture_converter_profile_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "max_passing_all_rows": int(summary["passing_all_rows"].max()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _converter_profile_by_name(
    profiles: tuple[ConverterCircuitProfile, ...],
    name: str,
) -> ConverterCircuitProfile:
    for profile in profiles:
        if profile.name == name:
            return profile
    raise ValueError(f"converter profile {name!r} not found")


def _modeled_dac_accuracy_penalty_percent(
    coding: DacSourceCodingScenario,
    target_enob: float,
) -> float:
    bit_gap = max(0.0, target_enob - float(coding.source_bits))
    return 0.35 * (bit_gap**1.7) * coding.accuracy_penalty_multiplier


def build_architecture_dac_coding_sweep_rows(
    config: ArchitectureDacCodingSweepConfig,
) -> pd.DataFrame:
    pareto = pd.read_csv(config.pareto_rows)
    pareto = pareto[pareto["candidate"] == config.candidate].copy()
    physical = _physical_scenario_summary(pd.read_csv(config.physical_rows))
    physical = physical[
        physical["architecture_profile"] == config.architecture_profile
    ].copy()
    if pareto.empty:
        raise ValueError(f"candidate {config.candidate!r} not found in {config.pareto_rows}")
    if physical.empty:
        raise ValueError(
            f"architecture profile {config.architecture_profile!r} not found in "
            f"{config.physical_rows}"
        )

    reference = _converter_profile_by_name(
        config.converter_profiles,
        config.reference_converter_profile,
    )
    scenario_by_name = {scenario.name: scenario for scenario in config.hardware_scenarios}
    photon_energy_fj = _photon_energy_fj()
    rows: list[dict[str, object]] = []

    for physical_row in physical.to_dict(orient="records"):
        detector_energy_floor_fj = (
            float(physical_row["detector_photons_per_sample"]) * photon_energy_fj
        )
        output_energy_floor_fj = (
            float(physical_row["output_photons_per_sample"]) * photon_energy_fj
        )
        profile_detector_energy_pj = (
            detector_energy_floor_fj
            * config.detector_energy_multiplier
            * config.output_dim
            * config.chain_length
            / 1000.0
        )
        profile_output_readout_energy_pj = (
            output_energy_floor_fj
            * config.output_readout_energy_multiplier
            * config.output_dim
            / 1000.0
        )
        thermal_control_energy_pj = (
            float(physical_row["thermal_excursion_k_per_step"])
            * config.thermal_control_pj_per_k_per_block
            * config.chain_length
        )
        resonance_tracking_energy_pj = (
            config.resonance_tracking_pj_per_block * config.chain_length
        )
        profile_energy_overhead_pj = (
            profile_detector_energy_pj
            + profile_output_readout_energy_pj
            + thermal_control_energy_pj
            + resonance_tracking_energy_pj
        )
        base_accuracy_percent = float(physical_row["accuracy_percent"])
        base_min_accuracy_percent = float(physical_row["min_accuracy_percent"])

        for budget_row in pareto.to_dict(orient="records"):
            hardware_name = str(budget_row["hardware_scenario"])
            if hardware_name not in scenario_by_name:
                continue

            base_energy_pj = float(budget_row["total_energy_per_chain_pj"])
            digital_energy_pj = base_energy_pj / float(
                budget_row["energy_ratio_vs_digital_chain"]
            )
            base_without_converter_readout_pj = (
                base_energy_pj
                - float(budget_row["converter_energy_pj"])
                - float(budget_row["detector_tia_energy_pj"])
            )
            base_latency_ns = float(budget_row["estimated_latency_ns"])
            digital_latency_ns = base_latency_ns / float(
                budget_row["latency_ratio_vs_digital_chain"]
            )
            hardware = scenario_by_name[hardware_name]
            latency_without_converter_ns = (
                base_latency_ns - hardware.dac_latency_ns - hardware.adc_latency_ns
            )

            for placement in config.placements:
                for coding in config.coding_scenarios:
                    dac_pj_per_input = (
                        config.target_dac_pj_per_sample
                        * (float(coding.source_bits) / config.target_dac_enob)
                        * coding.energy_overhead_multiplier
                    )
                    modeled_accuracy_penalty_percent = (
                        _modeled_dac_accuracy_penalty_percent(
                            coding,
                            config.target_dac_enob,
                        )
                    )
                    effective_accuracy_percent = (
                        base_accuracy_percent - modeled_accuracy_penalty_percent
                    )
                    effective_min_accuracy_percent = (
                        base_min_accuracy_percent - modeled_accuracy_penalty_percent
                    )
                    profiled_dac_energy_pj = (
                        config.input_dim
                        * dac_pj_per_input
                        * placement.input_dac_boundaries
                    )
                    profiled_adc_energy_pj = (
                        config.output_dim
                        * reference.adc.pj_per_sample
                        * placement.output_adc_boundaries
                    )
                    profiled_readout_energy_pj = (
                        config.output_dim
                        * (
                            reference.detector.pj_per_sample
                            + reference.tia.pj_per_sample
                        )
                        * placement.detector_readout_boundaries
                    )
                    profiled_converter_readout_pj = (
                        profiled_dac_energy_pj
                        + profiled_adc_energy_pj
                        + profiled_readout_energy_pj
                    )
                    total_energy_pj = (
                        base_without_converter_readout_pj
                        + profiled_converter_readout_pj
                        + profile_energy_overhead_pj
                    )
                    estimated_latency_ns = (
                        latency_without_converter_ns
                        + placement.latency_dac_boundaries
                        * reference.dac_latency_ns
                        * coding.latency_overhead_multiplier
                        + placement.latency_adc_boundaries
                        * reference.adc_latency_ns
                        + config.profile_latency_overhead_ns
                    )
                    energy_ratio = total_energy_pj / digital_energy_pj
                    latency_ratio = estimated_latency_ns / digital_latency_ns
                    passes_accuracy_floor = (
                        effective_min_accuracy_percent >= config.min_accuracy_percent
                        and int(physical_row["steps_below_85_percent"]) == 0
                    )
                    rows.append(
                        {
                            "candidate": config.candidate,
                            "architecture_profile": config.architecture_profile,
                            "physical_scenario": physical_row["physical_scenario"],
                            "hardware_scenario": hardware_name,
                            "converter_placement": placement.name,
                            "source_coding": coding.name,
                            "source_bits": coding.source_bits,
                            "target_dac_enob": config.target_dac_enob,
                            "dac_pj_per_input": dac_pj_per_input,
                            "dac_energy_overhead_multiplier": (
                                coding.energy_overhead_multiplier
                            ),
                            "dac_latency_overhead_multiplier": (
                                coding.latency_overhead_multiplier
                            ),
                            "accuracy_penalty_multiplier": (
                                coding.accuracy_penalty_multiplier
                            ),
                            "calibration_note": coding.calibration_note,
                            "reference_converter_profile": reference.name,
                            "adc_pj_per_output": reference.adc.pj_per_sample,
                            "readout_tia_pj_per_output": (
                                reference.detector.pj_per_sample
                                + reference.tia.pj_per_sample
                            ),
                            "profiled_dac_energy_pj": profiled_dac_energy_pj,
                            "profiled_adc_energy_pj": profiled_adc_energy_pj,
                            "profiled_readout_energy_pj": profiled_readout_energy_pj,
                            "profiled_converter_readout_pj": (
                                profiled_converter_readout_pj
                            ),
                            "profile_energy_overhead_pj": profile_energy_overhead_pj,
                            "profile_total_energy_pj": total_energy_pj,
                            "profile_energy_ratio_vs_digital": energy_ratio,
                            "profile_estimated_latency_ns": estimated_latency_ns,
                            "profile_latency_ratio_vs_digital": latency_ratio,
                            "base_accuracy_percent": base_accuracy_percent,
                            "base_min_accuracy_percent": base_min_accuracy_percent,
                            "modeled_accuracy_penalty_percent": (
                                modeled_accuracy_penalty_percent
                            ),
                            "effective_accuracy_percent": effective_accuracy_percent,
                            "effective_min_accuracy_percent": (
                                effective_min_accuracy_percent
                            ),
                            "projection_mse_mean": physical_row["projection_mse_mean"],
                            "passes_accuracy_floor": passes_accuracy_floor,
                            "energy_below_digital": energy_ratio < 1.0,
                            "latency_below_digital": latency_ratio < 1.0,
                            "passes_all_gates": (
                                passes_accuracy_floor
                                and energy_ratio < 1.0
                                and latency_ratio < 1.0
                            ),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        [
            "passes_all_gates",
            "converter_placement",
            "source_coding",
            "source_bits",
            "profile_energy_ratio_vs_digital",
        ],
        ascending=[False, True, True, False, True],
    )


def summarize_architecture_dac_coding_sweep_rows(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        rows.groupby(["converter_placement", "source_coding", "source_bits"], dropna=False)
        .agg(
            passing_all_rows=("passes_all_gates", "sum"),
            passing_accuracy_rows=("passes_accuracy_floor", "sum"),
            energy_below_digital_rows=("energy_below_digital", "sum"),
            latency_below_digital_rows=("latency_below_digital", "sum"),
            best_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "min"),
            median_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "median"),
            worst_energy_ratio_vs_digital=("profile_energy_ratio_vs_digital", "max"),
            best_latency_ratio_vs_digital=("profile_latency_ratio_vs_digital", "min"),
            median_effective_min_accuracy_percent=(
                "effective_min_accuracy_percent",
                "median",
            ),
            max_modeled_accuracy_penalty_percent=(
                "modeled_accuracy_penalty_percent",
                "max",
            ),
            dac_pj_per_input=("dac_pj_per_input", "first"),
            median_converter_readout_pj=("profiled_converter_readout_pj", "median"),
            rows=("candidate", "count"),
        )
        .reset_index()
    )
    return summary.sort_values(
        [
            "passing_all_rows",
            "source_bits",
            "best_energy_ratio_vs_digital",
        ],
        ascending=[False, True, True],
    )


def run_architecture_dac_coding_sweep_report(
    config: ArchitectureDacCodingSweepConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_architecture_dac_coding_sweep_rows(config)
    summary = summarize_architecture_dac_coding_sweep_rows(rows)
    rows_csv = config.output_dir / "architecture_dac_coding_sweep_rows.csv"
    summary_csv = config.output_dir / "architecture_dac_coding_sweep_summary.csv"
    summary_md = config.output_dir / "architecture_dac_coding_sweep_summary.md"
    manifest_path = config.output_dir / "architecture_dac_coding_sweep_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    passing = summary[summary["passing_all_rows"] > 0]
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "max_passing_all_rows": int(summary["passing_all_rows"].max()),
        "lowest_passing_source_bits": (
            None if passing.empty else int(passing["source_bits"].min())
        ),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _unshared_candidate_accuracy(config: UnsharedFourBlockStressConfig) -> dict[str, float]:
    summary = pd.read_csv(config.unshared_training_summary)
    row = summary[
        (summary["scenario"] == config.material_scenario)
        & (summary["chain_length"] == config.chain_length)
    ].iloc[0]
    return {
        "trained_chain_accuracy": float(row["trained_chain_accuracy_mean"]),
        "raw_constrained_accuracy": float(row["raw_constrained_accuracy_mean"]),
        "calibrated_constrained_accuracy": float(
            row["calibrated_constrained_accuracy_mean"]
        ),
        "calibrated_drop_vs_trained": float(row["calibrated_drop_vs_trained_mean"]),
        "raw_drop_vs_trained": float(row["raw_drop_vs_trained_mean"]),
    }


def build_unshared_four_block_stress_rows(
    config: UnsharedFourBlockStressConfig,
) -> pd.DataFrame:
    candidate = _unshared_candidate_accuracy(config)
    budget_rows = pd.read_csv(config.chained_budget_rows)
    budget = _best_budget_rows(
        budget_rows[budget_rows["chain_length"] == config.chain_length].copy()
    )
    scenario_by_name = {scenario.name: scenario for scenario in config.hardware_scenarios}
    rows: list[dict[str, float | int | str | bool]] = []
    for budget_row in budget.to_dict(orient="records"):
        scenario = scenario_by_name[str(budget_row["hardware_scenario"])]
        dac_energy = config.input_dim * scenario.dac_per_input_pj
        digital_latency = config.chain_length * scenario.digital_matmul_latency_ns
        for readout in config.readouts:
            lanes = max(1, min(readout.lanes, config.output_dim))
            serial_groups = math.ceil(config.output_dim / lanes)
            transfer_count = charge_transfer_count(config.output_dim, lanes)
            transfer_energy = transfer_count * readout.transfer_energy_pj
            adc_energy = (
                config.output_dim
                * scenario.adc_per_output_pj
                * readout.adc_energy_scale
            )
            output_amp_energy = config.output_dim * readout.output_amp_per_sample_pj
            detector_tia_energy = (
                config.output_dim * scenario.detector_tia_per_output_pj
                if readout.transfer_energy_pj == 0.0
                else output_amp_energy
            )
            readout_energy = adc_energy + detector_tia_energy + transfer_energy
            adjusted_energy = (
                float(budget_row["total_energy_per_chain_pj"])
                - float(budget_row["converter_energy_pj"])
                - float(budget_row["detector_tia_energy_pj"])
                + dac_energy
                + readout_energy
            )
            readout_latency = (
                readout.integration_time_ns
                + transfer_count * readout.transfer_latency_ns
                + serial_groups * scenario.adc_latency_ns * readout.adc_latency_scale
            )
            adjusted_latency = (
                scenario.dac_latency_ns
                + config.chain_length * scenario.optical_compute_latency_ns
                + readout_latency
                + scenario.correction_latency_ns
            )
            base_readout_drop = (
                transfer_count / 100.0 * readout.read_noise_drop_per_100_transfers
            )
            for stress in config.stress_profiles:
                drift_drop = (
                    candidate["calibrated_drop_vs_trained"]
                    * stress.drift_drop_multiplier
                )
                readout_drop = base_readout_drop * stress.readout_noise_multiplier
                stressed_accuracy = max(
                    0.0,
                    candidate["trained_chain_accuracy"] - drift_drop - readout_drop,
                )
                rows.append(
                    {
                        "candidate": "unshared_4_block",
                        "stress_profile": stress.name,
                        "hardware_scenario": scenario.name,
                        "readout_scenario": readout.name,
                        "lanes": lanes,
                        "serial_groups": serial_groups,
                        "charge_transfer_count": transfer_count,
                        "trained_chain_accuracy": candidate["trained_chain_accuracy"],
                        "stressed_accuracy": stressed_accuracy,
                        "stressed_accuracy_percent": stressed_accuracy * 100.0,
                        "accuracy_drop_vs_trained": drift_drop + readout_drop,
                        "drift_drop_component": drift_drop,
                        "readout_noise_drop_component": readout_drop,
                        "readout_energy_pj": readout_energy,
                        "adjusted_total_energy_per_chain_pj": adjusted_energy,
                        "energy_ratio_vs_digital_chain": adjusted_energy
                        / float(budget_row["digital_chain_energy_pj"]),
                        "readout_latency_ns": readout_latency,
                        "adjusted_latency_ns": adjusted_latency,
                        "latency_ratio_vs_digital_chain": adjusted_latency
                        / digital_latency,
                        "fails_85_percent_accuracy": stressed_accuracy < 0.85,
                        "fails_digital_energy": adjusted_energy
                        > float(budget_row["digital_chain_energy_pj"]),
                        "fails_digital_latency": adjusted_latency > digital_latency,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [
            "fails_85_percent_accuracy",
            "fails_digital_energy",
            "fails_digital_latency",
            "stressed_accuracy",
            "energy_ratio_vs_digital_chain",
        ],
        ascending=[True, True, True, False, True],
    )


def summarize_unshared_four_block_stress_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["stress_profile", "readout_scenario"], dropna=False)
    summary = grouped.agg(
        median_accuracy_percent=("stressed_accuracy_percent", "median"),
        worst_accuracy_percent=("stressed_accuracy_percent", "min"),
        best_energy_ratio_vs_digital_chain=("energy_ratio_vs_digital_chain", "min"),
        best_latency_ratio_vs_digital_chain=("latency_ratio_vs_digital_chain", "min"),
        hardware_cases_passing_85_percent=("fails_85_percent_accuracy", lambda x: int((~x).sum())),
        hardware_cases_energy_below_digital=("fails_digital_energy", lambda x: int((~x).sum())),
        hardware_cases_latency_below_digital=("fails_digital_latency", lambda x: int((~x).sum())),
        rows=("candidate", "count"),
    ).reset_index()
    return summary.sort_values(
        [
            "worst_accuracy_percent",
            "best_energy_ratio_vs_digital_chain",
            "best_latency_ratio_vs_digital_chain",
        ],
        ascending=[False, True, True],
    )


def run_unshared_four_block_stress_report(
    config: UnsharedFourBlockStressConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_unshared_four_block_stress_rows(config)
    summary = summarize_unshared_four_block_stress_rows(rows)
    rows_csv = config.output_dir / "unshared_4block_stress_rows.csv"
    summary_csv = config.output_dir / "unshared_4block_stress_summary.csv"
    summary_md = config.output_dir / "unshared_4block_stress_summary.md"
    manifest_path = config.output_dir / "unshared_4block_stress_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "chained_budget_rows": str(config.chained_budget_rows),
            "unshared_training_summary": str(config.unshared_training_summary),
            "output_dir": str(config.output_dir),
            "material_scenario": config.material_scenario,
            "chain_length": config.chain_length,
            "stress_profiles": [asdict(stress) for stress in config.stress_profiles],
            "readout_scenarios": [asdict(readout) for readout in config.readouts],
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "passing_85_percent_rows": int((~rows["fails_85_percent_accuracy"]).sum()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_energy_accuracy_pareto_report(
    config: EnergyAccuracyParetoConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_energy_accuracy_pareto_rows(config)
    summary = summarize_energy_accuracy_pareto_rows(rows)
    rows_csv = config.output_dir / "energy_accuracy_pareto_rows.csv"
    summary_csv = config.output_dir / "energy_accuracy_pareto_summary.csv"
    summary_md = config.output_dir / "energy_accuracy_pareto_summary.md"
    manifest_path = config.output_dir / "energy_accuracy_pareto_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "chained_budget_rows": str(config.chained_budget_rows),
            "shared_training_summary": str(config.shared_training_summary),
            "unshared_training_summary": str(config.unshared_training_summary),
            "output_dir": str(config.output_dir),
            "material_scenario": config.material_scenario,
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "pareto_efficient_rows": int(rows["pareto_efficient"].sum()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_charge_readout_budget(config: ChargeReadoutBudgetConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = estimate_charge_readout_rows(config)
    summary = summarize_charge_readout_rows(rows)
    rows_csv = config.output_dir / "charge_readout_budget_rows.csv"
    summary_csv = config.output_dir / "charge_readout_budget_summary.csv"
    summary_md = config.output_dir / "charge_readout_budget_summary.md"
    manifest_path = config.output_dir / "charge_readout_budget_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "calibration_summary": str(config.calibration_summary),
            "output_dir": str(config.output_dir),
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "inferences_per_drift_step": config.inferences_per_drift_step,
            "hardware_scenarios": [asdict(scenario) for scenario in config.scenarios],
            "readout_scenarios": [asdict(readout) for readout in config.readouts],
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_energy_budget(config: EnergyBudgetConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = estimate_budget_rows(config)
    summary = summarize_budget_rows(rows)
    rows_csv = config.output_dir / "energy_budget_rows.csv"
    summary_csv = config.output_dir / "energy_budget_summary.csv"
    summary_md = config.output_dir / "energy_budget_summary.md"
    manifest_path = config.output_dir / "energy_budget_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "calibration_summary": str(config.calibration_summary),
            "output_dir": str(config.output_dir),
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "inferences_per_drift_step": config.inferences_per_drift_step,
            "hardware_scenarios": [asdict(scenario) for scenario in config.scenarios],
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_chained_block_budget(config: ChainedBlockBudgetConfig) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = estimate_chained_block_rows(config)
    summary = summarize_chained_block_rows(rows)
    rows_csv = config.output_dir / "chained_block_budget_rows.csv"
    summary_csv = config.output_dir / "chained_block_budget_summary.csv"
    summary_md = config.output_dir / "chained_block_budget_summary.md"
    manifest_path = config.output_dir / "chained_block_budget_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "calibration_summary": str(config.calibration_summary),
            "output_dir": str(config.output_dir),
            "input_dim": config.input_dim,
            "output_dim": config.output_dim,
            "inferences_per_drift_step": config.inferences_per_drift_step,
            "chain_lengths": list(config.chain_lengths),
            "hardware_scenarios": [asdict(scenario) for scenario in config.scenarios],
        },
        "rows": int(len(rows)),
        "summary_rows": int(len(summary)),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
