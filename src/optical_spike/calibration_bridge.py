"""Translate normalized tolerance targets into first-pass physical quantities."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from optical_spike.reporting import markdown_table


@dataclass(frozen=True)
class CalibrationBridgeConfig:
    target_csv: Path
    output_dir: Path
    warning_edge_normalized_drift: float = 0.005
    warning_edge_resonance_shift_pm: float = 12.0
    thermal_sensitivity_pm_per_k: float = 68.2
    resonance_fwhm_pm: float = 100.0
    wavelength_nm: float = 1550.0
    assumed_temp_ramp_k_per_min: tuple[float, ...] = (1.0, 0.1, 0.01)


@dataclass(frozen=True)
class ArchitectureBridgeAssumption:
    name: str
    description: str
    warning_edge_resonance_shift_pm: float
    thermal_sensitivity_pm_per_k: float
    resonance_fwhm_pm: float
    electronics_energy_multiplier: float


DEFAULT_ARCHITECTURE_ASSUMPTIONS: tuple[ArchitectureBridgeAssumption, ...] = (
    ArchitectureBridgeAssumption(
        name="tight_mrr_weight_bank",
        description="narrow resonant ring weight bank with little drift margin",
        warning_edge_resonance_shift_pm=6.0,
        thermal_sensitivity_pm_per_k=68.2,
        resonance_fwhm_pm=80.0,
        electronics_energy_multiplier=20.0,
    ),
    ArchitectureBridgeAssumption(
        name="source_log_anchor_mrr",
        description="current source-log MRR anchor from the first bridge pass",
        warning_edge_resonance_shift_pm=12.0,
        thermal_sensitivity_pm_per_k=68.2,
        resonance_fwhm_pm=100.0,
        electronics_energy_multiplier=10.0,
    ),
    ArchitectureBridgeAssumption(
        name="broader_stabilized_mrr",
        description="broader or actively stabilized resonant implementation",
        warning_edge_resonance_shift_pm=24.0,
        thermal_sensitivity_pm_per_k=20.0,
        resonance_fwhm_pm=150.0,
        electronics_energy_multiplier=10.0,
    ),
    ArchitectureBridgeAssumption(
        name="athermal_compensated_mrr",
        description="ring implementation with roughly 10x lower effective thermal sensitivity",
        warning_edge_resonance_shift_pm=12.0,
        thermal_sensitivity_pm_per_k=6.82,
        resonance_fwhm_pm=100.0,
        electronics_energy_multiplier=10.0,
    ),
)


def _target_row(targets: pd.DataFrame, target: str) -> pd.Series:
    matches = targets[targets["target"] == target]
    if matches.empty:
        raise ValueError(f"missing target row: {target}")
    return matches.iloc[0]


def _float_value(row: pd.Series, column: str) -> float:
    value = row[column]
    if value == "":
        raise ValueError(f"missing numeric value for {column}")
    return float(value)


def _photon_energy_j(wavelength_nm: float) -> float:
    planck_j_s = 6.62607015e-34
    speed_of_light_m_s = 299_792_458
    return planck_j_s * speed_of_light_m_s / (wavelength_nm * 1e-9)


def _shot_noise_photons(relative_noise: float) -> float:
    return 1.0 / (relative_noise**2)


def _amplitude_snr_db(relative_noise: float) -> float:
    return 20.0 * math.log10(1.0 / relative_noise)


def _enob_from_snr_db(snr_db: float) -> float:
    return (snr_db - 1.76) / 6.02


def _target_values(
    targets: pd.DataFrame,
) -> tuple[float, float, float, float, float, float]:
    recommended = _target_row(targets, "recommended_operating_envelope")
    warning = _target_row(targets, "warning_edge")
    recalibration = _target_row(targets, "recalibration_cadence")

    return (
        _float_value(recommended, "max_thermal_drift_rate"),
        _float_value(recommended, "max_detector_noise"),
        _float_value(recommended, "max_output_readout_noise"),
        _float_value(recalibration, "max_recalibration_interval_steps"),
        _float_value(warning, "max_thermal_drift_rate"),
        _float_value(recommended, "projection_mse_warning_threshold"),
    )


def build_calibration_bridge_rows(config: CalibrationBridgeConfig) -> pd.DataFrame:
    targets = pd.read_csv(config.target_csv)
    (
        max_drift,
        max_detector_noise,
        max_output_noise,
        max_recalibration_steps,
        warning_drift,
        projection_mse_warning,
    ) = _target_values(targets)

    resonance_shift_per_norm = (
        config.warning_edge_resonance_shift_pm / config.warning_edge_normalized_drift
    )
    drift_shift_pm_per_step = max_drift * resonance_shift_per_norm
    warning_shift_pm_per_step = warning_drift * resonance_shift_per_norm
    temp_excursion_k_per_step = drift_shift_pm_per_step / config.thermal_sensitivity_pm_per_k
    warning_temp_excursion_k_per_step = (
        warning_shift_pm_per_step / config.thermal_sensitivity_pm_per_k
    )
    phase_error_rad = 2.0 * math.pi * drift_shift_pm_per_step / config.resonance_fwhm_pm
    photon_energy_j = _photon_energy_j(config.wavelength_nm)

    detector_photons = _shot_noise_photons(max_detector_noise)
    detector_shot_energy_fj = detector_photons * photon_energy_j * 1e15
    output_photons = _shot_noise_photons(max_output_noise)
    output_shot_energy_fj = output_photons * photon_energy_j * 1e15
    output_snr_db = _amplitude_snr_db(max_output_noise)

    recalibration_temp_window_k = temp_excursion_k_per_step * max_recalibration_steps
    time_rows = []
    for ramp in config.assumed_temp_ramp_k_per_min:
        minutes = recalibration_temp_window_k / ramp
        time_rows.append(f"{ramp:g} K/min -> {minutes * 60.0:.1f} s")

    rows = [
        {
            "bridge": "thermal_resonance_drift",
            "normalized_target": max_drift,
            "physical_quantity": "resonance shift per simulated drift step",
            "estimate": f"{drift_shift_pm_per_step:.2f} pm/step",
            "secondary_estimate": f"{temp_excursion_k_per_step:.3f} K/step",
            "assumption": (
                f"map normalized warning drift {config.warning_edge_normalized_drift:g} "
                f"to {config.warning_edge_resonance_shift_pm:g} pm"
            ),
            "status": "needs_device_calibration",
        },
        {
            "bridge": "warning_edge_resonance_drift",
            "normalized_target": warning_drift,
            "physical_quantity": "warning-edge resonance shift per drift step",
            "estimate": f"{warning_shift_pm_per_step:.2f} pm/step",
            "secondary_estimate": f"{warning_temp_excursion_k_per_step:.3f} K/step",
            "assumption": (
                f"thermal sensitivity {config.thermal_sensitivity_pm_per_k:g} pm/K"
            ),
            "status": "marginal_by_simulation",
        },
        {
            "bridge": "phase_error_budget",
            "normalized_target": max_drift,
            "physical_quantity": "equivalent phase drift around resonance linewidth",
            "estimate": f"{phase_error_rad:.3f} rad/step",
            "secondary_estimate": f"FWHM anchor {config.resonance_fwhm_pm:g} pm",
            "assumption": "linearized 2*pi*shift/FWHM placeholder",
            "status": "architecture_dependent",
        },
        {
            "bridge": "detector_noise_snr",
            "normalized_target": max_detector_noise,
            "physical_quantity": "shot-noise floor for per-block detector noise",
            "estimate": f">= {detector_photons:.0f} photons/sample",
            "secondary_estimate": (
                f"{_amplitude_snr_db(max_detector_noise):.1f} dB amplitude SNR; "
                f"{detector_shot_energy_fj:.3f} fJ shot-noise floor"
            ),
            "assumption": f"wavelength {config.wavelength_nm:g} nm, electronics excluded",
            "status": "energy_snr_tradeoff",
        },
        {
            "bridge": "output_readout_snr",
            "normalized_target": max_output_noise,
            "physical_quantity": "final-boundary readout noise floor",
            "estimate": f">= {output_photons:.0f} photons/sample",
            "secondary_estimate": (
                f"{output_snr_db:.1f} dB amplitude SNR; "
                f"{_enob_from_snr_db(output_snr_db):.2f} ENOB equivalent; "
                f"{output_shot_energy_fj:.3f} fJ shot-noise floor"
            ),
            "assumption": f"wavelength {config.wavelength_nm:g} nm, ADC/TIA excluded",
            "status": "demanding_but_plausible",
        },
        {
            "bridge": "recalibration_timing",
            "normalized_target": max_recalibration_steps,
            "physical_quantity": "temperature window before scheduled refresh",
            "estimate": f"{recalibration_temp_window_k:.3f} K per recalibration window",
            "secondary_estimate": "; ".join(time_rows),
            "assumption": (
                "uses thermal drift bridge; ramp rates are placeholders until measured"
            ),
            "status": "needs_wall_clock_mapping",
        },
        {
            "bridge": "projection_mse_warning",
            "normalized_target": projection_mse_warning,
            "physical_quantity": "internal transfer-health threshold",
            "estimate": f"MSE >= {projection_mse_warning:.3f}",
            "secondary_estimate": "not directly physical yet",
            "assumption": "must be regressed against measured optical transfer error",
            "status": "internal_monitor_only",
        },
    ]
    return pd.DataFrame(rows)


def build_architecture_sensitivity_rows(
    config: CalibrationBridgeConfig,
    assumptions: tuple[ArchitectureBridgeAssumption, ...] = DEFAULT_ARCHITECTURE_ASSUMPTIONS,
) -> pd.DataFrame:
    targets = pd.read_csv(config.target_csv)
    (
        max_drift,
        max_detector_noise,
        max_output_noise,
        max_recalibration_steps,
        warning_drift,
        _projection_mse_warning,
    ) = _target_values(targets)

    photon_energy_j = _photon_energy_j(config.wavelength_nm)
    detector_photons = _shot_noise_photons(max_detector_noise)
    detector_shot_energy_fj = detector_photons * photon_energy_j * 1e15
    output_photons = _shot_noise_photons(max_output_noise)
    output_shot_energy_fj = output_photons * photon_energy_j * 1e15
    output_snr_db = _amplitude_snr_db(max_output_noise)

    rows: list[dict[str, object]] = []
    for assumption in assumptions:
        shift_per_norm = (
            assumption.warning_edge_resonance_shift_pm / warning_drift
        )
        target_shift_pm_per_step = max_drift * shift_per_norm
        target_temp_k_per_step = (
            target_shift_pm_per_step / assumption.thermal_sensitivity_pm_per_k
        )
        recalibration_temp_window_k = target_temp_k_per_step * max_recalibration_steps
        phase_error_rad = (
            2.0
            * math.pi
            * target_shift_pm_per_step
            / assumption.resonance_fwhm_pm
        )
        detector_total_energy_fj = (
            detector_shot_energy_fj * assumption.electronics_energy_multiplier
        )
        output_total_energy_fj = (
            output_shot_energy_fj * assumption.electronics_energy_multiplier
        )
        if recalibration_temp_window_k < 0.1:
            status = "very_tight_thermal_control"
        elif recalibration_temp_window_k < 0.5:
            status = "active_control_needed"
        else:
            status = "plausible_with_stabilization"

        rows.append(
            {
                "architecture": assumption.name,
                "description": assumption.description,
                "warning_shift_pm": assumption.warning_edge_resonance_shift_pm,
                "thermal_sensitivity_pm_per_k": assumption.thermal_sensitivity_pm_per_k,
                "resonance_fwhm_pm": assumption.resonance_fwhm_pm,
                "recommended_shift_pm_per_step": target_shift_pm_per_step,
                "recommended_temp_k_per_step": target_temp_k_per_step,
                "recalibration_window_k": recalibration_temp_window_k,
                "phase_error_rad_per_step": phase_error_rad,
                "detector_photons_per_sample": detector_photons,
                "output_photons_per_sample": output_photons,
                "output_enob_equivalent": _enob_from_snr_db(output_snr_db),
                "detector_energy_floor_fj": detector_shot_energy_fj,
                "output_energy_floor_fj": output_shot_energy_fj,
                "electronics_energy_multiplier": assumption.electronics_energy_multiplier,
                "detector_energy_with_overhead_fj": detector_total_energy_fj,
                "output_energy_with_overhead_fj": output_total_energy_fj,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def run_calibration_bridge_report(
    config: CalibrationBridgeConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_calibration_bridge_rows(config)
    sensitivity = build_architecture_sensitivity_rows(config)
    rows_csv = config.output_dir / "calibration_bridge_rows.csv"
    rows_md = config.output_dir / "calibration_bridge.md"
    sensitivity_csv = config.output_dir / "calibration_bridge_sensitivity.csv"
    sensitivity_md = config.output_dir / "calibration_bridge_sensitivity.md"
    manifest_path = config.output_dir / "calibration_bridge_manifest.json"
    rows.to_csv(rows_csv, index=False)
    rows_md.write_text(markdown_table(rows), encoding="utf-8")
    sensitivity.to_csv(sensitivity_csv, index=False)
    sensitivity_md.write_text(markdown_table(sensitivity), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "rows": int(len(rows)),
        "sensitivity_rows": int(len(sensitivity)),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "rows_md": str(rows_md),
            "sensitivity_csv": str(sensitivity_csv),
            "sensitivity_md": str(sensitivity_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
