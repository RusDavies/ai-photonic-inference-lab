"""End-to-end training with material-inspired MLP activation functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from optical_spike.baseline import adam_update, cross_entropy, one_hot, relu, softmax
from optical_spike.calibration_cost import drifted_weights, train_transformer
from optical_spike.calibration_bridge import (
    DEFAULT_ARCHITECTURE_ASSUMPTIONS,
    ArchitectureBridgeAssumption,
)
from optical_spike.fabrication_constraints import (
    DEFAULT_CONSTRAINED_FIXED_SCENARIOS,
    ConstrainedFixedScenario,
    apply_constrained_fixed_weight_errors,
)
from optical_spike.material_nonlinearity import MaterialNonlinearityScenario
from optical_spike.quantization import quantize_symmetric
from optical_spike.reporting import markdown_table
from optical_spike.transformer_mlp import (
    TransformerMLPConfig,
    init_params,
    load_train_and_test,
    scenario_seed,
)
from optical_spike.tunable import fit_affine_calibration


DEFAULT_TRAINED_MATERIAL_ACTIVATIONS: tuple[MaterialNonlinearityScenario, ...] = (
    MaterialNonlinearityScenario(
        "saturable_absorber_trained",
        "saturable_absorber",
        "saturable_absorption",
        0.25,
    ),
    MaterialNonlinearityScenario(
        "reverse_saturable_trained",
        "reverse_saturable_absorber",
        "reverse_saturable_absorption",
        0.20,
    ),
    MaterialNonlinearityScenario(
        "two_photon_absorption_trained",
        "silicon_or_germanium_tpa",
        "two_photon_absorption",
        0.18,
    ),
    MaterialNonlinearityScenario(
        "kerr_phase_cubic_trained",
        "kerr_or_carrier_phase",
        "cubic_phase",
        0.08,
    ),
)


@dataclass(frozen=True)
class MaterialTrainingConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    scenarios: tuple[MaterialNonlinearityScenario, ...] = DEFAULT_TRAINED_MATERIAL_ACTIVATIONS


@dataclass(frozen=True)
class MaterialStrengthSweepConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    strengths: tuple[float, ...] = (0.1, 0.2, 0.35)
    calibration_samples: int = 512
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]


DEFAULT_FUNCTIONAL_CHAIN_MATERIALS: tuple[MaterialNonlinearityScenario, ...] = (
    MaterialNonlinearityScenario(
        "reverse_saturable_strength_0p1",
        "reverse_saturable_absorber",
        "reverse_saturable_absorption",
        0.1,
    ),
    MaterialNonlinearityScenario(
        "reverse_saturable_strength_0p2",
        "reverse_saturable_absorber",
        "reverse_saturable_absorption",
        0.2,
    ),
)


@dataclass(frozen=True)
class FunctionalChainConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    chain_lengths: tuple[int, ...] = (1, 2, 4)
    calibration_samples: int = 512
    scenarios: tuple[MaterialNonlinearityScenario, ...] = DEFAULT_FUNCTIONAL_CHAIN_MATERIALS
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]


@dataclass(frozen=True)
class ChainedMaterialTrainingConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    chain_lengths: tuple[int, ...] = (2, 4)
    calibration_samples: int = 512
    scenarios: tuple[MaterialNonlinearityScenario, ...] = DEFAULT_FUNCTIONAL_CHAIN_MATERIALS
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]


DEFAULT_UNSHARED_CHAIN_MATERIALS: tuple[MaterialNonlinearityScenario, ...] = (
    MaterialNonlinearityScenario(
        "reverse_saturable_strength_0p2",
        "reverse_saturable_absorber",
        "reverse_saturable_absorption",
        0.2,
    ),
)


@dataclass(frozen=True)
class UnsharedChainedMaterialTrainingConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    chain_lengths: tuple[int, ...] = (2, 4)
    calibration_samples: int = 512
    scenarios: tuple[MaterialNonlinearityScenario, ...] = DEFAULT_UNSHARED_CHAIN_MATERIALS
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]


@dataclass(frozen=True)
class PhysicalDriftNoiseScenario:
    name: str
    thermal_drift_rate: float
    detector_noise: float
    output_readout_noise: float
    recalibration_interval_steps: int
    time_steps: int
    architecture_profile: str = "normalized_baseline"
    resonance_shift_pm_per_step: float = 0.0
    thermal_excursion_k_per_step: float = 0.0
    detector_photons_per_sample: float = 0.0
    output_photons_per_sample: float = 0.0
    output_enob_equivalent: float = 0.0


@dataclass(frozen=True)
class SourceCodingScenario:
    name: str
    bits: int
    redundant_phases: int
    source_noise: float
    target_dac_pj_per_sample: float
    target_dac_enob: float
    phase_latency_ns: float
    note: str

    @property
    def dac_pj_per_input(self) -> float:
        return (
            self.target_dac_pj_per_sample
            * (self.bits / self.target_dac_enob)
            * self.redundant_phases
        )

    @property
    def latency_ns(self) -> float:
        return self.phase_latency_ns * self.redundant_phases


REDUNDANT_6BIT_SOURCE_CODING = SourceCodingScenario(
    name="redundant_6bit_2phase",
    bits=6,
    redundant_phases=2,
    source_noise=0.002,
    target_dac_pj_per_sample=2.5,
    target_dac_enob=7.35,
    phase_latency_ns=0.1,
    note=(
        "Surviving source-coding branch from the DAC coding sweep: two 6-bit "
        "source phases averaged after dithered quantization."
    ),
)

DIRECT_7BIT_SOURCE_CODING = SourceCodingScenario(
    name="direct_7bit",
    bits=7,
    redundant_phases=1,
    source_noise=0.0015,
    target_dac_pj_per_sample=2.5,
    target_dac_enob=7.35,
    phase_latency_ns=0.1,
    note="Single 7-bit source phase used as the direct precision comparison.",
)

REDUNDANT_7BIT_SOURCE_CODING = SourceCodingScenario(
    name="redundant_7bit_2phase",
    bits=7,
    redundant_phases=2,
    source_noise=0.0015,
    target_dac_pj_per_sample=2.5,
    target_dac_enob=7.35,
    phase_latency_ns=0.1,
    note="Two 7-bit source phases averaged after dithered quantization.",
)

SOURCE_CODING_STRESS_SCENARIOS: tuple[SourceCodingScenario, ...] = (
    REDUNDANT_6BIT_SOURCE_CODING,
    DIRECT_7BIT_SOURCE_CODING,
    REDUNDANT_7BIT_SOURCE_CODING,
)


DEFAULT_PHYSICAL_DRIFT_NOISE_SCENARIOS: tuple[PhysicalDriftNoiseScenario, ...] = (
    PhysicalDriftNoiseScenario("stable_cmos_boundary", 0.001, 0.01, 0.0, 4, 8),
    PhysicalDriftNoiseScenario("warm_cmos_boundary", 0.002, 0.02, 0.0, 4, 8),
    PhysicalDriftNoiseScenario("noisy_near_column_boundary", 0.002, 0.02, 0.005, 4, 8),
    PhysicalDriftNoiseScenario("hot_cmos_sparse_recalibration", 0.005, 0.03, 0.0, 8, 8),
    PhysicalDriftNoiseScenario("hot_noisy_boundary", 0.005, 0.03, 0.01, 4, 8),
)

CANDIDATE_PHYSICAL_ARCHITECTURE_PROFILE = "athermal_compensated_mrr"


def _architecture_assumption(profile: str) -> ArchitectureBridgeAssumption:
    for assumption in DEFAULT_ARCHITECTURE_ASSUMPTIONS:
        if assumption.name == profile:
            return assumption
    available = ", ".join(assumption.name for assumption in DEFAULT_ARCHITECTURE_ASSUMPTIONS)
    raise ValueError(
        f"unknown physical architecture profile {profile!r}; choose one of {available}"
    )


def _photons_for_relative_noise(relative_noise: float) -> float:
    if relative_noise <= 0.0:
        return 0.0
    return 1.0 / (relative_noise**2)


def _enob_for_relative_noise(relative_noise: float) -> float:
    if relative_noise <= 0.0:
        return 0.0
    snr_db = 20.0 * math.log10(1.0 / relative_noise)
    return (snr_db - 1.76) / 6.02


def _architecture_physical_scenario(
    *,
    assumption: ArchitectureBridgeAssumption,
    suffix: str,
    thermal_drift_rate: float,
    detector_noise: float,
    output_readout_noise: float,
    recalibration_interval_steps: int,
    time_steps: int = 8,
    warning_edge_normalized_drift: float = 0.005,
) -> PhysicalDriftNoiseScenario:
    shift_per_norm = assumption.warning_edge_resonance_shift_pm / warning_edge_normalized_drift
    resonance_shift_pm_per_step = thermal_drift_rate * shift_per_norm
    thermal_excursion_k_per_step = (
        resonance_shift_pm_per_step / assumption.thermal_sensitivity_pm_per_k
    )
    return PhysicalDriftNoiseScenario(
        name=f"{assumption.name}_{suffix}",
        thermal_drift_rate=thermal_drift_rate,
        detector_noise=detector_noise,
        output_readout_noise=output_readout_noise,
        recalibration_interval_steps=recalibration_interval_steps,
        time_steps=time_steps,
        architecture_profile=assumption.name,
        resonance_shift_pm_per_step=resonance_shift_pm_per_step,
        thermal_excursion_k_per_step=thermal_excursion_k_per_step,
        detector_photons_per_sample=_photons_for_relative_noise(detector_noise),
        output_photons_per_sample=_photons_for_relative_noise(output_readout_noise),
        output_enob_equivalent=_enob_for_relative_noise(output_readout_noise),
    )


def physical_scenarios_for_architecture(
    profile: str = CANDIDATE_PHYSICAL_ARCHITECTURE_PROFILE,
) -> tuple[PhysicalDriftNoiseScenario, ...]:
    assumption = _architecture_assumption(profile)
    return (
        _architecture_physical_scenario(
            assumption=assumption,
            suffix="nominal",
            thermal_drift_rate=0.002,
            detector_noise=0.02,
            output_readout_noise=0.005,
            recalibration_interval_steps=4,
        ),
        _architecture_physical_scenario(
            assumption=assumption,
            suffix="sparse_refresh",
            thermal_drift_rate=0.002,
            detector_noise=0.02,
            output_readout_noise=0.005,
            recalibration_interval_steps=8,
        ),
        _architecture_physical_scenario(
            assumption=assumption,
            suffix="thermal_edge",
            thermal_drift_rate=0.005,
            detector_noise=0.03,
            output_readout_noise=0.005,
            recalibration_interval_steps=4,
        ),
        _architecture_physical_scenario(
            assumption=assumption,
            suffix="readout_edge",
            thermal_drift_rate=0.002,
            detector_noise=0.02,
            output_readout_noise=0.01,
            recalibration_interval_steps=4,
        ),
    )


@dataclass(frozen=True)
class UnsharedPhysicalDriftNoiseConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    chain_length: int = 4
    calibration_samples: int = 512
    scenarios: tuple[MaterialNonlinearityScenario, ...] = DEFAULT_UNSHARED_CHAIN_MATERIALS
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]
    physical_scenarios: tuple[PhysicalDriftNoiseScenario, ...] = (
        DEFAULT_PHYSICAL_DRIFT_NOISE_SCENARIOS
    )
    source_coding: SourceCodingScenario | None = None
    output_dir_name: str = "unshared_physical_drift_noise"
    projection_mse_recalibration_threshold: float | None = None


@dataclass(frozen=True)
class SourceCodingCalibrationStressConfig:
    transformer: TransformerMLPConfig
    output_dir: Path
    seeds: tuple[int, ...] = (7, 11, 13)
    chain_length: int = 4
    transformer_material_scenarios: tuple[MaterialNonlinearityScenario, ...] = (
        DEFAULT_UNSHARED_CHAIN_MATERIALS
    )
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]
    calibration_sample_counts: tuple[int, ...] = (128, 512, 2048)
    source_codings: tuple[SourceCodingScenario, ...] = SOURCE_CODING_STRESS_SCENARIOS
    physical_scenarios: tuple[PhysicalDriftNoiseScenario, ...] = (
        DEFAULT_PHYSICAL_DRIFT_NOISE_SCENARIOS
    )
    projection_mse_recalibration_threshold: float = 0.18


@dataclass(frozen=True)
class DeviceToleranceTargetConfig:
    physical_summary: Path
    output_dir: Path
    physical_scenarios: tuple[PhysicalDriftNoiseScenario, ...] = (
        DEFAULT_PHYSICAL_DRIFT_NOISE_SCENARIOS
    )


def strength_sweep_scenarios(
    strengths: tuple[float, ...],
) -> tuple[MaterialNonlinearityScenario, ...]:
    scenarios: list[MaterialNonlinearityScenario] = []
    for strength in strengths:
        suffix = str(strength).replace(".", "p")
        scenarios.extend(
            [
                MaterialNonlinearityScenario(
                    f"saturable_absorber_strength_{suffix}",
                    "saturable_absorber",
                    "saturable_absorption",
                    strength,
                ),
                MaterialNonlinearityScenario(
                    f"reverse_saturable_strength_{suffix}",
                    "reverse_saturable_absorber",
                    "reverse_saturable_absorption",
                    strength,
                ),
            ]
        )
    return tuple(scenarios)


def material_activation_and_derivative(
    projection: np.ndarray,
    scenario: MaterialNonlinearityScenario,
) -> tuple[np.ndarray, np.ndarray]:
    scale = max(float(np.std(projection)), 1e-8)
    normalized = projection / scale
    positive_mask = projection > 0.0
    positive = np.maximum(projection, 0.0)
    if scenario.kind in {"saturable_absorption", "reverse_saturable_absorption"}:
        denominator = 1.0 + scenario.strength * np.maximum(normalized, 0.0)
        activated = positive / denominator
        derivative = np.where(positive_mask, 1.0 / (denominator * denominator), 0.0)
    elif scenario.kind == "two_photon_absorption":
        denominator = 1.0 + scenario.strength * normalized * normalized
        activated = positive / denominator
        derivative = np.where(
            positive_mask,
            (1.0 - scenario.strength * normalized * normalized) / (denominator * denominator),
            0.0,
        )
    elif scenario.kind == "cubic_phase":
        activated = positive + scenario.strength * positive * normalized * normalized
        derivative = np.where(
            positive_mask,
            1.0 + 3.0 * scenario.strength * normalized * normalized,
            0.0,
        )
    else:
        raise ValueError(f"Unsupported trainable activation kind: {scenario.kind}")
    return activated.astype(np.float32), derivative.astype(np.float32)


def forward_material(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    *,
    w_mlp_up: np.ndarray | None = None,
    detector_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    up_weights = params["w_mlp_up"] if w_mlp_up is None else w_mlp_up
    embed_projection = inputs @ params["w_embed"] + params["b_embed"]
    embed = relu(embed_projection)
    mlp_up_projection = embed @ up_weights + params["b_mlp_up"]
    if detector_noise > 0.0:
        noise_rng = np.random.default_rng(0) if rng is None else rng
        scale = max(float(np.std(mlp_up_projection)), 1e-8)
        mlp_up_projection = mlp_up_projection + noise_rng.normal(
            0.0,
            detector_noise * scale,
            size=mlp_up_projection.shape,
        )
    mlp_up, mlp_up_derivative = material_activation_and_derivative(mlp_up_projection, scenario)
    mlp_down_projection = mlp_up @ params["w_mlp_down"] + params["b_mlp_down"]
    residual = relu(embed + mlp_down_projection)
    logits = residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {
        "embed_projection": embed_projection,
        "embed": embed,
        "mlp_up_projection": mlp_up_projection,
        "mlp_up": mlp_up,
        "mlp_up_derivative": mlp_up_derivative,
        "mlp_down_projection": mlp_down_projection,
        "residual": residual,
        "logits": logits,
    }


def evaluate_material(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    *,
    w_mlp_up: np.ndarray | None = None,
    detector_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    probs, cache = forward_material(
        params,
        inputs,
        scenario,
        w_mlp_up=w_mlp_up,
        detector_noise=detector_noise,
        rng=rng,
    )
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "mlp_up_projection_mean": float(np.mean(cache["mlp_up_projection"])),
        "mlp_up_projection_std": float(np.std(cache["mlp_up_projection"])),
        "mlp_activation_mean": float(np.mean(cache["mlp_up"])),
        "mlp_activation_std": float(np.std(cache["mlp_up"])),
    }


def train_material_epoch(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    config: TransformerMLPConfig,
    scenario: MaterialNonlinearityScenario,
    rng: np.random.Generator,
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    start_step: int,
) -> int:
    indices = rng.permutation(train_images.shape[0])
    step = start_step
    classes = params["b_head"].shape[0]
    for start in range(0, train_images.shape[0], config.batch_size):
        batch_idx = indices[start : start + config.batch_size]
        x = train_images[batch_idx]
        y = train_labels[batch_idx]
        y_one_hot = one_hot(y, classes)

        probs, cache = forward_material(params, x, scenario)
        batch_size = x.shape[0]
        d_logits = (probs - y_one_hot) / batch_size
        grad_w_head = cache["residual"].T @ d_logits
        grad_b_head = np.sum(d_logits, axis=0)

        d_residual = d_logits @ params["w_head"].T
        d_residual_pre = d_residual * (cache["residual"] > 0.0)
        d_embed_from_residual = d_residual_pre
        d_mlp_down_projection = d_residual_pre

        grad_w_mlp_down = cache["mlp_up"].T @ d_mlp_down_projection
        grad_b_mlp_down = np.sum(d_mlp_down_projection, axis=0)

        d_mlp_up = d_mlp_down_projection @ params["w_mlp_down"].T
        d_mlp_up_projection = d_mlp_up * cache["mlp_up_derivative"]
        grad_w_mlp_up = cache["embed"].T @ d_mlp_up_projection
        grad_b_mlp_up = np.sum(d_mlp_up_projection, axis=0)

        d_embed_from_mlp = d_mlp_up_projection @ params["w_mlp_up"].T
        d_embed = d_embed_from_residual + d_embed_from_mlp
        d_embed_projection = d_embed * (cache["embed_projection"] > 0.0)
        grad_w_embed = x.T @ d_embed_projection
        grad_b_embed = np.sum(d_embed_projection, axis=0)

        grads = {
            "w_embed": grad_w_embed.astype(np.float32),
            "b_embed": grad_b_embed.astype(np.float32),
            "w_mlp_up": grad_w_mlp_up.astype(np.float32),
            "b_mlp_up": grad_b_mlp_up.astype(np.float32),
            "w_mlp_down": grad_w_mlp_down.astype(np.float32),
            "b_mlp_down": grad_b_mlp_down.astype(np.float32),
            "w_head": grad_w_head.astype(np.float32),
            "b_head": grad_b_head.astype(np.float32),
        }
        step += 1
        adam_update(params, grads, moments, velocities, step, config.learning_rate)
    return step


def train_material_model(
    config: TransformerMLPConfig,
    scenario: MaterialNonlinearityScenario,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], dict[str, object]]:
    rng = np.random.default_rng(config.seed)
    train_images, train_labels, test_images, test_labels = load_train_and_test(config)
    params = init_params(train_images.shape[1], config.hidden_dim, config.mlp_dim, 10, rng)
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    velocities = {name: np.zeros_like(value) for name, value in params.items()}
    history: list[dict[str, float | int]] = []
    step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        step = train_material_epoch(
            params,
            train_images,
            train_labels,
            config,
            scenario,
            rng,
            moments,
            velocities,
            step,
        )
        train_metrics = evaluate_material(params, train_images, train_labels, scenario)
        test_metrics = evaluate_material(params, test_images, test_labels, scenario)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
        )

    final_train = evaluate_material(params, train_images, train_labels, scenario)
    final_test = evaluate_material(params, test_images, test_labels, scenario)
    metrics: dict[str, object] = {
        "scenario": asdict(scenario),
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "elapsed_seconds": time.time() - started,
    }
    return params, (train_images, train_labels, test_images, test_labels), metrics


def material_projection(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    w_mlp_up: np.ndarray,
    detector_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    embed = relu(inputs @ params["w_embed"] + params["b_embed"])
    projection = embed @ w_mlp_up + params["b_mlp_up"]
    if detector_noise > 0.0:
        scale = max(float(np.std(projection)), 1e-8)
        projection = projection + rng.normal(0.0, detector_noise * scale, size=projection.shape)
    return projection.astype(np.float32)


def evaluate_material_projection(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    projection: np.ndarray,
) -> dict[str, object]:
    embed = relu(inputs @ params["w_embed"] + params["b_embed"])
    mlp_up, _ = material_activation_and_derivative(projection, scenario)
    mlp_down_projection = mlp_up @ params["w_mlp_down"] + params["b_mlp_down"]
    residual = relu(embed + mlp_down_projection)
    logits = residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "mlp_up_projection_mean": float(np.mean(projection)),
        "mlp_up_projection_std": float(np.std(projection)),
    }


def chain_forward(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    *,
    w_mlp_up_by_block: tuple[np.ndarray, ...] | None = None,
    detector_noise: float = 0.0,
    rng: np.random.Generator | None = None,
    calibrations: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    noise_rng = np.random.default_rng(0) if rng is None else rng
    residual = relu(inputs @ params["w_embed"] + params["b_embed"])
    projection_mse_values: list[float] = []
    projection_std_values: list[float] = []
    for block_index in range(chain_length):
        weights = (
            params["w_mlp_up"] if w_mlp_up_by_block is None else w_mlp_up_by_block[block_index]
        )
        projection = residual @ weights + params["b_mlp_up"]
        ideal_projection = residual @ params["w_mlp_up"] + params["b_mlp_up"]
        if detector_noise > 0.0:
            scale = max(float(np.std(projection)), 1e-8)
            projection = projection + noise_rng.normal(
                0.0,
                detector_noise * scale,
                size=projection.shape,
            )
        if calibrations is not None:
            gain, bias = calibrations[block_index]
            projection = projection * gain + bias
        projection_mse_values.append(float(np.mean((projection - ideal_projection) ** 2)))
        projection_std_values.append(float(np.std(projection)))
        mlp_up, _ = material_activation_and_derivative(projection, scenario)
        mlp_down_projection = mlp_up @ params["w_mlp_down"] + params["b_mlp_down"]
        residual = relu(residual + mlp_down_projection)
    logits = residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {
        "projection_mse_mean": float(np.mean(projection_mse_values)),
        "projection_mse_max": float(np.max(projection_mse_values)),
        "projection_std_mean": float(np.mean(projection_std_values)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
    }


def evaluate_chain(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    *,
    w_mlp_up_by_block: tuple[np.ndarray, ...] | None = None,
    detector_noise: float = 0.0,
    rng: np.random.Generator | None = None,
    calibrations: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
) -> dict[str, object]:
    probs, diagnostics = chain_forward(
        params,
        inputs,
        scenario,
        chain_length,
        w_mlp_up_by_block=w_mlp_up_by_block,
        detector_noise=detector_noise,
        rng=rng,
        calibrations=calibrations,
    )
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        **diagnostics,
    }


def fit_chain_calibrations(
    params: dict[str, np.ndarray],
    calibration_images: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    w_mlp_up_by_block: tuple[np.ndarray, ...],
    detector_noise: float,
    rng: np.random.Generator,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    ideal_residual = relu(calibration_images @ params["w_embed"] + params["b_embed"])
    observed_residual = ideal_residual.copy()
    calibrations: list[tuple[np.ndarray, np.ndarray]] = []
    for weights in w_mlp_up_by_block:
        ideal_projection = ideal_residual @ params["w_mlp_up"] + params["b_mlp_up"]
        observed_projection = observed_residual @ weights + params["b_mlp_up"]
        if detector_noise > 0.0:
            scale = max(float(np.std(observed_projection)), 1e-8)
            observed_projection = observed_projection + rng.normal(
                0.0,
                detector_noise * scale,
                size=observed_projection.shape,
            )
        gain, bias = fit_affine_calibration(ideal_projection, observed_projection)
        calibrations.append((gain, bias))
        calibrated_projection = observed_projection * gain + bias
        ideal_up, _ = material_activation_and_derivative(ideal_projection, scenario)
        observed_up, _ = material_activation_and_derivative(calibrated_projection, scenario)
        ideal_residual = relu(
            ideal_residual + ideal_up @ params["w_mlp_down"] + params["b_mlp_down"]
        )
        observed_residual = relu(
            observed_residual + observed_up @ params["w_mlp_down"] + params["b_mlp_down"]
        )
    return tuple(calibrations)


def forward_trainable_chain(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
) -> tuple[np.ndarray, dict[str, object]]:
    embed_projection = inputs @ params["w_embed"] + params["b_embed"]
    residual = relu(embed_projection)
    block_caches: list[dict[str, np.ndarray]] = []
    for _ in range(chain_length):
        residual_input = residual
        mlp_up_projection = residual_input @ params["w_mlp_up"] + params["b_mlp_up"]
        mlp_up, mlp_up_derivative = material_activation_and_derivative(
            mlp_up_projection,
            scenario,
        )
        mlp_down_projection = mlp_up @ params["w_mlp_down"] + params["b_mlp_down"]
        residual_pre = residual_input + mlp_down_projection
        residual = relu(residual_pre)
        block_caches.append(
            {
                "residual_input": residual_input,
                "mlp_up_projection": mlp_up_projection,
                "mlp_up": mlp_up,
                "mlp_up_derivative": mlp_up_derivative,
                "mlp_down_projection": mlp_down_projection,
                "residual_pre": residual_pre,
                "residual_output": residual,
            }
        )
    logits = residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {
        "embed_projection": embed_projection,
        "initial_residual": relu(embed_projection),
        "blocks": block_caches,
        "residual": residual,
        "logits": logits,
    }


def evaluate_trainable_chain(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
) -> dict[str, object]:
    probs, cache = forward_trainable_chain(params, inputs, scenario, chain_length)
    predictions = np.argmax(probs, axis=1)
    residual = cache["residual"]
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
    }


def train_chain_epoch(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    config: TransformerMLPConfig,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    rng: np.random.Generator,
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    start_step: int,
) -> int:
    indices = rng.permutation(train_images.shape[0])
    step = start_step
    classes = params["b_head"].shape[0]
    for start in range(0, train_images.shape[0], config.batch_size):
        batch_idx = indices[start : start + config.batch_size]
        x = train_images[batch_idx]
        y = train_labels[batch_idx]
        y_one_hot = one_hot(y, classes)

        probs, cache = forward_trainable_chain(params, x, scenario, chain_length)
        batch_size = x.shape[0]
        d_logits = (probs - y_one_hot) / batch_size
        residual = cache["residual"]
        grad_w_head = residual.T @ d_logits
        grad_b_head = np.sum(d_logits, axis=0)
        d_residual = d_logits @ params["w_head"].T

        grad_w_mlp_up = np.zeros_like(params["w_mlp_up"])
        grad_b_mlp_up = np.zeros_like(params["b_mlp_up"])
        grad_w_mlp_down = np.zeros_like(params["w_mlp_down"])
        grad_b_mlp_down = np.zeros_like(params["b_mlp_down"])

        for block_cache in reversed(cache["blocks"]):  # type: ignore[index]
            residual_pre = block_cache["residual_pre"]
            residual_input = block_cache["residual_input"]
            mlp_up = block_cache["mlp_up"]
            mlp_up_derivative = block_cache["mlp_up_derivative"]
            d_residual_pre = d_residual * (residual_pre > 0.0)
            grad_w_mlp_down += mlp_up.T @ d_residual_pre
            grad_b_mlp_down += np.sum(d_residual_pre, axis=0)
            d_mlp_up = d_residual_pre @ params["w_mlp_down"].T
            d_mlp_up_projection = d_mlp_up * mlp_up_derivative
            grad_w_mlp_up += residual_input.T @ d_mlp_up_projection
            grad_b_mlp_up += np.sum(d_mlp_up_projection, axis=0)
            d_residual = d_residual_pre + d_mlp_up_projection @ params["w_mlp_up"].T

        d_embed_projection = d_residual * (cache["embed_projection"] > 0.0)
        grad_w_embed = x.T @ d_embed_projection
        grad_b_embed = np.sum(d_embed_projection, axis=0)

        grads = {
            "w_embed": grad_w_embed.astype(np.float32),
            "b_embed": grad_b_embed.astype(np.float32),
            "w_mlp_up": grad_w_mlp_up.astype(np.float32),
            "b_mlp_up": grad_b_mlp_up.astype(np.float32),
            "w_mlp_down": grad_w_mlp_down.astype(np.float32),
            "b_mlp_down": grad_b_mlp_down.astype(np.float32),
            "w_head": grad_w_head.astype(np.float32),
            "b_head": grad_b_head.astype(np.float32),
        }
        step += 1
        adam_update(params, grads, moments, velocities, step, config.learning_rate)
    return step


def train_chain_model(
    config: TransformerMLPConfig,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], dict[str, object]]:
    rng = np.random.default_rng(config.seed)
    train_images, train_labels, test_images, test_labels = load_train_and_test(config)
    params = init_params(train_images.shape[1], config.hidden_dim, config.mlp_dim, 10, rng)
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    velocities = {name: np.zeros_like(value) for name, value in params.items()}
    history: list[dict[str, float | int]] = []
    step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        step = train_chain_epoch(
            params,
            train_images,
            train_labels,
            config,
            scenario,
            chain_length,
            rng,
            moments,
            velocities,
            step,
        )
        train_metrics = evaluate_trainable_chain(
            params,
            train_images,
            train_labels,
            scenario,
            chain_length,
        )
        test_metrics = evaluate_trainable_chain(
            params,
            test_images,
            test_labels,
            scenario,
            chain_length,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
        )

    final_train = evaluate_trainable_chain(
        params,
        train_images,
        train_labels,
        scenario,
        chain_length,
    )
    final_test = evaluate_trainable_chain(params, test_images, test_labels, scenario, chain_length)
    metrics: dict[str, object] = {
        "scenario": asdict(scenario),
        "chain_length": chain_length,
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "elapsed_seconds": time.time() - started,
    }
    return params, (train_images, train_labels, test_images, test_labels), metrics


def init_unshared_chain_params(
    input_dim: int,
    hidden_dim: int,
    mlp_dim: int,
    classes: int,
    chain_length: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    params: dict[str, np.ndarray] = {
        "w_embed": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(
            np.float32
        ),
        "b_embed": np.zeros(hidden_dim, dtype=np.float32),
        "w_head": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, classes)).astype(
            np.float32
        ),
        "b_head": np.zeros(classes, dtype=np.float32),
    }
    for block_index in range(chain_length):
        params[f"w_mlp_up_{block_index}"] = rng.normal(
            0.0,
            np.sqrt(2.0 / hidden_dim),
            size=(hidden_dim, mlp_dim),
        ).astype(np.float32)
        params[f"b_mlp_up_{block_index}"] = np.zeros(mlp_dim, dtype=np.float32)
        params[f"w_mlp_down_{block_index}"] = rng.normal(
            0.0,
            np.sqrt(2.0 / mlp_dim),
            size=(mlp_dim, hidden_dim),
        ).astype(np.float32)
        params[f"b_mlp_down_{block_index}"] = np.zeros(hidden_dim, dtype=np.float32)
    return params


def apply_source_coding(
    projection: np.ndarray,
    coding: SourceCodingScenario | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    if coding is None:
        return projection, {
            "source_coding": "none",
            "source_bits": 0,
            "source_redundant_phases": 0,
            "source_quantization_mse": 0.0,
            "source_quantization_max_error": 0.0,
            "source_noise": 0.0,
        }

    encoded_phases: list[np.ndarray] = []
    phase_mses: list[float] = []
    phase_max_errors: list[float] = []
    scale = max(float(np.std(projection)), 1e-8)
    for _ in range(coding.redundant_phases):
        dither = rng.normal(0.0, 0.5 * scale / (2**coding.bits), size=projection.shape)
        quantized, metrics = quantize_symmetric(
            (projection + dither).astype(np.float32),
            coding.bits,
        )
        if coding.source_noise > 0.0:
            quantized = quantized + rng.normal(
                0.0,
                coding.source_noise * scale,
                size=projection.shape,
            ).astype(np.float32)
        encoded_phases.append(quantized.astype(np.float32))
        phase_mses.append(float(metrics.get("mse", 0.0)))
        phase_max_errors.append(float(metrics.get("max_error", 0.0)))

    encoded = np.mean(encoded_phases, axis=0).astype(np.float32)
    return encoded, {
        "source_coding": coding.name,
        "source_bits": coding.bits,
        "source_redundant_phases": coding.redundant_phases,
        "source_quantization_mse": float(np.mean(phase_mses)),
        "source_quantization_max_error": float(np.max(phase_max_errors)),
        "source_noise": coding.source_noise,
    }


def unshared_chain_forward(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    *,
    w_mlp_up_by_block: tuple[np.ndarray, ...] | None = None,
    detector_noise: float = 0.0,
    rng: np.random.Generator | None = None,
    calibrations: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    output_readout_noise: float = 0.0,
    source_coding: SourceCodingScenario | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    noise_rng = np.random.default_rng(0) if rng is None else rng
    embed_projection = inputs @ params["w_embed"] + params["b_embed"]
    residual = relu(embed_projection)
    block_caches: list[dict[str, np.ndarray]] = []
    projection_mse_values: list[float] = []
    source_quantization_mse_values: list[float] = []
    source_quantization_max_error_values: list[float] = []
    for block_index in range(chain_length):
        up_key = f"w_mlp_up_{block_index}"
        weights = params[up_key] if w_mlp_up_by_block is None else w_mlp_up_by_block[block_index]
        ideal_weights = params[up_key]
        projection = residual @ weights + params[f"b_mlp_up_{block_index}"]
        ideal_projection = residual @ ideal_weights + params[f"b_mlp_up_{block_index}"]
        projection, source_metrics = apply_source_coding(
            projection,
            source_coding,
            noise_rng,
        )
        source_quantization_mse_values.append(float(source_metrics["source_quantization_mse"]))
        source_quantization_max_error_values.append(
            float(source_metrics["source_quantization_max_error"])
        )
        if detector_noise > 0.0:
            scale = max(float(np.std(projection)), 1e-8)
            projection = projection + noise_rng.normal(
                0.0,
                detector_noise * scale,
                size=projection.shape,
            )
        if calibrations is not None:
            gain, bias = calibrations[block_index]
            projection = projection * gain + bias
        projection_mse_values.append(float(np.mean((projection - ideal_projection) ** 2)))
        mlp_up, mlp_up_derivative = material_activation_and_derivative(projection, scenario)
        mlp_down_projection = (
            mlp_up @ params[f"w_mlp_down_{block_index}"] + params[f"b_mlp_down_{block_index}"]
        )
        residual_pre = residual + mlp_down_projection
        residual_output = relu(residual_pre)
        block_caches.append(
            {
                "residual_input": residual,
                "mlp_up_projection": projection,
                "mlp_up": mlp_up,
                "mlp_up_derivative": mlp_up_derivative,
                "residual_pre": residual_pre,
                "residual_output": residual_output,
            }
        )
        residual = residual_output
    readout_residual = residual
    if output_readout_noise > 0.0:
        scale = max(float(np.std(readout_residual)), 1e-8)
        readout_residual = readout_residual + noise_rng.normal(
            0.0,
            output_readout_noise * scale,
            size=readout_residual.shape,
        )
    logits = readout_residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {
        "embed_projection": embed_projection,
        "blocks": block_caches,
        "residual": residual,
        "projection_mse_mean": float(np.mean(projection_mse_values)),
        "projection_mse_max": float(np.max(projection_mse_values)),
        "source_quantization_mse_mean": float(np.mean(source_quantization_mse_values)),
        "source_quantization_max_error": float(np.max(source_quantization_max_error_values)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "readout_residual_std": float(np.std(readout_residual)),
    }


def evaluate_unshared_chain(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    *,
    w_mlp_up_by_block: tuple[np.ndarray, ...] | None = None,
    detector_noise: float = 0.0,
    rng: np.random.Generator | None = None,
    calibrations: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None,
    output_readout_noise: float = 0.0,
    source_coding: SourceCodingScenario | None = None,
) -> dict[str, object]:
    probs, diagnostics = unshared_chain_forward(
        params,
        inputs,
        scenario,
        chain_length,
        w_mlp_up_by_block=w_mlp_up_by_block,
        detector_noise=detector_noise,
        rng=rng,
        calibrations=calibrations,
        output_readout_noise=output_readout_noise,
        source_coding=source_coding,
    )
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        **diagnostics,
    }


def fit_unshared_chain_calibrations(
    params: dict[str, np.ndarray],
    calibration_images: np.ndarray,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    w_mlp_up_by_block: tuple[np.ndarray, ...],
    detector_noise: float,
    rng: np.random.Generator,
    source_coding: SourceCodingScenario | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    ideal_residual = relu(calibration_images @ params["w_embed"] + params["b_embed"])
    observed_residual = ideal_residual.copy()
    calibrations: list[tuple[np.ndarray, np.ndarray]] = []
    for block_index in range(chain_length):
        ideal_projection = (
            ideal_residual @ params[f"w_mlp_up_{block_index}"] + params[f"b_mlp_up_{block_index}"]
        )
        observed_projection = (
            observed_residual @ w_mlp_up_by_block[block_index] + params[f"b_mlp_up_{block_index}"]
        )
        observed_projection, _source_metrics = apply_source_coding(
            observed_projection,
            source_coding,
            rng,
        )
        if detector_noise > 0.0:
            scale = max(float(np.std(observed_projection)), 1e-8)
            observed_projection = observed_projection + rng.normal(
                0.0,
                detector_noise * scale,
                size=observed_projection.shape,
            )
        gain, bias = fit_affine_calibration(ideal_projection, observed_projection)
        calibrations.append((gain, bias))
        calibrated_projection = observed_projection * gain + bias
        ideal_up, _ = material_activation_and_derivative(ideal_projection, scenario)
        observed_up, _ = material_activation_and_derivative(calibrated_projection, scenario)
        ideal_residual = relu(
            ideal_residual
            + ideal_up @ params[f"w_mlp_down_{block_index}"]
            + params[f"b_mlp_down_{block_index}"]
        )
        observed_residual = relu(
            observed_residual
            + observed_up @ params[f"w_mlp_down_{block_index}"]
            + params[f"b_mlp_down_{block_index}"]
        )
    return tuple(calibrations)


def train_unshared_chain_epoch(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    config: TransformerMLPConfig,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
    rng: np.random.Generator,
    moments: dict[str, np.ndarray],
    velocities: dict[str, np.ndarray],
    start_step: int,
) -> int:
    indices = rng.permutation(train_images.shape[0])
    step = start_step
    classes = params["b_head"].shape[0]
    for start in range(0, train_images.shape[0], config.batch_size):
        batch_idx = indices[start : start + config.batch_size]
        x = train_images[batch_idx]
        y = train_labels[batch_idx]
        y_one_hot = one_hot(y, classes)

        probs, cache = unshared_chain_forward(params, x, scenario, chain_length)
        batch_size = x.shape[0]
        d_logits = (probs - y_one_hot) / batch_size
        residual = cache["residual"]
        grad_w_head = residual.T @ d_logits
        grad_b_head = np.sum(d_logits, axis=0)
        d_residual = d_logits @ params["w_head"].T
        grads = {
            "w_embed": np.zeros_like(params["w_embed"]),
            "b_embed": np.zeros_like(params["b_embed"]),
            "w_head": grad_w_head.astype(np.float32),
            "b_head": grad_b_head.astype(np.float32),
        }

        for block_index in reversed(range(chain_length)):
            block_cache = cache["blocks"][block_index]  # type: ignore[index]
            residual_pre = block_cache["residual_pre"]
            residual_input = block_cache["residual_input"]
            mlp_up = block_cache["mlp_up"]
            mlp_up_derivative = block_cache["mlp_up_derivative"]
            w_down_key = f"w_mlp_down_{block_index}"
            b_down_key = f"b_mlp_down_{block_index}"
            w_up_key = f"w_mlp_up_{block_index}"
            b_up_key = f"b_mlp_up_{block_index}"
            d_residual_pre = d_residual * (residual_pre > 0.0)
            grads[w_down_key] = (mlp_up.T @ d_residual_pre).astype(np.float32)
            grads[b_down_key] = np.sum(d_residual_pre, axis=0).astype(np.float32)
            d_mlp_up = d_residual_pre @ params[w_down_key].T
            d_mlp_up_projection = d_mlp_up * mlp_up_derivative
            grads[w_up_key] = (residual_input.T @ d_mlp_up_projection).astype(np.float32)
            grads[b_up_key] = np.sum(d_mlp_up_projection, axis=0).astype(np.float32)
            d_residual = d_residual_pre + d_mlp_up_projection @ params[w_up_key].T

        d_embed_projection = d_residual * (cache["embed_projection"] > 0.0)
        grads["w_embed"] = (x.T @ d_embed_projection).astype(np.float32)
        grads["b_embed"] = np.sum(d_embed_projection, axis=0).astype(np.float32)
        step += 1
        adam_update(params, grads, moments, velocities, step, config.learning_rate)
    return step


def train_unshared_chain_model(
    config: TransformerMLPConfig,
    scenario: MaterialNonlinearityScenario,
    chain_length: int,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], dict[str, object]]:
    rng = np.random.default_rng(config.seed)
    train_images, train_labels, test_images, test_labels = load_train_and_test(config)
    params = init_unshared_chain_params(
        train_images.shape[1],
        config.hidden_dim,
        config.mlp_dim,
        10,
        chain_length,
        rng,
    )
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    velocities = {name: np.zeros_like(value) for name, value in params.items()}
    history: list[dict[str, float | int]] = []
    step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        step = train_unshared_chain_epoch(
            params,
            train_images,
            train_labels,
            config,
            scenario,
            chain_length,
            rng,
            moments,
            velocities,
            step,
        )
        train_metrics = evaluate_unshared_chain(
            params,
            train_images,
            train_labels,
            scenario,
            chain_length,
        )
        test_metrics = evaluate_unshared_chain(
            params,
            test_images,
            test_labels,
            scenario,
            chain_length,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
        )
    final_train = evaluate_unshared_chain(
        params,
        train_images,
        train_labels,
        scenario,
        chain_length,
    )
    final_test = evaluate_unshared_chain(params, test_images, test_labels, scenario, chain_length)
    metrics: dict[str, object] = {
        "scenario": asdict(scenario),
        "chain_length": chain_length,
        "history": history,
        "final_train": final_train,
        "final_test": final_test,
        "elapsed_seconds": time.time() - started,
    }
    return params, (train_images, train_labels, test_images, test_labels), metrics


def run_material_training(config: MaterialTrainingConfig) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "material_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=config.transformer.calibration_samples,
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_test = baseline_metrics["baseline_test"]
        baseline_accuracy = float(baseline_test["accuracy"])  # type: ignore[index]
        seed_manifests.append({"seed": seed, "baseline_test": baseline_test})
        for scenario in config.scenarios:
            _, _, metrics = train_material_model(seed_transformer, scenario)
            final_test = metrics["final_test"]
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario.name,
                    "material_family": scenario.material_family,
                    "kind": scenario.kind,
                    "strength": scenario.strength,
                    "baseline_accuracy": baseline_accuracy,
                    "accuracy": float(final_test["accuracy"]),  # type: ignore[index]
                    "accuracy_delta_vs_relu": float(final_test["accuracy"]) - baseline_accuracy,  # type: ignore[index]
                    "loss": float(final_test["loss"]),  # type: ignore[index]
                    "train_accuracy": float(metrics["final_train"]["accuracy"]),  # type: ignore[index]
                    "elapsed_seconds": float(metrics["elapsed_seconds"]),
                }
            )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["scenario", "material_family", "kind"], dropna=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            accuracy_delta_vs_relu_mean=("accuracy_delta_vs_relu", "mean"),
            accuracy_delta_vs_relu_std=("accuracy_delta_vs_relu", "std"),
            loss_mean=("loss", "mean"),
            train_accuracy_mean=("train_accuracy", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    rows_csv = output_dir / "material_training_rows.csv"
    summary_csv = output_dir / "material_training_summary.csv"
    summary_md = output_dir / "material_training_summary.md"
    manifest_path = output_dir / "material_training_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "seeds": list(config.seeds),
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
            "full_split": config.transformer.max_train_samples is None
            and config.transformer.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(frame)),
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


def run_functional_chain_simulation(config: FunctionalChainConfig) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "functional_chain"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_accuracy = float(baseline_metrics["baseline_test"]["accuracy"])  # type: ignore[index]
        seed_manifests.append({"seed": seed, "baseline_test": baseline_metrics["baseline_test"]})
        for scenario in config.scenarios:
            params, data, metrics = train_material_model(seed_transformer, scenario)
            train_images, _, test_images, test_labels = data
            one_block_accuracy = float(metrics["final_test"]["accuracy"])  # type: ignore[index]
            calibration_count = min(config.calibration_samples, train_images.shape[0])
            calibration_images = train_images[:calibration_count]
            for chain_length in config.chain_lengths:
                scenario_rng = np.random.default_rng(
                    scenario_seed(
                        f"functional-chain:{scenario.name}:{chain_length}:"
                        f"{config.constrained_scenario.name}",
                        seed + 170_000,
                    )
                )
                constrained_weights: list[np.ndarray] = []
                weight_mses: list[float] = []
                for block_index in range(chain_length):
                    block_rng = np.random.default_rng(
                        scenario_seed(
                            f"{scenario.name}:{config.constrained_scenario.name}:"
                            f"{chain_length}:{block_index}",
                            seed + 180_000,
                        )
                    )
                    constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
                        params["w_mlp_up"],
                        config.constrained_scenario,
                        block_rng,
                    )
                    constrained_weights.append(constrained_w)
                    weight_mses.append(float(error_metrics["weight_error_mse"]))
                constrained_tuple = tuple(constrained_weights)
                ideal_metrics = evaluate_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                )
                raw_metrics = evaluate_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                    w_mlp_up_by_block=constrained_tuple,
                    detector_noise=config.constrained_scenario.detector_noise,
                    rng=scenario_rng,
                )
                calibration_rng = np.random.default_rng(
                    scenario_seed(
                        f"functional-chain-calibration:{scenario.name}:{chain_length}",
                        seed + 190_000,
                    )
                )
                calibrations = fit_chain_calibrations(
                    params,
                    calibration_images,
                    scenario,
                    constrained_tuple,
                    config.constrained_scenario.detector_noise,
                    calibration_rng,
                )
                calibrated_metrics = evaluate_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                    w_mlp_up_by_block=constrained_tuple,
                    detector_noise=config.constrained_scenario.detector_noise,
                    rng=np.random.default_rng(
                        scenario_seed(
                            f"functional-chain-test-calibrated:{scenario.name}:{chain_length}",
                            seed + 200_000,
                        )
                    ),
                    calibrations=calibrations,
                )
                ideal_accuracy = float(ideal_metrics["accuracy"])
                raw_accuracy = float(raw_metrics["accuracy"])
                calibrated_accuracy = float(calibrated_metrics["accuracy"])
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario.name,
                        "material_family": scenario.material_family,
                        "kind": scenario.kind,
                        "strength": scenario.strength,
                        "chain_length": chain_length,
                        "baseline_relu_accuracy": baseline_accuracy,
                        "one_block_material_accuracy": one_block_accuracy,
                        "ideal_chain_accuracy": ideal_accuracy,
                        "raw_constrained_accuracy": raw_accuracy,
                        "calibrated_constrained_accuracy": calibrated_accuracy,
                        "ideal_chain_delta_vs_one_block": ideal_accuracy - one_block_accuracy,
                        "raw_drop_vs_ideal_chain": ideal_accuracy - raw_accuracy,
                        "calibrated_drop_vs_ideal_chain": ideal_accuracy - calibrated_accuracy,
                        "constrained_scenario": config.constrained_scenario.name,
                        "weight_error_mse_mean": float(np.mean(weight_mses)),
                        "raw_projection_mse_mean": float(raw_metrics["projection_mse_mean"]),
                        "calibrated_projection_mse_mean": float(
                            calibrated_metrics["projection_mse_mean"]
                        ),
                        "ideal_residual_std": float(ideal_metrics["residual_std"]),
                        "calibrated_residual_std": float(calibrated_metrics["residual_std"]),
                    }
                )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(
            ["scenario", "material_family", "kind", "strength", "chain_length"], dropna=False
        )
        .agg(
            ideal_chain_accuracy_mean=("ideal_chain_accuracy", "mean"),
            raw_constrained_accuracy_mean=("raw_constrained_accuracy", "mean"),
            calibrated_constrained_accuracy_mean=("calibrated_constrained_accuracy", "mean"),
            ideal_chain_delta_vs_one_block_mean=("ideal_chain_delta_vs_one_block", "mean"),
            raw_drop_vs_ideal_chain_mean=("raw_drop_vs_ideal_chain", "mean"),
            calibrated_drop_vs_ideal_chain_mean=("calibrated_drop_vs_ideal_chain", "mean"),
            raw_projection_mse_mean=("raw_projection_mse_mean", "mean"),
            calibrated_projection_mse_mean=("calibrated_projection_mse_mean", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    rows_csv = output_dir / "functional_chain_rows.csv"
    summary_csv = output_dir / "functional_chain_summary.csv"
    summary_md = output_dir / "functional_chain_summary.md"
    manifest_path = output_dir / "functional_chain_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "seeds": list(config.seeds),
            "chain_lengths": list(config.chain_lengths),
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
            "calibration_samples": config.calibration_samples,
            "constrained_scenario": asdict(config.constrained_scenario),
            "full_split": config.transformer.max_train_samples is None
            and config.transformer.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(frame)),
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


def run_chained_material_training(config: ChainedMaterialTrainingConfig) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "chained_material_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_accuracy = float(baseline_metrics["baseline_test"]["accuracy"])  # type: ignore[index]
        seed_manifests.append({"seed": seed, "baseline_test": baseline_metrics["baseline_test"]})
        for scenario in config.scenarios:
            for chain_length in config.chain_lengths:
                params, data, metrics = train_chain_model(
                    seed_transformer,
                    scenario,
                    chain_length,
                )
                train_images, _, test_images, test_labels = data
                trained_accuracy = float(metrics["final_test"]["accuracy"])  # type: ignore[index]
                calibration_count = min(config.calibration_samples, train_images.shape[0])
                calibration_images = train_images[:calibration_count]
                constrained_weights: list[np.ndarray] = []
                weight_mses: list[float] = []
                for block_index in range(chain_length):
                    block_rng = np.random.default_rng(
                        scenario_seed(
                            f"chained-training:{scenario.name}:"
                            f"{config.constrained_scenario.name}:"
                            f"{chain_length}:{block_index}",
                            seed + 210_000,
                        )
                    )
                    constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
                        params["w_mlp_up"],
                        config.constrained_scenario,
                        block_rng,
                    )
                    constrained_weights.append(constrained_w)
                    weight_mses.append(float(error_metrics["weight_error_mse"]))
                constrained_tuple = tuple(constrained_weights)
                raw_metrics = evaluate_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                    w_mlp_up_by_block=constrained_tuple,
                    detector_noise=config.constrained_scenario.detector_noise,
                    rng=np.random.default_rng(
                        scenario_seed(
                            f"chained-training-raw:{scenario.name}:{chain_length}",
                            seed + 220_000,
                        )
                    ),
                )
                calibrations = fit_chain_calibrations(
                    params,
                    calibration_images,
                    scenario,
                    constrained_tuple,
                    config.constrained_scenario.detector_noise,
                    np.random.default_rng(
                        scenario_seed(
                            f"chained-training-calibration:{scenario.name}:{chain_length}",
                            seed + 230_000,
                        )
                    ),
                )
                calibrated_metrics = evaluate_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                    w_mlp_up_by_block=constrained_tuple,
                    detector_noise=config.constrained_scenario.detector_noise,
                    rng=np.random.default_rng(
                        scenario_seed(
                            f"chained-training-calibrated:{scenario.name}:{chain_length}",
                            seed + 240_000,
                        )
                    ),
                    calibrations=calibrations,
                )
                raw_accuracy = float(raw_metrics["accuracy"])
                calibrated_accuracy = float(calibrated_metrics["accuracy"])
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario.name,
                        "material_family": scenario.material_family,
                        "kind": scenario.kind,
                        "strength": scenario.strength,
                        "chain_length": chain_length,
                        "baseline_relu_accuracy": baseline_accuracy,
                        "trained_chain_accuracy": trained_accuracy,
                        "raw_constrained_accuracy": raw_accuracy,
                        "calibrated_constrained_accuracy": calibrated_accuracy,
                        "trained_delta_vs_relu": trained_accuracy - baseline_accuracy,
                        "raw_drop_vs_trained": trained_accuracy - raw_accuracy,
                        "calibrated_drop_vs_trained": trained_accuracy - calibrated_accuracy,
                        "constrained_scenario": config.constrained_scenario.name,
                        "weight_error_mse_mean": float(np.mean(weight_mses)),
                        "raw_projection_mse_mean": float(raw_metrics["projection_mse_mean"]),
                        "calibrated_projection_mse_mean": float(
                            calibrated_metrics["projection_mse_mean"]
                        ),
                        "trained_residual_std": float(metrics["final_test"]["residual_std"]),  # type: ignore[index]
                        "calibrated_residual_std": float(calibrated_metrics["residual_std"]),
                        "elapsed_seconds": float(metrics["elapsed_seconds"]),
                    }
                )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(
            ["scenario", "material_family", "kind", "strength", "chain_length"], dropna=False
        )
        .agg(
            trained_chain_accuracy_mean=("trained_chain_accuracy", "mean"),
            raw_constrained_accuracy_mean=("raw_constrained_accuracy", "mean"),
            calibrated_constrained_accuracy_mean=("calibrated_constrained_accuracy", "mean"),
            trained_delta_vs_relu_mean=("trained_delta_vs_relu", "mean"),
            raw_drop_vs_trained_mean=("raw_drop_vs_trained", "mean"),
            calibrated_drop_vs_trained_mean=("calibrated_drop_vs_trained", "mean"),
            raw_projection_mse_mean=("raw_projection_mse_mean", "mean"),
            calibrated_projection_mse_mean=("calibrated_projection_mse_mean", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    rows_csv = output_dir / "chained_material_training_rows.csv"
    summary_csv = output_dir / "chained_material_training_summary.csv"
    summary_md = output_dir / "chained_material_training_summary.md"
    manifest_path = output_dir / "chained_material_training_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "seeds": list(config.seeds),
            "chain_lengths": list(config.chain_lengths),
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
            "calibration_samples": config.calibration_samples,
            "constrained_scenario": asdict(config.constrained_scenario),
            "full_split": config.transformer.max_train_samples is None
            and config.transformer.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(frame)),
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


def physically_drift_unshared_weights(
    params: dict[str, np.ndarray],
    constrained_weights: tuple[np.ndarray, ...],
    chain_length: int,
    drift_rate: float,
    step: int,
    seed: int,
    label: str,
) -> tuple[np.ndarray, ...]:
    drifted: list[np.ndarray] = []
    for block_index in range(chain_length):
        rng = np.random.default_rng(
            scenario_seed(f"{label}:physical-drift:{step}:{block_index}", seed)
        )
        drifted.append(
            drifted_weights(
                constrained_weights[block_index],
                params[f"w_mlp_up_{block_index}"],
                drift_rate,
                step,
                rng,
            )
        )
    return tuple(drifted)


def summarize_unshared_physical_drift_noise(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "scenario",
                "material_family",
                "kind",
                "strength",
                "chain_length",
                "physical_scenario",
                "architecture_profile",
            ],
            dropna=False,
        )
        .agg(
            trained_chain_accuracy_mean=("trained_chain_accuracy", "mean"),
            calibrated_accuracy_mean=("calibrated_accuracy", "mean"),
            calibrated_accuracy_min=("calibrated_accuracy", "min"),
            final_step_accuracy_mean=("final_step_accuracy", "mean"),
            mean_drop_vs_trained=("drop_vs_trained", "mean"),
            worst_drop_vs_trained=("drop_vs_trained", "max"),
            projection_mse_mean=("projection_mse_mean", "mean"),
            source_coding=("source_coding", "first"),
            source_bits=("source_bits", "first"),
            source_redundant_phases=("source_redundant_phases", "first"),
            source_quantization_mse_mean=("source_quantization_mse_mean", "mean"),
            source_dac_pj_per_input=("source_dac_pj_per_input", "first"),
            shared_boundary_source_dac_energy_pj=(
                "shared_boundary_source_dac_energy_pj",
                "first",
            ),
            per_block_source_dac_energy_pj=("per_block_source_dac_energy_pj", "first"),
            source_coding_latency_ns=("source_coding_latency_ns", "first"),
            recalibration_policy=("recalibration_policy", "first"),
            projection_mse_recalibration_threshold=(
                "projection_mse_recalibration_threshold",
                "first",
            ),
            recalibrations_mean=("recalibration_count", "mean"),
            triggered_recalibrations_mean=("triggered_recalibration_count", "mean"),
            triggered_recalibration_steps=("triggered_recalibration", "sum"),
            steps_below_85_percent=("below_85_percent", "sum"),
            runs=("seed", "nunique"),
            rows=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )


def save_unshared_physical_drift_noise_plots(
    frame: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    plots: dict[str, str] = {}
    grouped = (
        frame.groupby(["physical_scenario", "time_step"], dropna=False)
        .agg(
            accuracy_mean=("calibrated_accuracy", "mean"),
            accuracy_min=("calibrated_accuracy", "min"),
            accuracy_max=("calibrated_accuracy", "max"),
            projection_mse_mean=("projection_mse_mean", "mean"),
            projection_mse_min=("projection_mse_mean", "min"),
            projection_mse_max=("projection_mse_mean", "max"),
            recalibration_count=("recalibration_count", "max"),
        )
        .reset_index()
    )

    def save_time_series_plot(
        *,
        metric_prefix: str,
        ylabel: str,
        title: str,
        filename: str,
        threshold: float | None = None,
    ) -> str:
        fig, ax = plt.subplots(figsize=(9.0, 5.0))
        for scenario, scenario_frame in grouped.groupby("physical_scenario", sort=False):
            plot_frame = scenario_frame.sort_values("time_step")
            x = plot_frame["time_step"].to_numpy(dtype=float)
            mean = plot_frame[f"{metric_prefix}_mean"].to_numpy(dtype=float)
            ymin = plot_frame[f"{metric_prefix}_min"].to_numpy(dtype=float)
            ymax = plot_frame[f"{metric_prefix}_max"].to_numpy(dtype=float)
            (line,) = ax.plot(x, mean, marker="o", linewidth=1.8, label=str(scenario))
            ax.fill_between(x, ymin, ymax, color=line.get_color(), alpha=0.12)

            recalibration_steps = plot_frame[plot_frame["recalibration_count"].diff().fillna(1) > 0]
            ax.scatter(
                recalibration_steps["time_step"],
                recalibration_steps[f"{metric_prefix}_mean"],
                marker="s",
                s=42,
                facecolors="white",
                edgecolors=line.get_color(),
                linewidths=1.2,
            )

        if threshold is not None:
            ax.axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.2)
            ax.text(
                0.0,
                threshold,
                f" {threshold:.2f}",
                color="#dc2626",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(title)
        ax.set_xlabel("Simulated time step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return str(path)

    plots["accuracy_time_series"] = save_time_series_plot(
        metric_prefix="accuracy",
        ylabel="Calibrated accuracy",
        title="Unshared 4-block physical drift/noise accuracy",
        filename="unshared_physical_drift_noise_accuracy.png",
        threshold=0.85,
    )
    plots["projection_mse_time_series"] = save_time_series_plot(
        metric_prefix="projection_mse",
        ylabel="Projection MSE",
        title="Unshared 4-block physical drift/noise projection MSE",
        filename="unshared_physical_drift_noise_projection_mse.png",
    )
    return plots


def build_device_tolerance_targets(config: DeviceToleranceTargetConfig) -> pd.DataFrame:
    summary = pd.read_csv(config.physical_summary)
    scenario_meta = {scenario.name: asdict(scenario) for scenario in config.physical_scenarios}
    enriched = summary.copy()
    enriched["passes_accuracy_floor"] = (enriched["calibrated_accuracy_min"] >= 0.85) & (
        enriched["steps_below_85_percent"] == 0
    )
    for field in (
        "thermal_drift_rate",
        "detector_noise",
        "output_readout_noise",
        "recalibration_interval_steps",
    ):
        enriched[field] = enriched["physical_scenario"].map(
            lambda name: scenario_meta[str(name)][field]
        )

    pass_rows = enriched[enriched["passes_accuracy_floor"]].copy()
    warning_rows = enriched[~enriched["passes_accuracy_floor"]].copy()
    projection_mse_warning = float(warning_rows["projection_mse_mean"].min())
    if np.isnan(projection_mse_warning):
        projection_mse_warning = float(enriched["projection_mse_mean"].max())

    rows: list[dict[str, object]] = []
    rows.append(
        {
            "target": "recommended_operating_envelope",
            "max_thermal_drift_rate": float(pass_rows["thermal_drift_rate"].max()),
            "max_detector_noise": float(pass_rows["detector_noise"].max()),
            "max_output_readout_noise": float(pass_rows["output_readout_noise"].max()),
            "max_recalibration_interval_steps": int(
                pass_rows["recalibration_interval_steps"].max()
            ),
            "min_accuracy_floor": 0.85,
            "projection_mse_warning_threshold": projection_mse_warning,
            "status": "pass_in_current_direct_model",
            "evidence": ", ".join(pass_rows["physical_scenario"].astype(str).tolist()),
            "notes": (
                "Use as the near-term target: no seed-level below-85 rows in the "
                "direct physical drift/noise evaluation."
            ),
        }
    )
    rows.append(
        {
            "target": "warning_edge",
            "max_thermal_drift_rate": float(warning_rows["thermal_drift_rate"].max()),
            "max_detector_noise": float(warning_rows["detector_noise"].max()),
            "max_output_readout_noise": float(warning_rows["output_readout_noise"].max()),
            "max_recalibration_interval_steps": int(
                warning_rows["recalibration_interval_steps"].max()
            ),
            "min_accuracy_floor": 0.85,
            "projection_mse_warning_threshold": projection_mse_warning,
            "status": "marginal",
            "evidence": ", ".join(warning_rows["physical_scenario"].astype(str).tolist()),
            "notes": (
                "Means stay above 85%, but at least one seed/time-step row falls below "
                "the floor; treat as a device-design warning edge."
            ),
        }
    )
    rows.append(
        {
            "target": "projection_mse_early_warning",
            "max_thermal_drift_rate": "",
            "max_detector_noise": "",
            "max_output_readout_noise": "",
            "max_recalibration_interval_steps": "",
            "min_accuracy_floor": 0.85,
            "projection_mse_warning_threshold": projection_mse_warning,
            "status": "monitor",
            "evidence": (
                f"first below-floor scenario projection MSE mean ~= {projection_mse_warning:.3f}"
            ),
            "notes": (
                "Escalate or recalibrate more aggressively when projection MSE trends "
                "toward this value before accuracy visibly falls."
            ),
        }
    )
    rows.append(
        {
            "target": "recalibration_cadence",
            "max_thermal_drift_rate": 0.002,
            "max_detector_noise": 0.02,
            "max_output_readout_noise": 0.005,
            "max_recalibration_interval_steps": 4,
            "min_accuracy_floor": 0.85,
            "projection_mse_warning_threshold": projection_mse_warning,
            "status": "recommended",
            "evidence": "warm_cmos_boundary, noisy_near_column_boundary",
            "notes": (
                "Use <=4 simulated steps between recalibrations for drift/noise at or "
                "below the recommended envelope."
            ),
        }
    )
    return pd.DataFrame(rows)


def run_device_tolerance_target_report(
    config: DeviceToleranceTargetConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    targets = build_device_tolerance_targets(config)
    targets_csv = config.output_dir / "device_tolerance_targets.csv"
    targets_md = config.output_dir / "device_tolerance_targets.md"
    manifest_path = config.output_dir / "device_tolerance_targets_manifest.json"
    targets.to_csv(targets_csv, index=False)
    targets_md.write_text(markdown_table(targets), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "physical_summary": str(config.physical_summary),
            "output_dir": str(config.output_dir),
            "physical_scenarios": [asdict(scenario) for scenario in config.physical_scenarios],
        },
        "rows": int(len(targets)),
        "artifacts": {
            "targets_csv": str(targets_csv),
            "targets_md": str(targets_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_unshared_physical_drift_noise(
    config: UnsharedPhysicalDriftNoiseConfig,
) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_accuracy = float(baseline_metrics["baseline_test"]["accuracy"])  # type: ignore[index]
        seed_manifests.append({"seed": seed, "baseline_test": baseline_metrics["baseline_test"]})
        for scenario in config.scenarios:
            params, data, metrics = train_unshared_chain_model(
                seed_transformer,
                scenario,
                config.chain_length,
            )
            train_images, _, test_images, test_labels = data
            trained_accuracy = float(metrics["final_test"]["accuracy"])  # type: ignore[index]
            calibration_count = min(config.calibration_samples, train_images.shape[0])
            calibration_images = train_images[:calibration_count]
            constrained_weights: list[np.ndarray] = []
            constrained_mses: list[float] = []
            for block_index in range(config.chain_length):
                block_rng = np.random.default_rng(
                    scenario_seed(
                        f"physical-unshared:{scenario.name}:"
                        f"{config.constrained_scenario.name}:"
                        f"{config.chain_length}:{block_index}",
                        seed + 290_000,
                    )
                )
                constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
                    params[f"w_mlp_up_{block_index}"],
                    config.constrained_scenario,
                    block_rng,
                )
                constrained_weights.append(constrained_w)
                constrained_mses.append(float(error_metrics["weight_error_mse"]))
            constrained_tuple = tuple(constrained_weights)
            for physical in config.physical_scenarios:
                calibrations: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
                recalibration_count = 0
                triggered_recalibration_count = 0
                scenario_rows: list[dict[str, object]] = []
                for step in range(physical.time_steps):
                    label = f"{scenario.name}:{physical.name}:{config.chain_length}"
                    drifted_tuple = physically_drift_unshared_weights(
                        params,
                        constrained_tuple,
                        config.chain_length,
                        physical.thermal_drift_rate,
                        step,
                        seed + 300_000,
                        label,
                    )
                    scheduled_recalibration = step % physical.recalibration_interval_steps == 0
                    if scheduled_recalibration:
                        calibrations = fit_unshared_chain_calibrations(
                            params,
                            calibration_images,
                            scenario,
                            config.chain_length,
                            drifted_tuple,
                            physical.detector_noise,
                            np.random.default_rng(
                                scenario_seed(
                                    f"physical-calibration:{label}:{step}",
                                    seed + 310_000,
                                )
                            ),
                            source_coding=config.source_coding,
                        )
                        recalibration_count += 1
                    calibrated_metrics = evaluate_unshared_chain(
                        params,
                        test_images,
                        test_labels,
                        scenario,
                        config.chain_length,
                        w_mlp_up_by_block=drifted_tuple,
                        detector_noise=physical.detector_noise,
                        rng=np.random.default_rng(
                            scenario_seed(f"physical-eval:{label}:{step}", seed + 320_000)
                        ),
                        calibrations=calibrations,
                        output_readout_noise=physical.output_readout_noise,
                        source_coding=config.source_coding,
                    )
                    triggered_recalibration = False
                    if (
                        config.projection_mse_recalibration_threshold is not None
                        and not scheduled_recalibration
                        and float(calibrated_metrics["projection_mse_mean"])
                        >= config.projection_mse_recalibration_threshold
                    ):
                        calibrations = fit_unshared_chain_calibrations(
                            params,
                            calibration_images,
                            scenario,
                            config.chain_length,
                            drifted_tuple,
                            physical.detector_noise,
                            np.random.default_rng(
                                scenario_seed(
                                    f"physical-triggered-calibration:{label}:{step}",
                                    seed + 330_000,
                                )
                            ),
                            source_coding=config.source_coding,
                        )
                        recalibration_count += 1
                        triggered_recalibration_count += 1
                        triggered_recalibration = True
                        calibrated_metrics = evaluate_unshared_chain(
                            params,
                            test_images,
                            test_labels,
                            scenario,
                            config.chain_length,
                            w_mlp_up_by_block=drifted_tuple,
                            detector_noise=physical.detector_noise,
                            rng=np.random.default_rng(
                                scenario_seed(
                                    f"physical-triggered-eval:{label}:{step}",
                                    seed + 340_000,
                                )
                            ),
                            calibrations=calibrations,
                            output_readout_noise=physical.output_readout_noise,
                            source_coding=config.source_coding,
                        )
                    accuracy = float(calibrated_metrics["accuracy"])
                    scenario_rows.append(
                        {
                            "seed": seed,
                            "scenario": scenario.name,
                            "material_family": scenario.material_family,
                            "kind": scenario.kind,
                            "strength": scenario.strength,
                            "chain_length": config.chain_length,
                            "physical_scenario": physical.name,
                            "time_step": step,
                            "baseline_relu_accuracy": baseline_accuracy,
                            "trained_chain_accuracy": trained_accuracy,
                            "calibrated_accuracy": accuracy,
                            "drop_vs_trained": trained_accuracy - accuracy,
                            "thermal_drift_rate": physical.thermal_drift_rate,
                            "detector_noise": physical.detector_noise,
                            "output_readout_noise": physical.output_readout_noise,
                            "recalibration_interval_steps": (physical.recalibration_interval_steps),
                            "architecture_profile": physical.architecture_profile,
                            "resonance_shift_pm_per_step": (physical.resonance_shift_pm_per_step),
                            "thermal_excursion_k_per_step": (physical.thermal_excursion_k_per_step),
                            "detector_photons_per_sample": (physical.detector_photons_per_sample),
                            "output_photons_per_sample": (physical.output_photons_per_sample),
                            "output_enob_equivalent": (physical.output_enob_equivalent),
                            "source_coding": (
                                "none"
                                if config.source_coding is None
                                else config.source_coding.name
                            ),
                            "source_bits": (
                                0 if config.source_coding is None else config.source_coding.bits
                            ),
                            "source_redundant_phases": (
                                0
                                if config.source_coding is None
                                else config.source_coding.redundant_phases
                            ),
                            "source_noise": (
                                0.0
                                if config.source_coding is None
                                else config.source_coding.source_noise
                            ),
                            "source_quantization_mse_mean": float(
                                calibrated_metrics["source_quantization_mse_mean"]
                            ),
                            "source_quantization_max_error": float(
                                calibrated_metrics["source_quantization_max_error"]
                            ),
                            "source_dac_pj_per_input": (
                                0.0
                                if config.source_coding is None
                                else config.source_coding.dac_pj_per_input
                            ),
                            "shared_boundary_source_dac_energy_pj": (
                                0.0
                                if config.source_coding is None
                                else config.transformer.hidden_dim
                                * config.source_coding.dac_pj_per_input
                            ),
                            "per_block_source_dac_energy_pj": (
                                0.0
                                if config.source_coding is None
                                else config.chain_length
                                * config.transformer.hidden_dim
                                * config.source_coding.dac_pj_per_input
                            ),
                            "source_coding_latency_ns": (
                                0.0
                                if config.source_coding is None
                                else config.source_coding.latency_ns
                            ),
                            "recalibration_policy": (
                                "scheduled_plus_projection_mse_trigger"
                                if config.projection_mse_recalibration_threshold is not None
                                else "scheduled_only"
                            ),
                            "projection_mse_recalibration_threshold": (
                                0.0
                                if config.projection_mse_recalibration_threshold is None
                                else config.projection_mse_recalibration_threshold
                            ),
                            "scheduled_recalibration": scheduled_recalibration,
                            "triggered_recalibration": triggered_recalibration,
                            "recalibration_count": recalibration_count,
                            "triggered_recalibration_count": (triggered_recalibration_count),
                            "constrained_scenario": config.constrained_scenario.name,
                            "weight_error_mse_mean": float(np.mean(constrained_mses)),
                            "projection_mse_mean": float(calibrated_metrics["projection_mse_mean"]),
                            "projection_mse_max": float(calibrated_metrics["projection_mse_max"]),
                            "residual_std": float(calibrated_metrics["residual_std"]),
                            "readout_residual_std": float(
                                calibrated_metrics["readout_residual_std"]
                            ),
                            "below_85_percent": accuracy < 0.85,
                            "is_final_step": step == physical.time_steps - 1,
                            "elapsed_seconds": float(metrics["elapsed_seconds"]),
                        }
                    )
                final_accuracy = float(scenario_rows[-1]["calibrated_accuracy"])
                for row in scenario_rows:
                    row["final_step_accuracy"] = final_accuracy
                rows.extend(scenario_rows)

    frame = pd.DataFrame(rows)
    summary = summarize_unshared_physical_drift_noise(frame)
    plots = save_unshared_physical_drift_noise_plots(frame, output_dir)
    rows_csv = output_dir / "unshared_physical_drift_noise_rows.csv"
    summary_csv = output_dir / "unshared_physical_drift_noise_summary.csv"
    summary_md = output_dir / "unshared_physical_drift_noise_summary.md"
    manifest_path = output_dir / "unshared_physical_drift_noise_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "seeds": list(config.seeds),
            "chain_length": config.chain_length,
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
            "calibration_samples": config.calibration_samples,
            "constrained_scenario": asdict(config.constrained_scenario),
            "physical_scenarios": [asdict(physical) for physical in config.physical_scenarios],
            "source_coding": (
                None if config.source_coding is None else asdict(config.source_coding)
            ),
            "output_dir_name": config.output_dir_name,
            "projection_mse_recalibration_threshold": (
                config.projection_mse_recalibration_threshold
            ),
            "full_split": config.transformer.max_train_samples is None
            and config.transformer.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(frame)),
        "summary_rows": int(len(summary)),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
            "plots": plots,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def summarize_source_coding_calibration_stress(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(
            [
                "source_coding",
                "source_bits",
                "source_redundant_phases",
                "calibration_samples",
                "recalibration_policy",
                "physical_scenario",
                "architecture_profile",
            ],
            dropna=False,
        )
        .agg(
            calibrated_accuracy_mean=("calibrated_accuracy", "mean"),
            calibrated_accuracy_min=("calibrated_accuracy", "min"),
            final_step_accuracy_mean=("final_step_accuracy", "mean"),
            projection_mse_mean=("projection_mse_mean", "mean"),
            projection_mse_max=("projection_mse_max", "max"),
            source_quantization_mse_mean=("source_quantization_mse_mean", "mean"),
            shared_boundary_source_dac_energy_pj=(
                "shared_boundary_source_dac_energy_pj",
                "first",
            ),
            per_block_source_dac_energy_pj=("per_block_source_dac_energy_pj", "first"),
            source_coding_latency_ns=("source_coding_latency_ns", "first"),
            recalibrations_mean=("recalibration_count", "mean"),
            triggered_recalibrations_mean=("triggered_recalibration_count", "mean"),
            steps_below_85_percent=("below_85_percent", "sum"),
            rows=("seed", "count"),
        )
        .reset_index()
        .sort_values(
            [
                "steps_below_85_percent",
                "calibrated_accuracy_min",
                "per_block_source_dac_energy_pj",
                "source_coding_latency_ns",
            ],
            ascending=[True, False, True, True],
        )
    )


def run_source_coding_calibration_stress(
    config: SourceCodingCalibrationStressConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    slice_dir = config.output_dir / "seed_slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    completed_seed_slices: list[int] = []
    skipped_seed_slices: list[int] = []
    policies: tuple[tuple[str, float | None], ...] = (
        ("scheduled_only", None),
        (
            "scheduled_plus_projection_mse_trigger",
            config.projection_mse_recalibration_threshold,
        ),
    )

    for seed in config.seeds:
        seed_rows_csv = slice_dir / f"seed_{seed}_rows.csv"
        seed_manifest_path = slice_dir / f"seed_{seed}_manifest.json"
        if seed_rows_csv.exists():
            seed_frame = pd.read_csv(seed_rows_csv)
            rows.extend(seed_frame.to_dict("records"))
            if seed_manifest_path.exists():
                seed_manifests.append(json.loads(seed_manifest_path.read_text(encoding="utf-8")))
            else:
                seed_manifests.append({"seed": seed, "baseline_test": None, "resumed": True})
            skipped_seed_slices.append(seed)
            print(
                json.dumps(
                    {
                        "event": "source_coding_stress_seed_skipped",
                        "seed": seed,
                        "rows_csv": str(seed_rows_csv),
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
            continue

        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=config.output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=max(config.calibration_sample_counts),
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_accuracy = float(baseline_metrics["baseline_test"]["accuracy"])  # type: ignore[index]
        seed_manifest = {
            "seed": seed,
            "baseline_test": baseline_metrics["baseline_test"],
            "rows_csv": str(seed_rows_csv),
        }
        seed_rows: list[dict[str, object]] = []
        for scenario in config.transformer_material_scenarios:
            params, data, metrics = train_unshared_chain_model(
                seed_transformer,
                scenario,
                config.chain_length,
            )
            train_images, _, test_images, test_labels = data
            trained_accuracy = float(metrics["final_test"]["accuracy"])  # type: ignore[index]
            constrained_weights: list[np.ndarray] = []
            constrained_mses: list[float] = []
            for block_index in range(config.chain_length):
                block_rng = np.random.default_rng(
                    scenario_seed(
                        f"source-stress:{scenario.name}:"
                        f"{config.constrained_scenario.name}:"
                        f"{config.chain_length}:{block_index}",
                        seed + 290_000,
                    )
                )
                constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
                    params[f"w_mlp_up_{block_index}"],
                    config.constrained_scenario,
                    block_rng,
                )
                constrained_weights.append(constrained_w)
                constrained_mses.append(float(error_metrics["weight_error_mse"]))
            constrained_tuple = tuple(constrained_weights)

            for coding in config.source_codings:
                for calibration_samples in config.calibration_sample_counts:
                    calibration_count = min(calibration_samples, train_images.shape[0])
                    calibration_images = train_images[:calibration_count]
                    for policy_name, threshold in policies:
                        for physical in config.physical_scenarios:
                            calibrations: tuple[tuple[np.ndarray, np.ndarray], ...] | None = None
                            recalibration_count = 0
                            triggered_recalibration_count = 0
                            scenario_rows: list[dict[str, object]] = []
                            for step in range(physical.time_steps):
                                label = (
                                    f"{scenario.name}:{coding.name}:"
                                    f"cal{calibration_samples}:{policy_name}:"
                                    f"{physical.name}:{config.chain_length}"
                                )
                                drifted_tuple = physically_drift_unshared_weights(
                                    params,
                                    constrained_tuple,
                                    config.chain_length,
                                    physical.thermal_drift_rate,
                                    step,
                                    seed + 300_000,
                                    label,
                                )
                                scheduled_recalibration = (
                                    step % physical.recalibration_interval_steps == 0
                                )
                                if scheduled_recalibration:
                                    calibrations = fit_unshared_chain_calibrations(
                                        params,
                                        calibration_images,
                                        scenario,
                                        config.chain_length,
                                        drifted_tuple,
                                        physical.detector_noise,
                                        np.random.default_rng(
                                            scenario_seed(
                                                f"source-stress-cal:{label}:{step}",
                                                seed + 310_000,
                                            )
                                        ),
                                        source_coding=coding,
                                    )
                                    recalibration_count += 1
                                calibrated_metrics = evaluate_unshared_chain(
                                    params,
                                    test_images,
                                    test_labels,
                                    scenario,
                                    config.chain_length,
                                    w_mlp_up_by_block=drifted_tuple,
                                    detector_noise=physical.detector_noise,
                                    rng=np.random.default_rng(
                                        scenario_seed(
                                            f"source-stress-eval:{label}:{step}",
                                            seed + 320_000,
                                        )
                                    ),
                                    calibrations=calibrations,
                                    output_readout_noise=(physical.output_readout_noise),
                                    source_coding=coding,
                                )
                                triggered_recalibration = False
                                if (
                                    threshold is not None
                                    and not scheduled_recalibration
                                    and float(calibrated_metrics["projection_mse_mean"])
                                    >= threshold
                                ):
                                    calibrations = fit_unshared_chain_calibrations(
                                        params,
                                        calibration_images,
                                        scenario,
                                        config.chain_length,
                                        drifted_tuple,
                                        physical.detector_noise,
                                        np.random.default_rng(
                                            scenario_seed(
                                                f"source-stress-triggered-cal:{label}:{step}",
                                                seed + 330_000,
                                            )
                                        ),
                                        source_coding=coding,
                                    )
                                    recalibration_count += 1
                                    triggered_recalibration_count += 1
                                    triggered_recalibration = True
                                    calibrated_metrics = evaluate_unshared_chain(
                                        params,
                                        test_images,
                                        test_labels,
                                        scenario,
                                        config.chain_length,
                                        w_mlp_up_by_block=drifted_tuple,
                                        detector_noise=physical.detector_noise,
                                        rng=np.random.default_rng(
                                            scenario_seed(
                                                f"source-stress-triggered-eval:{label}:{step}",
                                                seed + 340_000,
                                            )
                                        ),
                                        calibrations=calibrations,
                                        output_readout_noise=(physical.output_readout_noise),
                                        source_coding=coding,
                                    )
                                accuracy = float(calibrated_metrics["accuracy"])
                                scenario_rows.append(
                                    {
                                        "seed": seed,
                                        "scenario": scenario.name,
                                        "material_family": scenario.material_family,
                                        "kind": scenario.kind,
                                        "strength": scenario.strength,
                                        "chain_length": config.chain_length,
                                        "physical_scenario": physical.name,
                                        "time_step": step,
                                        "baseline_relu_accuracy": baseline_accuracy,
                                        "trained_chain_accuracy": trained_accuracy,
                                        "calibrated_accuracy": accuracy,
                                        "drop_vs_trained": (trained_accuracy - accuracy),
                                        "thermal_drift_rate": (physical.thermal_drift_rate),
                                        "detector_noise": physical.detector_noise,
                                        "output_readout_noise": (physical.output_readout_noise),
                                        "recalibration_interval_steps": (
                                            physical.recalibration_interval_steps
                                        ),
                                        "architecture_profile": (physical.architecture_profile),
                                        "resonance_shift_pm_per_step": (
                                            physical.resonance_shift_pm_per_step
                                        ),
                                        "thermal_excursion_k_per_step": (
                                            physical.thermal_excursion_k_per_step
                                        ),
                                        "detector_photons_per_sample": (
                                            physical.detector_photons_per_sample
                                        ),
                                        "output_photons_per_sample": (
                                            physical.output_photons_per_sample
                                        ),
                                        "output_enob_equivalent": (physical.output_enob_equivalent),
                                        "source_coding": coding.name,
                                        "source_bits": coding.bits,
                                        "source_redundant_phases": (coding.redundant_phases),
                                        "source_noise": coding.source_noise,
                                        "source_quantization_mse_mean": float(
                                            calibrated_metrics["source_quantization_mse_mean"]
                                        ),
                                        "source_quantization_max_error": float(
                                            calibrated_metrics["source_quantization_max_error"]
                                        ),
                                        "source_dac_pj_per_input": (coding.dac_pj_per_input),
                                        "shared_boundary_source_dac_energy_pj": (
                                            config.transformer.hidden_dim * coding.dac_pj_per_input
                                        ),
                                        "per_block_source_dac_energy_pj": (
                                            config.chain_length
                                            * config.transformer.hidden_dim
                                            * coding.dac_pj_per_input
                                        ),
                                        "source_coding_latency_ns": (coding.latency_ns),
                                        "calibration_samples": calibration_count,
                                        "recalibration_policy": policy_name,
                                        "projection_mse_recalibration_threshold": (
                                            0.0 if threshold is None else threshold
                                        ),
                                        "scheduled_recalibration": (scheduled_recalibration),
                                        "triggered_recalibration": (triggered_recalibration),
                                        "recalibration_count": (recalibration_count),
                                        "triggered_recalibration_count": (
                                            triggered_recalibration_count
                                        ),
                                        "constrained_scenario": (config.constrained_scenario.name),
                                        "weight_error_mse_mean": float(np.mean(constrained_mses)),
                                        "projection_mse_mean": float(
                                            calibrated_metrics["projection_mse_mean"]
                                        ),
                                        "projection_mse_max": float(
                                            calibrated_metrics["projection_mse_max"]
                                        ),
                                        "residual_std": float(calibrated_metrics["residual_std"]),
                                        "readout_residual_std": float(
                                            calibrated_metrics["readout_residual_std"]
                                        ),
                                        "below_85_percent": accuracy < 0.85,
                                        "is_final_step": (step == physical.time_steps - 1),
                                        "elapsed_seconds": float(metrics["elapsed_seconds"]),
                                    }
                                )
                            final_accuracy = float(scenario_rows[-1]["calibrated_accuracy"])
                            for row in scenario_rows:
                                row["final_step_accuracy"] = final_accuracy
                            seed_rows.extend(scenario_rows)

        seed_frame = pd.DataFrame(seed_rows)
        seed_frame.to_csv(seed_rows_csv, index=False)
        seed_manifest["rows"] = int(len(seed_frame))
        seed_manifest_path.write_text(
            json.dumps(seed_manifest, indent=2),
            encoding="utf-8",
        )
        rows.extend(seed_rows)
        seed_manifests.append(seed_manifest)
        completed_seed_slices.append(seed)
        print(
            json.dumps(
                {
                    "event": "source_coding_stress_seed_completed",
                    "seed": seed,
                    "rows": int(len(seed_frame)),
                    "rows_csv": str(seed_rows_csv),
                }
            ),
            file=sys.stderr,
            flush=True,
        )

    frame = pd.DataFrame(rows)
    summary = summarize_source_coding_calibration_stress(frame)
    rows_csv = config.output_dir / "source_coding_calibration_stress_rows.csv"
    summary_csv = config.output_dir / "source_coding_calibration_stress_summary.csv"
    summary_md = config.output_dir / "source_coding_calibration_stress_summary.md"
    manifest_path = config.output_dir / "source_coding_calibration_stress_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    passing = summary[summary["steps_below_85_percent"] == 0]
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "output_dir": str(config.output_dir),
            "seeds": list(config.seeds),
            "chain_length": config.chain_length,
            "scenarios": [asdict(scenario) for scenario in config.transformer_material_scenarios],
            "constrained_scenario": asdict(config.constrained_scenario),
            "calibration_sample_counts": list(config.calibration_sample_counts),
            "source_codings": [asdict(coding) for coding in config.source_codings],
            "physical_scenarios": [asdict(physical) for physical in config.physical_scenarios],
            "projection_mse_recalibration_threshold": (
                config.projection_mse_recalibration_threshold
            ),
            "slice_dir": str(slice_dir),
        },
        "rows": int(len(frame)),
        "summary_rows": int(len(summary)),
        "passing_summary_rows": int(len(passing)),
        "lowest_passing_source_bits": (
            None if passing.empty else int(passing["source_bits"].min())
        ),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
            "slice_dir": str(slice_dir),
        },
        "seed_manifests": seed_manifests,
        "completed_seed_slices": completed_seed_slices,
        "skipped_seed_slices": skipped_seed_slices,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_unshared_chained_material_training(
    config: UnsharedChainedMaterialTrainingConfig,
) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "unshared_chained_material_training"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_accuracy = float(baseline_metrics["baseline_test"]["accuracy"])  # type: ignore[index]
        seed_manifests.append({"seed": seed, "baseline_test": baseline_metrics["baseline_test"]})
        for scenario in config.scenarios:
            for chain_length in config.chain_lengths:
                params, data, metrics = train_unshared_chain_model(
                    seed_transformer,
                    scenario,
                    chain_length,
                )
                train_images, _, test_images, test_labels = data
                trained_accuracy = float(metrics["final_test"]["accuracy"])  # type: ignore[index]
                calibration_count = min(config.calibration_samples, train_images.shape[0])
                calibration_images = train_images[:calibration_count]
                constrained_weights: list[np.ndarray] = []
                weight_mses: list[float] = []
                for block_index in range(chain_length):
                    block_rng = np.random.default_rng(
                        scenario_seed(
                            f"unshared-chain:{scenario.name}:"
                            f"{config.constrained_scenario.name}:"
                            f"{chain_length}:{block_index}",
                            seed + 250_000,
                        )
                    )
                    constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
                        params[f"w_mlp_up_{block_index}"],
                        config.constrained_scenario,
                        block_rng,
                    )
                    constrained_weights.append(constrained_w)
                    weight_mses.append(float(error_metrics["weight_error_mse"]))
                constrained_tuple = tuple(constrained_weights)
                raw_metrics = evaluate_unshared_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                    w_mlp_up_by_block=constrained_tuple,
                    detector_noise=config.constrained_scenario.detector_noise,
                    rng=np.random.default_rng(
                        scenario_seed(
                            f"unshared-chain-raw:{scenario.name}:{chain_length}",
                            seed + 260_000,
                        )
                    ),
                )
                calibrations = fit_unshared_chain_calibrations(
                    params,
                    calibration_images,
                    scenario,
                    chain_length,
                    constrained_tuple,
                    config.constrained_scenario.detector_noise,
                    np.random.default_rng(
                        scenario_seed(
                            f"unshared-chain-calibration:{scenario.name}:{chain_length}",
                            seed + 270_000,
                        )
                    ),
                )
                calibrated_metrics = evaluate_unshared_chain(
                    params,
                    test_images,
                    test_labels,
                    scenario,
                    chain_length,
                    w_mlp_up_by_block=constrained_tuple,
                    detector_noise=config.constrained_scenario.detector_noise,
                    rng=np.random.default_rng(
                        scenario_seed(
                            f"unshared-chain-calibrated:{scenario.name}:{chain_length}",
                            seed + 280_000,
                        )
                    ),
                    calibrations=calibrations,
                )
                raw_accuracy = float(raw_metrics["accuracy"])
                calibrated_accuracy = float(calibrated_metrics["accuracy"])
                rows.append(
                    {
                        "seed": seed,
                        "scenario": scenario.name,
                        "material_family": scenario.material_family,
                        "kind": scenario.kind,
                        "strength": scenario.strength,
                        "chain_length": chain_length,
                        "baseline_relu_accuracy": baseline_accuracy,
                        "trained_chain_accuracy": trained_accuracy,
                        "raw_constrained_accuracy": raw_accuracy,
                        "calibrated_constrained_accuracy": calibrated_accuracy,
                        "trained_delta_vs_relu": trained_accuracy - baseline_accuracy,
                        "raw_drop_vs_trained": trained_accuracy - raw_accuracy,
                        "calibrated_drop_vs_trained": trained_accuracy - calibrated_accuracy,
                        "constrained_scenario": config.constrained_scenario.name,
                        "weight_error_mse_mean": float(np.mean(weight_mses)),
                        "raw_projection_mse_mean": float(raw_metrics["projection_mse_mean"]),
                        "calibrated_projection_mse_mean": float(
                            calibrated_metrics["projection_mse_mean"]
                        ),
                        "trained_residual_std": float(metrics["final_test"]["residual_std"]),  # type: ignore[index]
                        "calibrated_residual_std": float(calibrated_metrics["residual_std"]),
                        "elapsed_seconds": float(metrics["elapsed_seconds"]),
                    }
                )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(
            ["scenario", "material_family", "kind", "strength", "chain_length"], dropna=False
        )
        .agg(
            trained_chain_accuracy_mean=("trained_chain_accuracy", "mean"),
            raw_constrained_accuracy_mean=("raw_constrained_accuracy", "mean"),
            calibrated_constrained_accuracy_mean=("calibrated_constrained_accuracy", "mean"),
            trained_delta_vs_relu_mean=("trained_delta_vs_relu", "mean"),
            raw_drop_vs_trained_mean=("raw_drop_vs_trained", "mean"),
            calibrated_drop_vs_trained_mean=("calibrated_drop_vs_trained", "mean"),
            raw_projection_mse_mean=("raw_projection_mse_mean", "mean"),
            calibrated_projection_mse_mean=("calibrated_projection_mse_mean", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    rows_csv = output_dir / "unshared_chained_material_training_rows.csv"
    summary_csv = output_dir / "unshared_chained_material_training_summary.csv"
    summary_md = output_dir / "unshared_chained_material_training_summary.md"
    manifest_path = output_dir / "unshared_chained_material_training_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "seeds": list(config.seeds),
            "chain_lengths": list(config.chain_lengths),
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
            "calibration_samples": config.calibration_samples,
            "constrained_scenario": asdict(config.constrained_scenario),
            "full_split": config.transformer.max_train_samples is None
            and config.transformer.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(frame)),
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


def run_material_strength_sweep(config: MaterialStrengthSweepConfig) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "material_strength_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    scenarios = strength_sweep_scenarios(config.strengths)
    for seed in config.seeds:
        seed_transformer = TransformerMLPConfig(
            data_dir=config.transformer.data_dir,
            dataset=config.transformer.dataset,
            output_dir=output_dir / f"seed_{seed}" / "transformer_mlp",
            hidden_dim=config.transformer.hidden_dim,
            mlp_dim=config.transformer.mlp_dim,
            epochs=config.transformer.epochs,
            batch_size=config.transformer.batch_size,
            learning_rate=config.transformer.learning_rate,
            seed=seed,
            max_train_samples=config.transformer.max_train_samples,
            max_test_samples=config.transformer.max_test_samples,
            synthetic=config.transformer.synthetic,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.transformer.adaptation_epochs,
            adaptation_learning_rate=config.transformer.adaptation_learning_rate,
        )
        _, _, _, _, _, baseline_metrics = train_transformer(seed_transformer)
        baseline_accuracy = float(baseline_metrics["baseline_test"]["accuracy"])  # type: ignore[index]
        seed_manifests.append({"seed": seed, "baseline_test": baseline_metrics["baseline_test"]})
        for scenario in scenarios:
            params, data, metrics = train_material_model(seed_transformer, scenario)
            train_images, _, test_images, test_labels = data
            ideal_accuracy = float(metrics["final_test"]["accuracy"])  # type: ignore[index]
            calibration_count = min(config.calibration_samples, train_images.shape[0])
            calibration_images = train_images[:calibration_count]
            scenario_rng = np.random.default_rng(
                scenario_seed(f"{scenario.name}:{config.constrained_scenario.name}", seed + 130_000)
            )
            constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
                params["w_mlp_up"],
                config.constrained_scenario,
                scenario_rng,
            )
            ideal_test_projection = material_projection(
                params,
                test_images,
                params["w_mlp_up"],
                0.0,
                scenario_rng,
            )
            raw_test_projection = material_projection(
                params,
                test_images,
                constrained_w,
                config.constrained_scenario.detector_noise,
                scenario_rng,
            )
            ideal_calibration = material_projection(
                params,
                calibration_images,
                params["w_mlp_up"],
                0.0,
                scenario_rng,
            )
            observed_calibration = material_projection(
                params,
                calibration_images,
                constrained_w,
                config.constrained_scenario.detector_noise,
                scenario_rng,
            )
            gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
            calibrated_test_projection = raw_test_projection * gain + bias
            raw_metrics = evaluate_material_projection(
                params,
                test_images,
                test_labels,
                scenario,
                raw_test_projection,
            )
            calibrated_metrics = evaluate_material_projection(
                params,
                test_images,
                test_labels,
                scenario,
                calibrated_test_projection,
            )
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario.name,
                    "material_family": scenario.material_family,
                    "kind": scenario.kind,
                    "strength": scenario.strength,
                    "baseline_relu_accuracy": baseline_accuracy,
                    "ideal_material_accuracy": ideal_accuracy,
                    "ideal_delta_vs_relu": ideal_accuracy - baseline_accuracy,
                    "constrained_raw_accuracy": float(raw_metrics["accuracy"]),
                    "constrained_calibrated_accuracy": float(calibrated_metrics["accuracy"]),
                    "raw_drop_vs_ideal": ideal_accuracy - float(raw_metrics["accuracy"]),
                    "calibrated_drop_vs_ideal": ideal_accuracy
                    - float(calibrated_metrics["accuracy"]),
                    "constrained_scenario": config.constrained_scenario.name,
                    "weight_error_mse": error_metrics["weight_error_mse"],
                    "fan_pressure": error_metrics["routing"]["fan_pressure"],
                    "effective_loss": error_metrics["loss"]["effective"],
                    "effective_crosstalk": error_metrics["geometry"]["effective_crosstalk"],
                    "projection_mse_raw": float(
                        np.mean((raw_test_projection - ideal_test_projection) ** 2)
                    ),
                    "projection_mse_calibrated": float(
                        np.mean((calibrated_test_projection - ideal_test_projection) ** 2)
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(["scenario", "material_family", "kind", "strength"], dropna=False)
        .agg(
            ideal_material_accuracy_mean=("ideal_material_accuracy", "mean"),
            ideal_delta_vs_relu_mean=("ideal_delta_vs_relu", "mean"),
            constrained_raw_accuracy_mean=("constrained_raw_accuracy", "mean"),
            constrained_calibrated_accuracy_mean=("constrained_calibrated_accuracy", "mean"),
            raw_drop_vs_ideal_mean=("raw_drop_vs_ideal", "mean"),
            calibrated_drop_vs_ideal_mean=("calibrated_drop_vs_ideal", "mean"),
            projection_mse_raw_mean=("projection_mse_raw", "mean"),
            projection_mse_calibrated_mean=("projection_mse_calibrated", "mean"),
            runs=("seed", "count"),
        )
        .reset_index()
        .fillna(0.0)
    )
    rows_csv = output_dir / "material_strength_sweep_rows.csv"
    summary_csv = output_dir / "material_strength_sweep_summary.csv"
    summary_md = output_dir / "material_strength_sweep_summary.md"
    manifest_path = output_dir / "material_strength_sweep_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "transformer": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.transformer).items()
            },
            "seeds": list(config.seeds),
            "strengths": list(config.strengths),
            "scenarios": [asdict(scenario) for scenario in scenarios],
            "calibration_samples": config.calibration_samples,
            "constrained_scenario": asdict(config.constrained_scenario),
            "full_split": config.transformer.max_train_samples is None
            and config.transformer.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(frame)),
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
