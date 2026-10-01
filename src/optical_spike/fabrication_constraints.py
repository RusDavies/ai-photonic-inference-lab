"""Stronger fixed-fabrication constraints for optical matrix targets."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from optical_spike.quantization import quantize_symmetric


@dataclass(frozen=True)
class ConstrainedFixedScenario:
    name: str
    bits: int
    fabrication_error: float
    optical_loss_budget: float
    base_optical_loss: float
    loss_per_extra_tile: float
    fan_in_limit: int
    fan_out_limit: int
    geometry_crosstalk: float
    detector_noise: float
    differential_noise: float


DEFAULT_CONSTRAINED_FIXED_SCENARIOS: tuple[ConstrainedFixedScenario, ...] = (
    ConstrainedFixedScenario(
        "signed_tiled_mild",
        8,
        0.005,
        0.06,
        0.02,
        0.01,
        128,
        256,
        0.002,
        0.005,
        0.002,
    ),
    ConstrainedFixedScenario(
        "signed_tiled_moderate",
        6,
        0.01,
        0.08,
        0.04,
        0.02,
        64,
        128,
        0.008,
        0.01,
        0.006,
    ),
    ConstrainedFixedScenario(
        "signed_tiled_severe",
        4,
        0.025,
        0.10,
        0.07,
        0.035,
        32,
        64,
        0.02,
        0.02,
        0.015,
    ),
)


def geometry_mixing(size: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if scale <= 0.0:
        return np.eye(size, dtype=np.float32)
    indices = np.arange(size)
    distance = np.abs(indices[:, None] - indices[None, :]).astype(np.float32)
    decay = 1.0 / (1.0 + distance)
    noise = rng.normal(0.0, scale, size=(size, size)).astype(np.float32) * decay
    np.fill_diagonal(noise, 0.0)
    return (np.eye(size, dtype=np.float32) + noise).astype(np.float32)


def apply_constrained_fixed_weight_errors(
    weights: np.ndarray,
    scenario: ConstrainedFixedScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    quantized, quantization_metrics = quantize_symmetric(weights, scenario.bits)
    max_abs = max(float(np.max(np.abs(weights))), 1e-8)

    positive_path = np.maximum(quantized, 0.0)
    negative_path = np.maximum(-quantized, 0.0)
    path_noise_scale = scenario.fabrication_error * max_abs
    positive_path = positive_path + rng.normal(0.0, path_noise_scale, size=weights.shape).astype(
        np.float32
    )
    negative_path = negative_path + rng.normal(0.0, path_noise_scale, size=weights.shape).astype(
        np.float32
    )
    positive_path *= rng.normal(1.0, scenario.differential_noise, size=weights.shape).astype(
        np.float32
    )
    negative_path *= rng.normal(1.0, scenario.differential_noise, size=weights.shape).astype(
        np.float32
    )
    constrained = positive_path - negative_path

    fan_in_tiles = max(1, math.ceil(weights.shape[0] / scenario.fan_in_limit))
    fan_out_tiles = max(1, math.ceil(weights.shape[1] / scenario.fan_out_limit))
    extra_tiles = max(0, fan_in_tiles + fan_out_tiles - 2)
    required_loss = scenario.base_optical_loss + scenario.loss_per_extra_tile * extra_tiles
    budget_excess = max(0.0, required_loss - scenario.optical_loss_budget)
    effective_loss = min(0.95, required_loss + budget_excess)
    constrained *= 1.0 - effective_loss

    fan_pressure = max(
        weights.shape[0] / scenario.fan_in_limit,
        weights.shape[1] / scenario.fan_out_limit,
        1.0,
    )
    crosstalk_scale = scenario.geometry_crosstalk * fan_pressure
    constrained = constrained @ geometry_mixing(weights.shape[1], crosstalk_scale, rng)

    error = constrained - weights
    return constrained.astype(np.float32), {
        "quantization": quantization_metrics,
        "signed_encoding": {
            "positive_path_nonzero": int(np.count_nonzero(positive_path)),
            "negative_path_nonzero": int(np.count_nonzero(negative_path)),
            "differential_noise": scenario.differential_noise,
        },
        "routing": {
            "fan_in_limit": scenario.fan_in_limit,
            "fan_out_limit": scenario.fan_out_limit,
            "fan_in_tiles": fan_in_tiles,
            "fan_out_tiles": fan_out_tiles,
            "extra_tiles": extra_tiles,
            "fan_pressure": float(fan_pressure),
        },
        "loss": {
            "budget": scenario.optical_loss_budget,
            "required": float(required_loss),
            "budget_excess": float(budget_excess),
            "effective": float(effective_loss),
        },
        "geometry": {
            "base_crosstalk": scenario.geometry_crosstalk,
            "effective_crosstalk": float(crosstalk_scale),
        },
        "weight_error_mse": float(np.mean(error * error)),
        "weight_error_max": float(np.max(np.abs(error))),
    }
