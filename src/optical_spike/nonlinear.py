"""Nonlinear optical projection variants and compensation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from optical_spike.baseline import BaselineConfig, run_baseline
from optical_spike.tunable import compare_projection, evaluate_projection, load_train_and_test, project


@dataclass(frozen=True)
class NonlinearScenario:
    name: str
    kind: str
    strength: float


@dataclass(frozen=True)
class NonlinearSweepConfig:
    baseline: BaselineConfig
    calibration_samples: int = 128
    scenarios: tuple[NonlinearScenario, ...] = (
        NonlinearScenario("saturation_mild", "saturation", 0.25),
        NonlinearScenario("saturation_strong", "saturation", 0.75),
        NonlinearScenario("detector_clipping_mild", "clipping", 2.5),
        NonlinearScenario("detector_clipping_strong", "clipping", 1.25),
        NonlinearScenario("intensity_distortion_mild", "intensity_distortion", 0.10),
        NonlinearScenario("intensity_distortion_strong", "intensity_distortion", 0.30),
    )


def apply_nonlinearity(projection: np.ndarray, scenario: NonlinearScenario) -> np.ndarray:
    scale = max(float(np.std(projection)), 1e-8)
    if scenario.kind == "saturation":
        distorted = projection / (1.0 + scenario.strength * np.abs(projection) / scale)
    elif scenario.kind == "clipping":
        limit = scenario.strength * scale
        distorted = np.clip(projection, -limit, limit)
    elif scenario.kind == "intensity_distortion":
        distorted = projection + scenario.strength * projection * np.abs(projection) / scale
    else:
        raise ValueError(f"Unknown nonlinearity kind: {scenario.kind}")
    return distorted.astype(np.float32)


def fit_polynomial_compensator(
    observed_projection: np.ndarray,
    ideal_projection: np.ndarray,
) -> np.ndarray:
    channels = observed_projection.shape[1]
    coefficients = np.zeros((channels, 3), dtype=np.float32)
    for channel in range(channels):
        observed = observed_projection[:, channel]
        design = np.column_stack([observed, observed * observed, np.ones_like(observed)])
        solution, *_ = np.linalg.lstsq(design, ideal_projection[:, channel], rcond=None)
        coefficients[channel] = solution.astype(np.float32)
    return coefficients


def apply_polynomial_compensator(
    observed_projection: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    linear = observed_projection * coefficients[:, 0]
    quadratic = observed_projection * observed_projection * coefficients[:, 1]
    bias = coefficients[:, 2]
    return (linear + quadratic + bias).astype(np.float32)


def run_nonlinear_variant_sweeps(config: NonlinearSweepConfig) -> dict[str, object]:
    baseline_metrics = run_baseline(config.baseline)
    weights_path = Path(str(baseline_metrics["artifacts"]["weights"]))
    weights = np.load(weights_path)
    params = {name: weights[name].astype(np.float32) for name in weights.files}
    train_images, _, test_images, test_labels = load_train_and_test(config.baseline)

    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]

    rng = np.random.default_rng(config.baseline.seed + 3000)
    ideal_test_projection = project(params, test_images, params["w_opt"], rng, detector_noise=0.0)
    ideal_calibration = project(params, calibration_images, params["w_opt"], rng, detector_noise=0.0)
    baseline_eval = evaluate_projection(params, test_images, test_labels, ideal_test_projection)

    sweeps: list[dict[str, object]] = []
    for scenario in config.scenarios:
        raw_test_projection = apply_nonlinearity(ideal_test_projection, scenario)
        raw_calibration_projection = apply_nonlinearity(ideal_calibration, scenario)
        raw_metrics = evaluate_projection(params, test_images, test_labels, raw_test_projection)

        coefficients = fit_polynomial_compensator(raw_calibration_projection, ideal_calibration)
        compensated_projection = apply_polynomial_compensator(raw_test_projection, coefficients)
        compensated_metrics = evaluate_projection(
            params,
            test_images,
            test_labels,
            compensated_projection,
        )

        raw_error = raw_test_projection - ideal_test_projection
        compensated_error = compensated_projection - ideal_test_projection
        sweeps.append(
            {
                "scenario": asdict(scenario),
                "raw_metrics": raw_metrics,
                "compensated_metrics": compensated_metrics,
                "raw_accuracy_drop": baseline_eval["accuracy"] - raw_metrics["accuracy"],
                "compensated_accuracy_drop": baseline_eval["accuracy"]
                - compensated_metrics["accuracy"],
                "raw_diagnostics": compare_projection(
                    params,
                    test_labels,
                    ideal_test_projection,
                    raw_test_projection,
                ),
                "compensated_diagnostics": compare_projection(
                    params,
                    test_labels,
                    ideal_test_projection,
                    compensated_projection,
                ),
                "projection_error": {
                    "raw_mse": float(np.mean(raw_error * raw_error)),
                    "compensated_mse": float(np.mean(compensated_error * compensated_error)),
                    "raw_max_error": float(np.max(np.abs(raw_error))),
                    "compensated_max_error": float(np.max(np.abs(compensated_error))),
                },
                "compensator": {
                    "samples": calibration_count,
                    "linear_mean": float(np.mean(coefficients[:, 0])),
                    "quadratic_mean": float(np.mean(coefficients[:, 1])),
                    "bias_mean": float(np.mean(coefficients[:, 2])),
                },
            }
        )

    output_dir = config.baseline.output_dir.parent / "nonlinear_variants"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "nonlinear_variant_sweeps.json"
    payload: dict[str, object] = {
        "config": {
            "baseline": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.baseline).items()
            },
            "calibration_samples": config.calibration_samples,
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
        },
        "baseline": baseline_metrics,
        "baseline_eval": baseline_eval,
        "sweeps": sweeps,
        "artifacts": {
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
