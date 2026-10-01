"""Fixed fabricated optical projection error models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from optical_spike.baseline import BaselineConfig, run_baseline
from optical_spike.quantization import quantize_symmetric
from optical_spike.tunable import (
    compare_projection,
    evaluate_projection,
    fit_affine_calibration,
    load_train_and_test,
    project,
)


@dataclass(frozen=True)
class FixedScenario:
    name: str
    bits: int
    fabrication_error: float
    optical_loss: float
    alignment_error: float
    crosstalk: float
    detector_noise: float


@dataclass(frozen=True)
class FixedSweepConfig:
    baseline: BaselineConfig
    calibration_samples: int = 128
    scenarios: tuple[FixedScenario, ...] = (
        FixedScenario("near_ideal_fixed", 8, 0.001, 0.005, 0.002, 0.001, 0.002),
        FixedScenario("mild_fixed", 8, 0.005, 0.01, 0.005, 0.002, 0.005),
        FixedScenario("moderate_fixed", 6, 0.01, 0.03, 0.01, 0.01, 0.01),
        FixedScenario("severe_fixed", 4, 0.03, 0.08, 0.03, 0.03, 0.02),
    )


def apply_fixed_weight_errors(
    weights: np.ndarray,
    scenario: FixedScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    quantized, quantization_metrics = quantize_symmetric(weights, scenario.bits)
    max_abs = max(float(np.max(np.abs(weights))), 1e-8)

    fabrication_noise = rng.normal(
        0.0,
        scenario.fabrication_error * max_abs,
        size=weights.shape,
    ).astype(np.float32)
    fabricated = quantized + fabrication_noise

    global_loss = max(0.0, 1.0 - scenario.optical_loss)
    input_alignment = rng.normal(
        1.0,
        scenario.alignment_error,
        size=(weights.shape[0], 1),
    ).astype(np.float32)
    output_alignment = rng.normal(
        1.0,
        scenario.alignment_error,
        size=(1, weights.shape[1]),
    ).astype(np.float32)
    fabricated = fabricated * global_loss * input_alignment * output_alignment

    if scenario.crosstalk > 0.0:
        mixing = rng.normal(0.0, scenario.crosstalk, size=(weights.shape[1], weights.shape[1])).astype(
            np.float32
        )
        np.fill_diagonal(mixing, 1.0)
        fabricated = fabricated @ mixing

    error = fabricated - weights
    return fabricated.astype(np.float32), {
        "quantization": quantization_metrics,
        "weight_error_mse": float(np.mean(error * error)),
        "weight_error_max": float(np.max(np.abs(error))),
        "global_loss": float(global_loss),
        "input_alignment_std": float(np.std(input_alignment)),
        "output_alignment_std": float(np.std(output_alignment)),
    }


def run_fixed_error_sweeps(config: FixedSweepConfig) -> dict[str, object]:
    baseline_metrics = run_baseline(config.baseline)
    weights_path = Path(str(baseline_metrics["artifacts"]["weights"]))
    weights = np.load(weights_path)
    params = {name: weights[name].astype(np.float32) for name in weights.files}
    train_images, _, test_images, test_labels = load_train_and_test(config.baseline)

    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]

    rng = np.random.default_rng(config.baseline.seed + 2000)
    ideal_test_projection = project(params, test_images, params["w_opt"], rng, detector_noise=0.0)
    baseline_eval = evaluate_projection(params, test_images, test_labels, ideal_test_projection)

    sweeps: list[dict[str, object]] = []
    for scenario in config.scenarios:
        name_seed = sum((index + 1) * ord(char) for index, char in enumerate(scenario.name))
        scenario_rng = np.random.default_rng(config.baseline.seed + 20_000 + name_seed)
        fabricated_w, error_metrics = apply_fixed_weight_errors(
            params["w_opt"],
            scenario,
            scenario_rng,
        )
        raw_projection = project(params, test_images, fabricated_w, scenario_rng, scenario.detector_noise)
        raw_metrics = evaluate_projection(params, test_images, test_labels, raw_projection)

        ideal_calibration = project(params, calibration_images, params["w_opt"], scenario_rng, 0.0)
        observed_calibration = project(
            params,
            calibration_images,
            fabricated_w,
            scenario_rng,
            scenario.detector_noise,
        )
        gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
        calibrated_projection = raw_projection * gain + bias
        calibrated_metrics = evaluate_projection(
            params,
            test_images,
            test_labels,
            calibrated_projection,
        )

        sweeps.append(
            {
                "scenario": asdict(scenario),
                "error_model": error_metrics,
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
                "raw_accuracy_drop": baseline_eval["accuracy"] - raw_metrics["accuracy"],
                "calibrated_accuracy_drop": baseline_eval["accuracy"]
                - calibrated_metrics["accuracy"],
                "raw_diagnostics": compare_projection(
                    params,
                    test_labels,
                    ideal_test_projection,
                    raw_projection,
                ),
                "calibrated_diagnostics": compare_projection(
                    params,
                    test_labels,
                    ideal_test_projection,
                    calibrated_projection,
                ),
                "calibration": {
                    "samples": calibration_count,
                    "gain_mean": float(np.mean(gain)),
                    "gain_std": float(np.std(gain)),
                    "bias_mean": float(np.mean(bias)),
                    "bias_std": float(np.std(bias)),
                },
            }
        )

    output_dir = config.baseline.output_dir.parent / "fixed_errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "fixed_error_sweeps.json"
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
