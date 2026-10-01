"""Tunable optical projection error models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from optical_spike.baseline import (
    BaselineConfig,
    cross_entropy,
    limit_samples,
    relu,
    run_baseline,
    softmax,
)
from optical_spike.data import load_cifar10, load_fashion_mnist, make_synthetic_dataset
from optical_spike.diagnostics import (
    confusion_matrix,
    prediction_delta,
    projection_diagnostics,
)
from optical_spike.quantization import quantize_symmetric


@dataclass(frozen=True)
class TunableScenario:
    name: str
    bits: int
    setting_error: float
    gain_error: float
    crosstalk: float
    detector_noise: float
    drift: float


@dataclass(frozen=True)
class TunableSweepConfig:
    baseline: BaselineConfig
    calibration_samples: int = 128
    scenarios: tuple[TunableScenario, ...] = (
        TunableScenario("quantized_8bit", 8, 0.0, 0.0, 0.0, 0.0, 0.0),
        TunableScenario("mild", 8, 0.005, 0.005, 0.001, 0.005, 0.0),
        TunableScenario("moderate", 6, 0.01, 0.01, 0.005, 0.01, 0.005),
        TunableScenario("severe", 4, 0.02, 0.02, 0.02, 0.02, 0.02),
    )


def load_train_and_test(config: BaselineConfig) -> tuple[np.ndarray, ...]:
    if config.synthetic:
        train_images, train_labels, test_images, test_labels = make_synthetic_dataset(
            seed=config.seed
        )
    elif config.dataset == "fashion_mnist":
        train_images, train_labels, test_images, test_labels = load_fashion_mnist(config.data_dir)
    elif config.dataset == "cifar10":
        train_images, train_labels, test_images, test_labels = load_cifar10(config.data_dir)
    else:
        raise ValueError(
            f"Unsupported dataset {config.dataset!r}; choose fashion_mnist, cifar10, or --synthetic"
        )
    train_images, train_labels = limit_samples(train_images, train_labels, config.max_train_samples)
    test_images, test_labels = limit_samples(test_images, test_labels, config.max_test_samples)
    return train_images, train_labels, test_images, test_labels


def apply_tunable_weight_errors(
    weights: np.ndarray,
    scenario: TunableScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, object]]:
    quantized, quantization_metrics = quantize_symmetric(weights, scenario.bits)
    max_abs = max(float(np.max(np.abs(weights))), 1e-8)

    setting_noise = rng.normal(0.0, scenario.setting_error * max_abs, size=weights.shape).astype(
        np.float32
    )
    drift_noise = rng.normal(0.0, scenario.drift * max_abs, size=weights.shape).astype(np.float32)
    errored = quantized + setting_noise + drift_noise

    input_gain = rng.normal(1.0, scenario.gain_error, size=(weights.shape[0], 1)).astype(np.float32)
    output_gain = rng.normal(1.0, scenario.gain_error, size=(1, weights.shape[1])).astype(
        np.float32
    )
    errored = errored * input_gain * output_gain

    if scenario.crosstalk > 0.0:
        mixing = rng.normal(
            0.0, scenario.crosstalk, size=(weights.shape[1], weights.shape[1])
        ).astype(np.float32)
        np.fill_diagonal(mixing, 1.0)
        errored = errored @ mixing

    error = errored - weights
    return errored.astype(np.float32), {
        "quantization": quantization_metrics,
        "weight_error_mse": float(np.mean(error * error)),
        "weight_error_max": float(np.max(np.abs(error))),
        "input_gain_std": float(np.std(input_gain)),
        "output_gain_std": float(np.std(output_gain)),
    }


def project(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    w_opt: np.ndarray,
    rng: np.random.Generator,
    detector_noise: float,
) -> np.ndarray:
    projection = inputs @ w_opt + params["b_opt"]
    if detector_noise > 0.0:
        scale = max(float(np.std(projection)), 1e-8)
        projection = projection + rng.normal(0.0, detector_noise * scale, size=projection.shape)
    return projection.astype(np.float32)


def fit_affine_calibration(
    ideal_projection: np.ndarray,
    observed_projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    observed_mean = np.mean(observed_projection, axis=0)
    ideal_mean = np.mean(ideal_projection, axis=0)
    observed_centered = observed_projection - observed_mean
    ideal_centered = ideal_projection - ideal_mean
    denominator = np.sum(observed_centered * observed_centered, axis=0) + 1e-8
    gain = np.sum(observed_centered * ideal_centered, axis=0) / denominator
    bias = ideal_mean - gain * observed_mean
    return gain.astype(np.float32), bias.astype(np.float32)


def evaluate_projection(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    projection: np.ndarray,
) -> dict[str, object]:
    hidden = relu(projection)
    logits = hidden @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "projection_mean": float(np.mean(projection)),
        "projection_std": float(np.std(projection)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def predict_projection(params: dict[str, np.ndarray], projection: np.ndarray) -> np.ndarray:
    hidden = relu(projection)
    logits = hidden @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return np.argmax(probs, axis=1)


def compare_projection(
    params: dict[str, np.ndarray],
    labels: np.ndarray,
    ideal_projection: np.ndarray,
    observed_projection: np.ndarray,
) -> dict[str, object]:
    baseline_predictions = predict_projection(params, ideal_projection)
    observed_predictions = predict_projection(params, observed_projection)
    baseline_confusion = confusion_matrix(labels, baseline_predictions)
    observed_confusion = confusion_matrix(labels, observed_predictions)
    return {
        "projection": projection_diagnostics(ideal_projection, observed_projection),
        "prediction": prediction_delta(
            labels,
            baseline_predictions,
            observed_predictions,
            baseline_confusion,
            observed_confusion,
        ),
    }


def run_tunable_error_sweeps(config: TunableSweepConfig) -> dict[str, object]:
    baseline_metrics = run_baseline(config.baseline)
    weights_path = Path(str(baseline_metrics["artifacts"]["weights"]))
    weights = np.load(weights_path)
    params = {name: weights[name].astype(np.float32) for name in weights.files}
    train_images, _, test_images, test_labels = load_train_and_test(config.baseline)

    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]

    rng = np.random.default_rng(config.baseline.seed + 1000)
    ideal_test_projection = project(params, test_images, params["w_opt"], rng, detector_noise=0.0)
    baseline_eval = evaluate_projection(params, test_images, test_labels, ideal_test_projection)

    sweeps: list[dict[str, object]] = []
    for scenario in config.scenarios:
        name_seed = sum((index + 1) * ord(char) for index, char in enumerate(scenario.name))
        scenario_rng = np.random.default_rng(config.baseline.seed + name_seed)
        errored_w, error_metrics = apply_tunable_weight_errors(
            params["w_opt"],
            scenario,
            scenario_rng,
        )
        raw_projection = project(
            params, test_images, errored_w, scenario_rng, scenario.detector_noise
        )
        raw_metrics = evaluate_projection(params, test_images, test_labels, raw_projection)

        ideal_calibration = project(params, calibration_images, params["w_opt"], scenario_rng, 0.0)
        observed_calibration = project(
            params,
            calibration_images,
            errored_w,
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

    output_dir = config.baseline.output_dir.parent / "tunable_errors"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "tunable_error_sweeps.json"
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
