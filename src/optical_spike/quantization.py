"""Quantized projection sweeps for the optical-candidate layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from optical_spike.baseline import BaselineConfig, evaluate, forward, limit_samples, run_baseline
from optical_spike.data import load_cifar10, load_fashion_mnist, make_synthetic_dataset
from optical_spike.diagnostics import confusion_matrix, prediction_delta, projection_diagnostics


@dataclass(frozen=True)
class QuantizationSweepConfig:
    baseline: BaselineConfig
    bits: tuple[int, ...] = (8, 6, 4, 3, 2)


def parse_bits(value: str) -> tuple[int, ...]:
    bits = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not bits:
        raise ValueError("At least one quantization bit width is required")
    for bit_width in bits:
        if bit_width < 2:
            raise ValueError(f"Quantization bit width must be >= 2, got {bit_width}")
    return bits


def quantize_symmetric(values: np.ndarray, bits: int) -> tuple[np.ndarray, dict[str, float | int]]:
    levels = (2 ** (bits - 1)) - 1
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return values.copy(), {"bits": bits, "scale": 0.0, "max_abs": 0.0, "levels": levels}

    scale = max_abs / levels
    quantized_int = np.clip(np.round(values / scale), -levels, levels)
    quantized = (quantized_int * scale).astype(np.float32)
    error = quantized - values
    return quantized, {
        "bits": bits,
        "scale": float(scale),
        "max_abs": max_abs,
        "levels": levels,
        "mse": float(np.mean(error * error)),
        "max_error": float(np.max(np.abs(error))),
    }


def load_eval_data(config: BaselineConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.synthetic:
        _, _, test_images, test_labels = make_synthetic_dataset(seed=config.seed)
    elif config.dataset == "fashion_mnist":
        _, _, test_images, test_labels = load_fashion_mnist(config.data_dir)
    elif config.dataset == "cifar10":
        _, _, test_images, test_labels = load_cifar10(config.data_dir)
    else:
        raise ValueError(
            f"Unsupported dataset {config.dataset!r}; choose fashion_mnist, cifar10, or --synthetic"
        )
    return limit_samples(test_images, test_labels, config.max_test_samples)


def run_quantized_projection_sweeps(config: QuantizationSweepConfig) -> dict[str, object]:
    baseline_metrics = run_baseline(config.baseline)
    weights_path = Path(str(baseline_metrics["artifacts"]["weights"]))
    weights = np.load(weights_path)
    params = {name: weights[name].astype(np.float32) for name in weights.files}
    test_images, test_labels = load_eval_data(config.baseline)

    baseline_eval = evaluate(params, test_images, test_labels)
    sweeps: list[dict[str, object]] = []
    for bit_width in config.bits:
        quantized_w, quantization_metrics = quantize_symmetric(params["w_opt"], bit_width)
        quantized_params = dict(params)
        quantized_params["w_opt"] = quantized_w
        metrics = evaluate(quantized_params, test_images, test_labels)
        baseline_probs, baseline_cache = forward(params, test_images)
        quantized_probs, quantized_cache = forward(quantized_params, test_images)
        baseline_predictions = np.argmax(baseline_probs, axis=1)
        quantized_predictions = np.argmax(quantized_probs, axis=1)
        sweeps.append(
            {
                "bits": bit_width,
                "quantization": quantization_metrics,
                "metrics": metrics,
                "absolute_accuracy_drop": baseline_eval["accuracy"] - metrics["accuracy"],
                "loss_delta": metrics["loss"] - baseline_eval["loss"],
                "diagnostics": {
                    "projection": projection_diagnostics(
                        baseline_cache["projection"],
                        quantized_cache["projection"],
                    ),
                    "prediction": prediction_delta(
                        test_labels,
                        baseline_predictions,
                        quantized_predictions,
                        confusion_matrix(test_labels, baseline_predictions),
                        confusion_matrix(test_labels, quantized_predictions),
                    ),
                },
            }
        )

    output_dir = config.baseline.output_dir.parent / "quantized_sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "quantized_projection_sweeps.json"
    payload: dict[str, object] = {
        "config": {
            "baseline": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.baseline).items()
            },
            "bits": list(config.bits),
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
