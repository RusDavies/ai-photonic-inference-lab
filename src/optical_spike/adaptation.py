"""Calibration and adaptation modes for transferred optical projections."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from optical_spike.baseline import BaselineConfig, one_hot, relu, run_baseline, softmax
from optical_spike.diagnostics import confusion_matrix, prediction_delta, projection_diagnostics
from optical_spike.fixed import FixedScenario, apply_fixed_weight_errors
from optical_spike.tunable import (
    evaluate_projection,
    fit_affine_calibration,
    load_train_and_test,
    project,
)


@dataclass(frozen=True)
class AdaptationConfig:
    baseline: BaselineConfig
    calibration_samples: int = 128
    adaptation_epochs: int = 8
    adaptation_learning_rate: float = 0.01
    distillation_temperature: float = 2.0
    scenario: FixedScenario = FixedScenario(
        "moderate_fixed_adaptation",
        6,
        0.01,
        0.03,
        0.01,
        0.01,
        0.01,
    )


def copy_head(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "w_head": params["w_head"].copy(),
        "b_head": params["b_head"].copy(),
    }


def init_head(hidden_dim: int, classes: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "w_head": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, classes)).astype(
            np.float32
        ),
        "b_head": np.zeros(classes, dtype=np.float32),
    }


def evaluate_head(
    head: dict[str, np.ndarray],
    projection: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    hidden = relu(projection)
    logits = hidden @ head["w_head"] + head["b_head"]
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-8, 1.0)
    return {
        "loss": float(-np.mean(np.log(clipped))),
        "accuracy": float(np.mean(predictions == labels)),
        "projection_mean": float(np.mean(projection)),
        "projection_std": float(np.std(projection)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def predict_head(head: dict[str, np.ndarray], projection: np.ndarray) -> np.ndarray:
    hidden = relu(projection)
    logits = hidden @ head["w_head"] + head["b_head"]
    return np.argmax(softmax(logits), axis=1)


def train_head(
    head: dict[str, np.ndarray],
    projection: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    rng: np.random.Generator,
    soft_targets: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    trained = {name: value.copy() for name, value in head.items()}
    classes = trained["b_head"].shape[0]
    hidden = relu(projection)
    targets = soft_targets if soft_targets is not None else one_hot(labels, classes)

    for _ in range(epochs):
        indices = rng.permutation(projection.shape[0])
        for start in range(0, projection.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_hidden = hidden[batch_idx]
            batch_targets = targets[batch_idx]
            probs = softmax(batch_hidden @ trained["w_head"] + trained["b_head"])
            d_logits = (probs - batch_targets) / batch_hidden.shape[0]
            grad_w_head = batch_hidden.T @ d_logits
            grad_b_head = np.sum(d_logits, axis=0)
            trained["w_head"] -= learning_rate * grad_w_head.astype(np.float32)
            trained["b_head"] -= learning_rate * grad_b_head.astype(np.float32)

    return trained


def softened_teacher_probs(
    params: dict[str, np.ndarray],
    ideal_projection: np.ndarray,
    temperature: float,
) -> np.ndarray:
    hidden = relu(ideal_projection)
    logits = hidden @ params["w_head"] + params["b_head"]
    return softmax(logits / temperature).astype(np.float32)


def mode_payload(
    name: str,
    baseline_accuracy: float,
    metrics: dict[str, object],
    details: dict[str, object] | None = None,
    diagnostics: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "mode": name,
        "metrics": metrics,
        "accuracy_drop": baseline_accuracy - float(metrics["accuracy"]),
    }
    if details:
        payload["details"] = details
    if diagnostics:
        payload["diagnostics"] = diagnostics
    return payload


def adaptation_diagnostics(
    labels: np.ndarray,
    ideal_projection: np.ndarray,
    observed_projection: np.ndarray,
    baseline_predictions: np.ndarray,
    observed_predictions: np.ndarray,
) -> dict[str, object]:
    return {
        "projection": projection_diagnostics(ideal_projection, observed_projection),
        "prediction": prediction_delta(
            labels,
            baseline_predictions,
            observed_predictions,
            confusion_matrix(labels, baseline_predictions),
            confusion_matrix(labels, observed_predictions),
        ),
    }


def run_adaptation_modes(config: AdaptationConfig) -> dict[str, object]:
    baseline_metrics = run_baseline(config.baseline)
    weights_path = Path(str(baseline_metrics["artifacts"]["weights"]))
    weights = np.load(weights_path)
    params = {name: weights[name].astype(np.float32) for name in weights.files}
    train_images, train_labels, test_images, test_labels = load_train_and_test(config.baseline)

    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]
    adaptation_images = train_images[calibration_count:]
    adaptation_labels = train_labels[calibration_count:]
    if adaptation_images.shape[0] == 0:
        adaptation_images = train_images
        adaptation_labels = train_labels

    rng = np.random.default_rng(config.baseline.seed + 4000)
    hardware_w, error_metrics = apply_fixed_weight_errors(
        params["w_opt"],
        config.scenario,
        rng,
    )

    ideal_test_projection = project(params, test_images, params["w_opt"], rng, detector_noise=0.0)
    raw_test_projection = project(
        params,
        test_images,
        hardware_w,
        rng,
        config.scenario.detector_noise,
    )
    baseline_eval = evaluate_projection(params, test_images, test_labels, ideal_test_projection)
    baseline_accuracy = baseline_eval["accuracy"]
    baseline_predictions = predict_head(params, ideal_test_projection)

    ideal_calibration = project(params, calibration_images, params["w_opt"], rng, detector_noise=0.0)
    observed_calibration = project(
        params,
        calibration_images,
        hardware_w,
        rng,
        config.scenario.detector_noise,
    )
    gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
    calibrated_test_projection = raw_test_projection * gain + bias

    observed_adaptation_projection = project(
        params,
        adaptation_images,
        hardware_w,
        rng,
        config.scenario.detector_noise,
    )
    ideal_adaptation_projection = project(
        params,
        adaptation_images,
        params["w_opt"],
        rng,
        detector_noise=0.0,
    )

    modes: list[dict[str, object]] = [
        mode_payload(
            "no_adaptation",
            float(baseline_accuracy),
            evaluate_projection(params, test_images, test_labels, raw_test_projection),
            diagnostics=adaptation_diagnostics(
                test_labels,
                ideal_test_projection,
                raw_test_projection,
                baseline_predictions,
                predict_head(params, raw_test_projection),
            ),
        ),
        mode_payload(
            "calibration_only",
            float(baseline_accuracy),
            evaluate_projection(params, test_images, test_labels, calibrated_test_projection),
            {
                "samples": calibration_count,
                "gain_mean": float(np.mean(gain)),
                "gain_std": float(np.std(gain)),
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
            },
            diagnostics=adaptation_diagnostics(
                test_labels,
                ideal_test_projection,
                calibrated_test_projection,
                baseline_predictions,
                predict_head(params, calibrated_test_projection),
            ),
        ),
    ]

    retrained_head = train_head(
        init_head(params["w_head"].shape[0], params["w_head"].shape[1], rng),
        observed_adaptation_projection,
        adaptation_labels,
        epochs=config.adaptation_epochs,
        learning_rate=config.adaptation_learning_rate,
        batch_size=config.baseline.batch_size,
        rng=rng,
    )
    modes.append(
        mode_payload(
            "retrained_downstream_head",
            float(baseline_accuracy),
            evaluate_head(retrained_head, raw_test_projection, test_labels),
            {"adaptation_samples": int(adaptation_images.shape[0])},
            diagnostics=adaptation_diagnostics(
                test_labels,
                ideal_test_projection,
                raw_test_projection,
                baseline_predictions,
                predict_head(retrained_head, raw_test_projection),
            ),
        )
    )

    fine_tuned_head = train_head(
        copy_head(params),
        observed_adaptation_projection,
        adaptation_labels,
        epochs=config.adaptation_epochs,
        learning_rate=config.adaptation_learning_rate * 0.5,
        batch_size=config.baseline.batch_size,
        rng=rng,
    )
    modes.append(
        mode_payload(
            "hardware_aware_fine_tuning",
            float(baseline_accuracy),
            evaluate_head(fine_tuned_head, raw_test_projection, test_labels),
            {"adaptation_samples": int(adaptation_images.shape[0])},
            diagnostics=adaptation_diagnostics(
                test_labels,
                ideal_test_projection,
                raw_test_projection,
                baseline_predictions,
                predict_head(fine_tuned_head, raw_test_projection),
            ),
        )
    )

    teacher_probs = softened_teacher_probs(
        params,
        ideal_adaptation_projection,
        config.distillation_temperature,
    )
    distilled_head = train_head(
        copy_head(params),
        observed_adaptation_projection,
        adaptation_labels,
        epochs=config.adaptation_epochs,
        learning_rate=config.adaptation_learning_rate * 0.5,
        batch_size=config.baseline.batch_size,
        rng=rng,
        soft_targets=teacher_probs,
    )
    modes.append(
        mode_payload(
            "distillation",
            float(baseline_accuracy),
            evaluate_head(distilled_head, raw_test_projection, test_labels),
            {
                "adaptation_samples": int(adaptation_images.shape[0]),
                "teacher": "digital_baseline",
                "temperature": config.distillation_temperature,
            },
            diagnostics=adaptation_diagnostics(
                test_labels,
                ideal_test_projection,
                raw_test_projection,
                baseline_predictions,
                predict_head(distilled_head, raw_test_projection),
            ),
        )
    )

    output_dir = config.baseline.output_dir.parent / "adaptation_modes"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "adaptation_mode_sweeps.json"
    payload: dict[str, object] = {
        "config": {
            "baseline": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.baseline).items()
            },
            "calibration_samples": config.calibration_samples,
            "adaptation_epochs": config.adaptation_epochs,
            "adaptation_learning_rate": config.adaptation_learning_rate,
            "distillation_temperature": config.distillation_temperature,
            "scenario": asdict(config.scenario),
        },
        "baseline": baseline_metrics,
        "baseline_eval": baseline_eval,
        "hardware_error_model": error_metrics,
        "modes": modes,
        "best_mode": min(modes, key=lambda item: float(item["accuracy_drop"]))["mode"],
        "artifacts": {
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
