"""Transformer-style MLP block target for optical transfer checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np

from optical_spike.baseline import (
    adam_update,
    cross_entropy,
    limit_samples,
    one_hot,
    relu,
    softmax,
)
from optical_spike.data import load_cifar10, load_fashion_mnist, make_synthetic_dataset
from optical_spike.diagnostics import confusion_matrix, prediction_delta, projection_diagnostics
from optical_spike.fabrication_constraints import (
    DEFAULT_CONSTRAINED_FIXED_SCENARIOS,
    ConstrainedFixedScenario,
    apply_constrained_fixed_weight_errors,
)
from optical_spike.fixed import FixedScenario, apply_fixed_weight_errors
from optical_spike.quantization import parse_bits, quantize_symmetric
from optical_spike.tunable import (
    TunableScenario,
    apply_tunable_weight_errors,
    fit_affine_calibration,
)


@dataclass(frozen=True)
class TransformerMLPConfig:
    data_dir: Path = Path("data/fashion-mnist")
    dataset: str = "fashion_mnist"
    output_dir: Path = Path("artifacts/spike/transformer_mlp")
    hidden_dim: int = 64
    mlp_dim: int = 256
    epochs: int = 5
    batch_size: int = 256
    learning_rate: float = 0.003
    seed: int = 7
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    synthetic: bool = False
    quantization_bits: tuple[int, ...] = (8, 6, 4, 3, 2)
    calibration_samples: int = 128
    adaptation_epochs: int = 8
    adaptation_learning_rate: float = 0.01
    tunable_scenarios: tuple[TunableScenario, ...] = (
        TunableScenario("quantized_8bit", 8, 0.0, 0.0, 0.0, 0.0, 0.0),
        TunableScenario("mild", 8, 0.005, 0.005, 0.001, 0.005, 0.0),
        TunableScenario("moderate", 6, 0.01, 0.01, 0.005, 0.01, 0.005),
        TunableScenario("severe", 4, 0.02, 0.02, 0.02, 0.02, 0.02),
    )
    fixed_scenarios: tuple[FixedScenario, ...] = (
        FixedScenario("near_ideal_fixed", 8, 0.001, 0.005, 0.002, 0.001, 0.002),
        FixedScenario("mild_fixed", 8, 0.005, 0.01, 0.005, 0.002, 0.005),
        FixedScenario("moderate_fixed", 6, 0.01, 0.03, 0.01, 0.01, 0.01),
        FixedScenario("severe_fixed", 4, 0.03, 0.08, 0.03, 0.03, 0.02),
    )
    constrained_fixed_scenarios: tuple[ConstrainedFixedScenario, ...] = (
        DEFAULT_CONSTRAINED_FIXED_SCENARIOS
    )


def load_train_and_test(config: TransformerMLPConfig) -> tuple[np.ndarray, ...]:
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


def init_params(
    input_dim: int,
    hidden_dim: int,
    mlp_dim: int,
    classes: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    return {
        "w_embed": rng.normal(0.0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(
            np.float32
        ),
        "b_embed": np.zeros(hidden_dim, dtype=np.float32),
        "w_mlp_up": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, mlp_dim)).astype(
            np.float32
        ),
        "b_mlp_up": np.zeros(mlp_dim, dtype=np.float32),
        "w_mlp_down": rng.normal(0.0, np.sqrt(2.0 / mlp_dim), size=(mlp_dim, hidden_dim)).astype(
            np.float32
        ),
        "b_mlp_down": np.zeros(hidden_dim, dtype=np.float32),
        "w_head": rng.normal(0.0, np.sqrt(2.0 / hidden_dim), size=(hidden_dim, classes)).astype(
            np.float32
        ),
        "b_head": np.zeros(classes, dtype=np.float32),
    }


def forward(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    *,
    w_mlp_up: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    up_weights = params["w_mlp_up"] if w_mlp_up is None else w_mlp_up
    embed_projection = inputs @ params["w_embed"] + params["b_embed"]
    embed = relu(embed_projection)
    mlp_up_projection = embed @ up_weights + params["b_mlp_up"]
    mlp_up = relu(mlp_up_projection)
    mlp_down_projection = mlp_up @ params["w_mlp_down"] + params["b_mlp_down"]
    residual = relu(embed + mlp_down_projection)
    logits = residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {
        "embed_projection": embed_projection,
        "embed": embed,
        "mlp_up_projection": mlp_up_projection,
        "mlp_up": mlp_up,
        "mlp_down_projection": mlp_down_projection,
        "residual": residual,
        "logits": logits,
    }


def embed_inputs(params: dict[str, np.ndarray], inputs: np.ndarray) -> np.ndarray:
    return relu(inputs @ params["w_embed"] + params["b_embed"])


def project_mlp_up(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    w_mlp_up: np.ndarray,
    rng: np.random.Generator,
    detector_noise: float = 0.0,
) -> np.ndarray:
    projection = embed_inputs(params, inputs) @ w_mlp_up + params["b_mlp_up"]
    if detector_noise > 0.0:
        scale = max(float(np.std(projection)), 1e-8)
        projection = projection + rng.normal(0.0, detector_noise * scale, size=projection.shape)
    return projection.astype(np.float32)


def forward_from_mlp_up_projection(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    mlp_up_projection: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    embed = embed_inputs(params, inputs)
    mlp_up = relu(mlp_up_projection)
    mlp_down_projection = mlp_up @ params["w_mlp_down"] + params["b_mlp_down"]
    residual = relu(embed + mlp_down_projection)
    logits = residual @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    return probs, {
        "embed": embed,
        "mlp_up_projection": mlp_up_projection,
        "mlp_up": mlp_up,
        "mlp_down_projection": mlp_down_projection,
        "residual": residual,
        "logits": logits,
    }


def evaluate(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    *,
    w_mlp_up: np.ndarray | None = None,
) -> dict[str, object]:
    probs, cache = forward(params, inputs, w_mlp_up=w_mlp_up)
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "mlp_up_projection_mean": float(np.mean(cache["mlp_up_projection"])),
        "mlp_up_projection_std": float(np.std(cache["mlp_up_projection"])),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def evaluate_mlp_up_projection(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    mlp_up_projection: np.ndarray,
) -> dict[str, object]:
    probs, cache = forward_from_mlp_up_projection(params, inputs, mlp_up_projection)
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "mlp_up_projection_mean": float(np.mean(cache["mlp_up_projection"])),
        "mlp_up_projection_std": float(np.std(cache["mlp_up_projection"])),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def train_epoch(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    config: TransformerMLPConfig,
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

        probs, cache = forward(params, x)
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
        d_mlp_up_projection = d_mlp_up * (cache["mlp_up_projection"] > 0.0)
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


def block_diagnostics(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    observed_w_mlp_up: np.ndarray,
) -> dict[str, object]:
    baseline_probs, baseline_cache = forward(params, inputs)
    observed_probs, observed_cache = forward(params, inputs, w_mlp_up=observed_w_mlp_up)
    baseline_predictions = np.argmax(baseline_probs, axis=1)
    observed_predictions = np.argmax(observed_probs, axis=1)
    return {
        "projection": projection_diagnostics(
            baseline_cache["mlp_up_projection"],
            observed_cache["mlp_up_projection"],
        ),
        "prediction": prediction_delta(
            labels,
            baseline_predictions,
            observed_predictions,
            confusion_matrix(labels, baseline_predictions),
            confusion_matrix(labels, observed_predictions),
        ),
    }


def projection_diagnostics_payload(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    ideal_projection: np.ndarray,
    observed_projection: np.ndarray,
) -> dict[str, object]:
    baseline_probs, _ = forward_from_mlp_up_projection(params, inputs, ideal_projection)
    observed_probs, _ = forward_from_mlp_up_projection(params, inputs, observed_projection)
    baseline_predictions = np.argmax(baseline_probs, axis=1)
    observed_predictions = np.argmax(observed_probs, axis=1)
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


def scenario_seed(name: str, base: int) -> int:
    name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return base + name_seed


def run_optical_error_sweeps(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    baseline_accuracy: float,
    config: TransformerMLPConfig,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]
    adaptation_images = train_images[calibration_count:]
    if adaptation_images.shape[0] == 0:
        adaptation_images = train_images

    rng = np.random.default_rng(config.seed + 5000)
    ideal_test_projection = project_mlp_up(params, test_images, params["w_mlp_up"], rng, 0.0)
    ideal_calibration = project_mlp_up(params, calibration_images, params["w_mlp_up"], rng, 0.0)

    tunable_sweeps: list[dict[str, object]] = []
    for scenario in config.tunable_scenarios:
        scenario_rng = np.random.default_rng(scenario_seed(scenario.name, config.seed + 50_000))
        errored_w, error_metrics = apply_tunable_weight_errors(
            params["w_mlp_up"],
            scenario,
            scenario_rng,
        )
        raw_projection = project_mlp_up(
            params,
            test_images,
            errored_w,
            scenario_rng,
            scenario.detector_noise,
        )
        observed_calibration = project_mlp_up(
            params,
            calibration_images,
            errored_w,
            scenario_rng,
            scenario.detector_noise,
        )
        gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
        calibrated_projection = raw_projection * gain + bias
        raw_metrics = evaluate_mlp_up_projection(params, test_images, test_labels, raw_projection)
        calibrated_metrics = evaluate_mlp_up_projection(
            params,
            test_images,
            test_labels,
            calibrated_projection,
        )
        tunable_sweeps.append(
            {
                "scenario": asdict(scenario),
                "error_model": error_metrics,
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
                "raw_accuracy_drop": baseline_accuracy - float(raw_metrics["accuracy"]),
                "calibrated_accuracy_drop": baseline_accuracy
                - float(calibrated_metrics["accuracy"]),
                "raw_diagnostics": projection_diagnostics_payload(
                    params,
                    test_images,
                    test_labels,
                    ideal_test_projection,
                    raw_projection,
                ),
                "calibrated_diagnostics": projection_diagnostics_payload(
                    params,
                    test_images,
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

    fixed_sweeps: list[dict[str, object]] = []
    fixed_weights: dict[str, np.ndarray] = {}
    for scenario in config.fixed_scenarios:
        scenario_rng = np.random.default_rng(scenario_seed(scenario.name, config.seed + 60_000))
        fixed_w, error_metrics = apply_fixed_weight_errors(
            params["w_mlp_up"],
            scenario,
            scenario_rng,
        )
        fixed_weights[scenario.name] = fixed_w
        raw_projection = project_mlp_up(
            params,
            test_images,
            fixed_w,
            scenario_rng,
            scenario.detector_noise,
        )
        observed_calibration = project_mlp_up(
            params,
            calibration_images,
            fixed_w,
            scenario_rng,
            scenario.detector_noise,
        )
        gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
        calibrated_projection = raw_projection * gain + bias
        raw_metrics = evaluate_mlp_up_projection(params, test_images, test_labels, raw_projection)
        calibrated_metrics = evaluate_mlp_up_projection(
            params,
            test_images,
            test_labels,
            calibrated_projection,
        )
        fixed_sweeps.append(
            {
                "scenario": asdict(scenario),
                "error_model": error_metrics,
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
                "raw_accuracy_drop": baseline_accuracy - float(raw_metrics["accuracy"]),
                "calibrated_accuracy_drop": baseline_accuracy
                - float(calibrated_metrics["accuracy"]),
                "raw_diagnostics": projection_diagnostics_payload(
                    params,
                    test_images,
                    test_labels,
                    ideal_test_projection,
                    raw_projection,
                ),
                "calibrated_diagnostics": projection_diagnostics_payload(
                    params,
                    test_images,
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

    constrained_fixed_sweeps: list[dict[str, object]] = []
    for scenario in config.constrained_fixed_scenarios:
        scenario_rng = np.random.default_rng(scenario_seed(scenario.name, config.seed + 65_000))
        constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
            params["w_mlp_up"],
            scenario,
            scenario_rng,
        )
        raw_projection = project_mlp_up(
            params,
            test_images,
            constrained_w,
            scenario_rng,
            scenario.detector_noise,
        )
        observed_calibration = project_mlp_up(
            params,
            calibration_images,
            constrained_w,
            scenario_rng,
            scenario.detector_noise,
        )
        gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
        calibrated_projection = raw_projection * gain + bias
        raw_metrics = evaluate_mlp_up_projection(params, test_images, test_labels, raw_projection)
        calibrated_metrics = evaluate_mlp_up_projection(
            params,
            test_images,
            test_labels,
            calibrated_projection,
        )
        constrained_fixed_sweeps.append(
            {
                "scenario": asdict(scenario),
                "error_model": error_metrics,
                "raw_metrics": raw_metrics,
                "calibrated_metrics": calibrated_metrics,
                "raw_accuracy_drop": baseline_accuracy - float(raw_metrics["accuracy"]),
                "calibrated_accuracy_drop": baseline_accuracy
                - float(calibrated_metrics["accuracy"]),
                "raw_diagnostics": projection_diagnostics_payload(
                    params,
                    test_images,
                    test_labels,
                    ideal_test_projection,
                    raw_projection,
                ),
                "calibrated_diagnostics": projection_diagnostics_payload(
                    params,
                    test_images,
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

    adaptation = run_mlp_adaptation_check(
        params,
        train_images,
        train_labels,
        test_images,
        test_labels,
        ideal_test_projection,
        fixed_weights["moderate_fixed"],
        baseline_accuracy,
        config,
    )
    return tunable_sweeps, fixed_sweeps, constrained_fixed_sweeps, adaptation


def copy_tail(params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "w_mlp_down": params["w_mlp_down"].copy(),
        "b_mlp_down": params["b_mlp_down"].copy(),
        "w_head": params["w_head"].copy(),
        "b_head": params["b_head"].copy(),
    }


def evaluate_tail(
    params: dict[str, np.ndarray],
    tail: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    mlp_up_projection: np.ndarray,
) -> dict[str, object]:
    embed = embed_inputs(params, inputs)
    mlp_up = relu(mlp_up_projection)
    mlp_down_projection = mlp_up @ tail["w_mlp_down"] + tail["b_mlp_down"]
    residual = relu(embed + mlp_down_projection)
    logits = residual @ tail["w_head"] + tail["b_head"]
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "mlp_up_projection_mean": float(np.mean(mlp_up_projection)),
        "mlp_up_projection_std": float(np.std(mlp_up_projection)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def train_tail(
    params: dict[str, np.ndarray],
    tail: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    mlp_up_projection: np.ndarray,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    trained = {name: value.copy() for name, value in tail.items()}
    embed = embed_inputs(params, inputs)
    mlp_up = relu(mlp_up_projection)
    classes = trained["b_head"].shape[0]
    targets = one_hot(labels, classes)
    for _ in range(epochs):
        indices = rng.permutation(inputs.shape[0])
        for start in range(0, inputs.shape[0], batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_embed = embed[batch_idx]
            batch_mlp_up = mlp_up[batch_idx]
            batch_targets = targets[batch_idx]
            mlp_down_projection = batch_mlp_up @ trained["w_mlp_down"] + trained["b_mlp_down"]
            residual = relu(batch_embed + mlp_down_projection)
            probs = softmax(residual @ trained["w_head"] + trained["b_head"])

            d_logits = (probs - batch_targets) / batch_embed.shape[0]
            grad_w_head = residual.T @ d_logits
            grad_b_head = np.sum(d_logits, axis=0)
            d_residual = d_logits @ trained["w_head"].T
            d_residual_pre = d_residual * (residual > 0.0)
            grad_w_mlp_down = batch_mlp_up.T @ d_residual_pre
            grad_b_mlp_down = np.sum(d_residual_pre, axis=0)

            trained["w_head"] -= learning_rate * grad_w_head.astype(np.float32)
            trained["b_head"] -= learning_rate * grad_b_head.astype(np.float32)
            trained["w_mlp_down"] -= learning_rate * grad_w_mlp_down.astype(np.float32)
            trained["b_mlp_down"] -= learning_rate * grad_b_mlp_down.astype(np.float32)
    return trained


def run_mlp_adaptation_check(
    params: dict[str, np.ndarray],
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    ideal_test_projection: np.ndarray,
    hardware_w: np.ndarray,
    baseline_accuracy: float,
    config: TransformerMLPConfig,
) -> dict[str, object]:
    rng = np.random.default_rng(config.seed + 70_000)
    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]
    adaptation_images = train_images[calibration_count:]
    adaptation_labels = train_labels[calibration_count:]
    if adaptation_images.shape[0] == 0:
        adaptation_images = train_images
        adaptation_labels = train_labels

    scenario = FixedScenario("moderate_fixed_adaptation", 6, 0.01, 0.03, 0.01, 0.01, 0.01)
    raw_test_projection = project_mlp_up(
        params,
        test_images,
        hardware_w,
        rng,
        scenario.detector_noise,
    )
    ideal_calibration = project_mlp_up(params, calibration_images, params["w_mlp_up"], rng, 0.0)
    observed_calibration = project_mlp_up(
        params,
        calibration_images,
        hardware_w,
        rng,
        scenario.detector_noise,
    )
    gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
    calibrated_test_projection = raw_test_projection * gain + bias

    observed_adaptation_projection = project_mlp_up(
        params,
        adaptation_images,
        hardware_w,
        rng,
        scenario.detector_noise,
    )
    fine_tuned_tail = train_tail(
        params,
        copy_tail(params),
        adaptation_images,
        adaptation_labels,
        observed_adaptation_projection,
        epochs=config.adaptation_epochs,
        learning_rate=config.adaptation_learning_rate,
        batch_size=config.batch_size,
        rng=rng,
    )
    fine_tuned_metrics = evaluate_tail(
        params,
        fine_tuned_tail,
        test_images,
        test_labels,
        raw_test_projection,
    )

    modes = [
        {
            "mode": "no_adaptation",
            "metrics": evaluate_mlp_up_projection(
                params, test_images, test_labels, raw_test_projection
            ),
            "diagnostics": projection_diagnostics_payload(
                params,
                test_images,
                test_labels,
                ideal_test_projection,
                raw_test_projection,
            ),
        },
        {
            "mode": "calibration_only",
            "metrics": evaluate_mlp_up_projection(
                params,
                test_images,
                test_labels,
                calibrated_test_projection,
            ),
            "diagnostics": projection_diagnostics_payload(
                params,
                test_images,
                test_labels,
                ideal_test_projection,
                calibrated_test_projection,
            ),
            "details": {
                "samples": calibration_count,
                "gain_mean": float(np.mean(gain)),
                "gain_std": float(np.std(gain)),
                "bias_mean": float(np.mean(bias)),
                "bias_std": float(np.std(bias)),
            },
        },
        {
            "mode": "hardware_aware_tail_fine_tuning",
            "metrics": fine_tuned_metrics,
            "diagnostics": projection_diagnostics_payload(
                params,
                test_images,
                test_labels,
                ideal_test_projection,
                raw_test_projection,
            ),
            "details": {
                "adaptation_samples": int(adaptation_images.shape[0]),
                "trained": ["w_mlp_down", "b_mlp_down", "w_head", "b_head"],
            },
        },
    ]
    for mode in modes:
        metrics = mode["metrics"]
        mode["accuracy_drop"] = baseline_accuracy - float(metrics["accuracy"])  # type: ignore[index]

    return {
        "scenario": asdict(scenario),
        "calibration_samples": calibration_count,
        "adaptation_epochs": config.adaptation_epochs,
        "adaptation_learning_rate": config.adaptation_learning_rate,
        "modes": modes,
        "best_mode": min(modes, key=lambda item: float(item["accuracy_drop"]))["mode"],
    }


def run_transformer_mlp_target(config: TransformerMLPConfig) -> dict[str, object]:
    rng = np.random.default_rng(config.seed)
    train_images, train_labels, test_images, test_labels = load_train_and_test(config)
    params = init_params(train_images.shape[1], config.hidden_dim, config.mlp_dim, 10, rng)
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    velocities = {name: np.zeros_like(value) for name, value in params.items()}

    history: list[dict[str, float | int]] = []
    step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        step = train_epoch(
            params, train_images, train_labels, config, rng, moments, velocities, step
        )
        train_metrics = evaluate(params, train_images, train_labels)
        test_metrics = evaluate(params, test_images, test_labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
        )

    baseline_train = evaluate(params, train_images, train_labels)
    baseline_test = evaluate(params, test_images, test_labels)
    baseline_accuracy = float(baseline_test["accuracy"])
    sweeps: list[dict[str, object]] = []
    for bit_width in config.quantization_bits:
        quantized_w, quantization_metrics = quantize_symmetric(params["w_mlp_up"], bit_width)
        metrics = evaluate(params, test_images, test_labels, w_mlp_up=quantized_w)
        sweeps.append(
            {
                "target": "mlp_up_projection",
                "bits": bit_width,
                "quantization": quantization_metrics,
                "metrics": metrics,
                "absolute_accuracy_drop": baseline_accuracy - float(metrics["accuracy"]),
                "loss_delta": float(metrics["loss"]) - float(baseline_test["loss"]),
                "diagnostics": block_diagnostics(params, test_images, test_labels, quantized_w),
            }
        )

    tunable_sweeps, fixed_sweeps, constrained_fixed_sweeps, adaptation = run_optical_error_sweeps(
        params,
        train_images,
        train_labels,
        test_images,
        test_labels,
        baseline_accuracy,
        config,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config.output_dir / "transformer_mlp_weights.npz"
    metrics_path = config.output_dir / "transformer_mlp_target.json"
    np.savez_compressed(weights_path, **params)
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dataset": "synthetic" if config.synthetic else config.dataset,
        "train_samples": int(train_images.shape[0]),
        "test_samples": int(test_images.shape[0]),
        "elapsed_seconds": time.time() - started,
        "target": {
            "name": "mlp_up_projection",
            "input_dim": config.hidden_dim,
            "output_dim": config.mlp_dim,
            "matrix_shape": list(params["w_mlp_up"].shape),
            "rationale": "Transformer feed-forward expansion projection; sequence-free first target.",
        },
        "history": history,
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "sweeps": sweeps,
        "tunable_sweeps": tunable_sweeps,
        "fixed_sweeps": fixed_sweeps,
        "constrained_fixed_sweeps": constrained_fixed_sweeps,
        "adaptation": adaptation,
        "artifacts": {
            "weights": str(weights_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_transformer_bits(value: str) -> tuple[int, ...]:
    return parse_bits(value)
