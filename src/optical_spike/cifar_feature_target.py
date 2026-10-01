"""CIFAR-10 spatial feature target for optical-candidate projection tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from optical_spike.baseline import (
    BaselineConfig,
    cross_entropy,
    evaluate,
    init_params,
    relu,
    softmax,
    train_epoch,
)
from optical_spike.calibration_cost import drifted_weights
from optical_spike.data import load_cifar10
from optical_spike.fabrication_constraints import (
    DEFAULT_CONSTRAINED_FIXED_SCENARIOS,
    ConstrainedFixedScenario,
    apply_constrained_fixed_weight_errors,
)
from optical_spike.material_training import (
    SOURCE_CODING_STRESS_SCENARIOS,
    PhysicalDriftNoiseScenario,
    SourceCodingScenario,
    apply_source_coding,
)
from optical_spike.quantization import quantize_symmetric
from optical_spike.reporting import markdown_table
from optical_spike.transformer_mlp import scenario_seed
from optical_spike.tunable import fit_affine_calibration


@dataclass(frozen=True)
class CifarFeatureTargetConfig:
    data_dir: Path = Path("data/cifar10")
    output_dir: Path = Path("artifacts/spike/cifar_feature_target")
    hidden_dim: int = 256
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 0.003
    seed: int = 7
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    quantization_bits: tuple[int, ...] = (8, 6, 4)


@dataclass(frozen=True)
class CifarFeatureSourceCodingStressConfig:
    target: CifarFeatureTargetConfig
    output_dir: Path
    calibration_sample_counts: tuple[int, ...] = (128,)
    source_codings: tuple[SourceCodingScenario, ...] = SOURCE_CODING_STRESS_SCENARIOS
    constrained_scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]
    physical_scenarios: tuple[PhysicalDriftNoiseScenario, ...] = ()
    projection_mse_recalibration_threshold: float = 0.18


def _pooled_channels(images: np.ndarray, grid: int) -> np.ndarray:
    samples = images.shape[0]
    block = 32 // grid
    return (
        images.reshape(samples, grid, block, grid, block, 3)
        .mean(axis=(2, 4))
        .reshape(samples, -1)
    )


def _pooled_plane(values: np.ndarray, grid: int) -> np.ndarray:
    samples = values.shape[0]
    block = 32 // grid
    return values.reshape(samples, grid, block, grid, block).mean(axis=(2, 4)).reshape(samples, -1)


def _local_color_std(images: np.ndarray, grid: int) -> np.ndarray:
    samples = images.shape[0]
    block = 32 // grid
    return (
        images.reshape(samples, grid, block, grid, block, 3)
        .std(axis=(2, 4))
        .reshape(samples, -1)
    )


def _raw_cifar_features(flat_images: np.ndarray) -> np.ndarray:
    samples = flat_images.shape[0]
    images = flat_images.reshape(samples, 3, 32, 32).transpose(0, 2, 3, 1)
    gray = images.mean(axis=3)
    dx = np.zeros_like(gray)
    dy = np.zeros_like(gray)
    dx[:, :, 1:-1] = gray[:, :, 2:] - gray[:, :, :-2]
    dy[:, 1:-1, :] = gray[:, 2:, :] - gray[:, :-2, :]
    edge_magnitude = np.sqrt((dx * dx) + (dy * dy))

    feature_blocks = [
        _pooled_channels(images, 8),
        _pooled_channels(images, 4),
        _pooled_channels(images, 2),
        _pooled_plane(gray, 8),
        _pooled_plane(gray, 4),
        _pooled_plane(edge_magnitude, 8),
        _pooled_plane(edge_magnitude, 4),
        _local_color_std(images, 4),
    ]
    return np.concatenate(feature_blocks, axis=1).astype(np.float32)


def extract_cifar_spatial_features(
    flat_images: np.ndarray,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract deterministic spatial/color features and z-score them."""
    raw_features = _raw_cifar_features(flat_images)
    feature_mean = raw_features.mean(axis=0, keepdims=True) if mean is None else mean
    feature_std = raw_features.std(axis=0, keepdims=True) if std is None else std
    feature_std = np.maximum(feature_std, 1e-6)
    standardized = (raw_features - feature_mean) / feature_std
    return standardized.astype(np.float32), feature_mean.astype(np.float32), feature_std.astype(
        np.float32
    )


def _limit_samples(
    images: np.ndarray,
    labels: np.ndarray,
    limit: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if limit is None:
        return images, labels
    return images[:limit], labels[:limit]


def _train_cifar_feature_projection(
    config: CifarFeatureTargetConfig,
) -> tuple[dict[str, np.ndarray], tuple[np.ndarray, ...], dict[str, object]]:
    rng = np.random.default_rng(config.seed)
    train_images, train_labels, test_images, test_labels = load_cifar10(config.data_dir)
    train_images, train_labels = _limit_samples(
        train_images, train_labels, config.max_train_samples
    )
    test_images, test_labels = _limit_samples(test_images, test_labels, config.max_test_samples)
    train_features, feature_mean, feature_std = extract_cifar_spatial_features(train_images)
    test_features, _, _ = extract_cifar_spatial_features(
        test_images,
        mean=feature_mean,
        std=feature_std,
    )

    train_config = BaselineConfig(
        output_dir=config.output_dir,
        hidden_dim=config.hidden_dim,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        seed=config.seed,
    )
    params = init_params(train_features.shape[1], config.hidden_dim, 10, rng)
    moments = {name: np.zeros_like(value) for name, value in params.items()}
    velocities = {name: np.zeros_like(value) for name, value in params.items()}

    history: list[dict[str, float | int]] = []
    step = 0
    started = time.time()
    for epoch in range(1, config.epochs + 1):
        step = train_epoch(
            params,
            train_features,
            train_labels,
            train_config,
            rng,
            moments,
            velocities,
            step,
        )
        train_metrics = evaluate(params, train_features, train_labels)
        test_metrics = evaluate(params, test_features, test_labels)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_accuracy": float(train_metrics["accuracy"]),
                "test_loss": float(test_metrics["loss"]),
                "test_accuracy": float(test_metrics["accuracy"]),
            }
        )

    metrics: dict[str, object] = {
        "history": history,
        "final_train": evaluate(params, train_features, train_labels),
        "final_test": evaluate(params, test_features, test_labels),
        "elapsed_seconds": time.time() - started,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }
    data = (
        train_features,
        train_labels,
        test_features,
        test_labels,
        feature_mean,
        feature_std,
    )
    return params, data, metrics


def run_cifar_feature_target(config: CifarFeatureTargetConfig) -> dict[str, object]:
    params, data, training_metrics = _train_cifar_feature_projection(config)
    train_features, _, test_features, test_labels, feature_mean, feature_std = data
    final_train = training_metrics["final_train"]
    final_test = training_metrics["final_test"]
    baseline_accuracy = float(final_test["accuracy"])  # type: ignore[index]
    quantized_sweeps: list[dict[str, object]] = []
    for bits in config.quantization_bits:
        quantized_w, quantization_metrics = quantize_symmetric(params["w_opt"], bits)
        quantized_params = dict(params)
        quantized_params["w_opt"] = quantized_w
        metrics = evaluate(quantized_params, test_features, test_labels)
        quantized_sweeps.append(
            {
                "target": "cifar_spatial_feature_projection",
                "bits": bits,
                "quantization": quantization_metrics,
                "metrics": metrics,
                "absolute_accuracy_drop": baseline_accuracy - float(metrics["accuracy"]),
                "loss_delta": float(metrics["loss"]) - float(final_test["loss"]),
            }
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = config.output_dir / "cifar_feature_target_weights.npz"
    metrics_path = config.output_dir / "cifar_feature_target.json"
    np.savez_compressed(
        weights_path,
        **params,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "dataset": "cifar10",
        "train_samples": int(train_features.shape[0]),
        "test_samples": int(test_features.shape[0]),
        "feature_dim": int(train_features.shape[1]),
        "elapsed_seconds": training_metrics["elapsed_seconds"],
        "target": {
            "name": "cifar_spatial_feature_projection",
            "input_dim": int(train_features.shape[1]),
            "output_dim": config.hidden_dim,
            "matrix_shape": list(params["w_opt"].shape),
            "rationale": (
                "Frozen spatial/color CIFAR-10 feature extractor followed by "
                "a trainable optical-candidate projection."
            ),
        },
        "history": training_metrics["history"],
        "final_train": final_train,
        "final_test": final_test,
        "quantized_sweeps": quantized_sweeps,
        "artifacts": {
            "weights": str(weights_path),
            "metrics": str(metrics_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _evaluate_projected_features(
    params: dict[str, np.ndarray],
    features: np.ndarray,
    labels: np.ndarray,
    *,
    w_opt: np.ndarray | None = None,
    detector_noise: float = 0.0,
    output_readout_noise: float = 0.0,
    calibration: tuple[np.ndarray, np.ndarray] | None = None,
    source_coding: SourceCodingScenario | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    noise_rng = np.random.default_rng(0) if rng is None else rng
    weights = params["w_opt"] if w_opt is None else w_opt
    ideal_projection = features @ params["w_opt"] + params["b_opt"]
    projection = features @ weights + params["b_opt"]
    projection, source_metrics = apply_source_coding(projection, source_coding, noise_rng)
    if detector_noise > 0.0:
        scale = max(float(np.std(projection)), 1e-8)
        projection = projection + noise_rng.normal(
            0.0,
            detector_noise * scale,
            size=projection.shape,
        ).astype(np.float32)
    if calibration is not None:
        gain, bias = calibration
        projection = projection * gain + bias
    hidden = relu(projection)
    if output_readout_noise > 0.0:
        scale = max(float(np.std(hidden)), 1e-8)
        hidden = hidden + noise_rng.normal(
            0.0,
            output_readout_noise * scale,
            size=hidden.shape,
        ).astype(np.float32)
    logits = hidden @ params["w_head"] + params["b_head"]
    probs = softmax(logits)
    predictions = np.argmax(probs, axis=1)
    projection_delta = projection - ideal_projection
    return {
        "loss": cross_entropy(probs, labels),
        "accuracy": float(np.mean(predictions == labels)),
        "projection_mse_mean": float(np.mean(projection_delta * projection_delta)),
        "projection_mse_max": float(np.max(projection_delta * projection_delta)),
        "source_quantization_mse_mean": float(source_metrics["source_quantization_mse"]),
        "source_quantization_max_error": float(source_metrics["source_quantization_max_error"]),
        "hidden_std": float(np.std(hidden)),
    }


def _fit_feature_projection_calibration(
    params: dict[str, np.ndarray],
    calibration_features: np.ndarray,
    w_opt: np.ndarray,
    detector_noise: float,
    source_coding: SourceCodingScenario,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    ideal_projection = calibration_features @ params["w_opt"] + params["b_opt"]
    observed_projection = calibration_features @ w_opt + params["b_opt"]
    observed_projection, _ = apply_source_coding(observed_projection, source_coding, rng)
    if detector_noise > 0.0:
        scale = max(float(np.std(observed_projection)), 1e-8)
        observed_projection = observed_projection + rng.normal(
            0.0,
            detector_noise * scale,
            size=observed_projection.shape,
        ).astype(np.float32)
    return fit_affine_calibration(ideal_projection, observed_projection)


def _summarize_feature_source_coding(frame: pd.DataFrame) -> pd.DataFrame:
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
            target_accuracy=("target_accuracy", "mean"),
            calibrated_accuracy_mean=("calibrated_accuracy", "mean"),
            calibrated_accuracy_min=("calibrated_accuracy", "min"),
            final_step_accuracy_mean=("final_step_accuracy", "mean"),
            mean_drop_vs_target=("drop_vs_target", "mean"),
            worst_drop_vs_target=("drop_vs_target", "max"),
            projection_mse_mean=("projection_mse_mean", "mean"),
            projection_mse_max=("projection_mse_max", "max"),
            source_quantization_mse_mean=("source_quantization_mse_mean", "mean"),
            shared_boundary_source_dac_energy_pj=(
                "shared_boundary_source_dac_energy_pj",
                "first",
            ),
            source_coding_latency_ns=("source_coding_latency_ns", "first"),
            recalibrations_mean=("recalibration_count", "mean"),
            triggered_recalibrations_mean=("triggered_recalibration_count", "mean"),
            steps_below_50_percent=("below_50_percent", "sum"),
            rows=("seed", "count"),
        )
        .reset_index()
        .sort_values(
            [
                "steps_below_50_percent",
                "calibrated_accuracy_min",
                "shared_boundary_source_dac_energy_pj",
                "source_coding_latency_ns",
            ],
            ascending=[True, False, True, True],
        )
    )


def run_cifar_feature_source_coding_stress(
    config: CifarFeatureSourceCodingStressConfig,
) -> dict[str, object]:
    if not config.physical_scenarios:
        raise ValueError("At least one physical scenario is required.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    params, data, training_metrics = _train_cifar_feature_projection(config.target)
    train_features, _, test_features, test_labels, _, _ = data
    target_accuracy = float(training_metrics["final_test"]["accuracy"])  # type: ignore[index]
    constrained_rng = np.random.default_rng(
        scenario_seed(
            f"cifar-feature:{config.constrained_scenario.name}",
            config.target.seed + 410_000,
        )
    )
    constrained_w, weight_error_metrics = apply_constrained_fixed_weight_errors(
        params["w_opt"],
        config.constrained_scenario,
        constrained_rng,
    )
    policies: tuple[tuple[str, float | None], ...] = (
        ("scheduled_only", None),
        (
            "scheduled_plus_projection_mse_trigger",
            config.projection_mse_recalibration_threshold,
        ),
    )
    rows: list[dict[str, object]] = []
    for coding in config.source_codings:
        for calibration_samples in config.calibration_sample_counts:
            calibration_count = min(calibration_samples, train_features.shape[0])
            calibration_features = train_features[:calibration_count]
            for policy_name, threshold in policies:
                for physical in config.physical_scenarios:
                    calibration: tuple[np.ndarray, np.ndarray] | None = None
                    recalibration_count = 0
                    triggered_recalibration_count = 0
                    scenario_rows: list[dict[str, object]] = []
                    label = (
                        f"{coding.name}:cal{calibration_count}:"
                        f"{policy_name}:{physical.name}"
                    )
                    for step in range(physical.time_steps):
                        drift_rng = np.random.default_rng(
                            scenario_seed(
                                f"cifar-feature-drift:{label}:{step}",
                                config.target.seed + 420_000,
                            )
                        )
                        drifted_w = drifted_weights(
                            constrained_w,
                            params["w_opt"],
                            physical.thermal_drift_rate,
                            step,
                            drift_rng,
                        )
                        scheduled_recalibration = (
                            step % physical.recalibration_interval_steps == 0
                        )
                        if scheduled_recalibration:
                            calibration = _fit_feature_projection_calibration(
                                params,
                                calibration_features,
                                drifted_w,
                                physical.detector_noise,
                                coding,
                                np.random.default_rng(
                                    scenario_seed(
                                        f"cifar-feature-cal:{label}:{step}",
                                        config.target.seed + 430_000,
                                    )
                                ),
                            )
                            recalibration_count += 1
                        metrics = _evaluate_projected_features(
                            params,
                            test_features,
                            test_labels,
                            w_opt=drifted_w,
                            detector_noise=physical.detector_noise,
                            output_readout_noise=physical.output_readout_noise,
                            calibration=calibration,
                            source_coding=coding,
                            rng=np.random.default_rng(
                                scenario_seed(
                                    f"cifar-feature-eval:{label}:{step}",
                                    config.target.seed + 440_000,
                                )
                            ),
                        )
                        triggered_recalibration = False
                        if (
                            threshold is not None
                            and not scheduled_recalibration
                            and metrics["projection_mse_mean"] >= threshold
                        ):
                            calibration = _fit_feature_projection_calibration(
                                params,
                                calibration_features,
                                drifted_w,
                                physical.detector_noise,
                                coding,
                                np.random.default_rng(
                                    scenario_seed(
                                        f"cifar-feature-triggered-cal:{label}:{step}",
                                        config.target.seed + 450_000,
                                    )
                                ),
                            )
                            recalibration_count += 1
                            triggered_recalibration_count += 1
                            triggered_recalibration = True
                            metrics = _evaluate_projected_features(
                                params,
                                test_features,
                                test_labels,
                                w_opt=drifted_w,
                                detector_noise=physical.detector_noise,
                                output_readout_noise=physical.output_readout_noise,
                                calibration=calibration,
                                source_coding=coding,
                                rng=np.random.default_rng(
                                    scenario_seed(
                                        f"cifar-feature-triggered-eval:{label}:{step}",
                                        config.target.seed + 460_000,
                                    )
                                ),
                            )
                        accuracy = float(metrics["accuracy"])
                        scenario_rows.append(
                            {
                                "seed": config.target.seed,
                                "target": "cifar_spatial_feature_projection",
                                "physical_scenario": physical.name,
                                "time_step": step,
                                "target_accuracy": target_accuracy,
                                "calibrated_accuracy": accuracy,
                                "drop_vs_target": target_accuracy - accuracy,
                                "thermal_drift_rate": physical.thermal_drift_rate,
                                "detector_noise": physical.detector_noise,
                                "output_readout_noise": physical.output_readout_noise,
                                "recalibration_interval_steps": (
                                    physical.recalibration_interval_steps
                                ),
                                "architecture_profile": physical.architecture_profile,
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
                                "output_enob_equivalent": physical.output_enob_equivalent,
                                "source_coding": coding.name,
                                "source_bits": coding.bits,
                                "source_redundant_phases": coding.redundant_phases,
                                "source_noise": coding.source_noise,
                                "source_quantization_mse_mean": (
                                    metrics["source_quantization_mse_mean"]
                                ),
                                "source_quantization_max_error": (
                                    metrics["source_quantization_max_error"]
                                ),
                                "source_dac_pj_per_input": coding.dac_pj_per_input,
                                "shared_boundary_source_dac_energy_pj": (
                                    config.target.hidden_dim * coding.dac_pj_per_input
                                ),
                                "source_coding_latency_ns": coding.latency_ns,
                                "calibration_samples": calibration_count,
                                "recalibration_policy": policy_name,
                                "projection_mse_recalibration_threshold": (
                                    0.0 if threshold is None else threshold
                                ),
                                "scheduled_recalibration": scheduled_recalibration,
                                "triggered_recalibration": triggered_recalibration,
                                "recalibration_count": recalibration_count,
                                "triggered_recalibration_count": (
                                    triggered_recalibration_count
                                ),
                                "constrained_scenario": config.constrained_scenario.name,
                                "weight_error_mse": weight_error_metrics["weight_error_mse"],
                                "projection_mse_mean": metrics["projection_mse_mean"],
                                "projection_mse_max": metrics["projection_mse_max"],
                                "hidden_std": metrics["hidden_std"],
                                "below_50_percent": accuracy < 0.50,
                                "is_final_step": step == physical.time_steps - 1,
                            }
                        )
                    final_accuracy = float(scenario_rows[-1]["calibrated_accuracy"])
                    for row in scenario_rows:
                        row["final_step_accuracy"] = final_accuracy
                    rows.extend(scenario_rows)

    frame = pd.DataFrame(rows)
    summary = _summarize_feature_source_coding(frame)
    rows_csv = config.output_dir / "cifar_feature_source_coding_stress_rows.csv"
    summary_csv = config.output_dir / "cifar_feature_source_coding_stress_summary.csv"
    summary_md = config.output_dir / "cifar_feature_source_coding_stress_summary.md"
    manifest_path = config.output_dir / "cifar_feature_source_coding_stress_manifest.json"
    frame.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    payload: dict[str, object] = {
        "config": {
            "target": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.target).items()
            },
            "output_dir": str(config.output_dir),
            "calibration_sample_counts": list(config.calibration_sample_counts),
            "source_codings": [asdict(coding) for coding in config.source_codings],
            "constrained_scenario": asdict(config.constrained_scenario),
            "physical_scenarios": [asdict(scenario) for scenario in config.physical_scenarios],
            "projection_mse_recalibration_threshold": (
                config.projection_mse_recalibration_threshold
            ),
        },
        "rows": int(len(frame)),
        "summary_rows": int(len(summary)),
        "target_accuracy": target_accuracy,
        "calibrated_accuracy_min": float(frame["calibrated_accuracy"].min()),
        "calibrated_accuracy_mean": float(frame["calibrated_accuracy"].mean()),
        "calibrated_accuracy_max": float(frame["calibrated_accuracy"].max()),
        "steps_below_50_percent": int(frame["below_50_percent"].sum()),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
