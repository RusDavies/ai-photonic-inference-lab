"""Calibration cost and drift cadence sweeps for optical transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from optical_spike.fabrication_constraints import (
    DEFAULT_CONSTRAINED_FIXED_SCENARIOS,
    ConstrainedFixedScenario,
    apply_constrained_fixed_weight_errors,
)
from optical_spike.reporting import markdown_table
from optical_spike.transformer_mlp import (
    TransformerMLPConfig,
    evaluate,
    evaluate_mlp_up_projection,
    init_params,
    load_train_and_test,
    project_mlp_up,
    scenario_seed,
    train_epoch,
)
from optical_spike.tunable import fit_affine_calibration


@dataclass(frozen=True)
class CalibrationCostConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    calibration_sample_counts: tuple[int, ...] = (32, 128, 512, 2048)
    recalibration_intervals: tuple[int, ...] = (1, 4, 16)
    drift_rates: tuple[float, ...] = (0.0, 0.002, 0.005)
    time_steps: int = 16
    scenario: ConstrainedFixedScenario = DEFAULT_CONSTRAINED_FIXED_SCENARIOS[-1]


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("At least one integer value is required")
    return parsed


def parse_float_tuple(value: str) -> tuple[float, ...]:
    parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not parsed:
        raise ValueError("At least one float value is required")
    return parsed


def train_transformer(
    config: TransformerMLPConfig,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, object],
]:
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
    metrics: dict[str, object] = {
        "history": history,
        "baseline_train": baseline_train,
        "baseline_test": baseline_test,
        "elapsed_seconds": time.time() - started,
    }
    return params, train_images, train_labels, test_images, test_labels, metrics


def drifted_weights(
    base_weights: np.ndarray,
    ideal_weights: np.ndarray,
    drift_rate: float,
    step: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if drift_rate <= 0.0 or step <= 0:
        return base_weights
    max_abs = max(float(np.max(np.abs(ideal_weights))), 1e-8)
    drift_scale = drift_rate * np.sqrt(float(step)) * max_abs
    drift = rng.normal(0.0, drift_scale, size=base_weights.shape).astype(np.float32)
    return (base_weights + drift).astype(np.float32)


def calibration_overhead(
    *,
    calibration_samples: int,
    recalibrations: int,
    output_dim: int,
    time_steps: int,
) -> dict[str, float | int]:
    affine_values = 2 * output_dim
    total_calibration_examples = calibration_samples * recalibrations
    total_control_values = affine_values * recalibrations
    return {
        "recalibrations": recalibrations,
        "total_calibration_examples": total_calibration_examples,
        "calibration_examples_per_step": total_calibration_examples / time_steps,
        "affine_control_values_per_calibration": affine_values,
        "total_affine_control_values": total_control_values,
        "affine_values_per_step": total_control_values / time_steps,
        "digital_correction_ops_per_inference": affine_values,
    }


def run_seed_calibration_cost(
    config: CalibrationCostConfig,
    seed: int,
    seed_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    transformer = TransformerMLPConfig(
        data_dir=config.transformer.data_dir,
        dataset=config.transformer.dataset,
        output_dir=seed_dir / "transformer_mlp",
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
    params, train_images, _, test_images, test_labels, metrics = train_transformer(transformer)
    baseline_accuracy = float(metrics["baseline_test"]["accuracy"])  # type: ignore[index]
    scenario_rng = np.random.default_rng(scenario_seed(config.scenario.name, seed + 105_000))
    hardware_w, error_metrics = apply_constrained_fixed_weight_errors(
        params["w_mlp_up"],
        config.scenario,
        scenario_rng,
    )
    rows: list[dict[str, object]] = []
    max_calibration_count = min(max(config.calibration_sample_counts), train_images.shape[0])
    calibration_pool = train_images[:max_calibration_count]
    for sample_count in config.calibration_sample_counts:
        actual_samples = min(sample_count, calibration_pool.shape[0])
        calibration_images = calibration_pool[:actual_samples]
        ideal_calibration = project_mlp_up(
            params,
            calibration_images,
            params["w_mlp_up"],
            scenario_rng,
            0.0,
        )
        for interval in config.recalibration_intervals:
            recalibrations = ((config.time_steps - 1) // interval) + 1
            overhead = calibration_overhead(
                calibration_samples=actual_samples,
                recalibrations=recalibrations,
                output_dim=config.transformer.mlp_dim,
                time_steps=config.time_steps,
            )
            for drift_rate in config.drift_rates:
                current_gain: np.ndarray | None = None
                current_bias: np.ndarray | None = None
                accuracies: list[float] = []
                drops: list[float] = []
                raw_accuracies: list[float] = []
                for step in range(config.time_steps):
                    drift_rng = np.random.default_rng(
                        scenario_seed(
                            f"{config.scenario.name}:{sample_count}:{interval}:{drift_rate}:{step}",
                            seed + 110_000,
                        )
                    )
                    step_w = drifted_weights(
                        hardware_w,
                        params["w_mlp_up"],
                        drift_rate,
                        step,
                        drift_rng,
                    )
                    if step % interval == 0 or current_gain is None or current_bias is None:
                        observed_calibration = project_mlp_up(
                            params,
                            calibration_images,
                            step_w,
                            drift_rng,
                            config.scenario.detector_noise,
                        )
                        current_gain, current_bias = fit_affine_calibration(
                            ideal_calibration,
                            observed_calibration,
                        )
                    raw_projection = project_mlp_up(
                        params,
                        test_images,
                        step_w,
                        drift_rng,
                        config.scenario.detector_noise,
                    )
                    raw_metrics = evaluate_mlp_up_projection(
                        params,
                        test_images,
                        test_labels,
                        raw_projection,
                    )
                    calibrated_projection = raw_projection * current_gain + current_bias
                    calibrated_metrics = evaluate_mlp_up_projection(
                        params,
                        test_images,
                        test_labels,
                        calibrated_projection,
                    )
                    raw_accuracies.append(float(raw_metrics["accuracy"]))
                    accuracies.append(float(calibrated_metrics["accuracy"]))
                    drops.append(baseline_accuracy - float(calibrated_metrics["accuracy"]))

                rows.append(
                    {
                        "seed": seed,
                        "scenario": config.scenario.name,
                        "calibration_samples": actual_samples,
                        "recalibration_interval_steps": interval,
                        "drift_rate": drift_rate,
                        "time_steps": config.time_steps,
                        "baseline_accuracy": baseline_accuracy,
                        "raw_accuracy_mean": float(np.mean(raw_accuracies)),
                        "calibrated_accuracy_mean": float(np.mean(accuracies)),
                        "calibrated_accuracy_min": float(np.min(accuracies)),
                        "calibrated_accuracy_drop_mean": float(np.mean(drops)),
                        "calibrated_accuracy_drop_max": float(np.max(drops)),
                        "weight_error_mse": error_metrics["weight_error_mse"],
                        "fan_pressure": error_metrics["routing"]["fan_pressure"],
                        "effective_loss": error_metrics["loss"]["effective"],
                        "effective_crosstalk": error_metrics["geometry"]["effective_crosstalk"],
                        **overhead,
                    }
                )
    return rows, metrics


def summarize_calibration_cost(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(
        ["scenario", "calibration_samples", "recalibration_interval_steps", "drift_rate"],
        dropna=False,
    )
    summary = grouped.agg(
        baseline_accuracy_mean=("baseline_accuracy", "mean"),
        raw_accuracy_mean=("raw_accuracy_mean", "mean"),
        calibrated_accuracy_mean=("calibrated_accuracy_mean", "mean"),
        calibrated_accuracy_min_mean=("calibrated_accuracy_min", "mean"),
        calibrated_accuracy_drop_mean=("calibrated_accuracy_drop_mean", "mean"),
        calibrated_accuracy_drop_max_mean=("calibrated_accuracy_drop_max", "mean"),
        total_calibration_examples_mean=("total_calibration_examples", "mean"),
        calibration_examples_per_step_mean=("calibration_examples_per_step", "mean"),
        affine_values_per_step_mean=("affine_values_per_step", "mean"),
        digital_correction_ops_per_inference_mean=(
            "digital_correction_ops_per_inference",
            "mean",
        ),
        runs=("seed", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def run_calibration_cost(config: CalibrationCostConfig) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "calibration_cost"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_rows, seed_metrics = run_seed_calibration_cost(config, seed, seed_dir)
        rows.extend(seed_rows)
        seed_manifests.append(
            {
                "seed": seed,
                "directory": str(seed_dir),
                "baseline_test": seed_metrics["baseline_test"],
            }
        )

    frame = pd.DataFrame(rows)
    summary = summarize_calibration_cost(frame)
    rows_csv = output_dir / "calibration_cost_rows.csv"
    summary_csv = output_dir / "calibration_cost_summary.csv"
    summary_md = output_dir / "calibration_cost_summary.md"
    manifest_path = output_dir / "calibration_cost_manifest.json"
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
            "calibration_sample_counts": list(config.calibration_sample_counts),
            "recalibration_intervals": list(config.recalibration_intervals),
            "drift_rates": list(config.drift_rates),
            "time_steps": config.time_steps,
            "scenario": asdict(config.scenario),
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
