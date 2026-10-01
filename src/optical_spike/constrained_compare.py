"""Compare constrained fixed-fabrication impact across optical targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optical_spike.baseline import BaselineConfig, run_baseline
from optical_spike.fabrication_constraints import (
    DEFAULT_CONSTRAINED_FIXED_SCENARIOS,
    ConstrainedFixedScenario,
    apply_constrained_fixed_weight_errors,
)
from optical_spike.reporting import markdown_table
from optical_spike.transformer_mlp import TransformerMLPConfig, run_transformer_mlp_target
from optical_spike.tunable import (
    compare_projection,
    evaluate_projection,
    fit_affine_calibration,
    load_train_and_test,
    project,
)


@dataclass(frozen=True)
class ConstrainedComparisonConfig:
    baseline: BaselineConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    mlp_dim: int = 256
    calibration_samples: int = 512
    adaptation_epochs: int = 8
    adaptation_learning_rate: float = 0.01
    scenarios: tuple[ConstrainedFixedScenario, ...] = DEFAULT_CONSTRAINED_FIXED_SCENARIOS


def scenario_seed(name: str, base: int) -> int:
    name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return base + name_seed


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["target", "scenario", "mode"], dropna=False)
    summary = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        accuracy_drop_mean=("accuracy_drop", "mean"),
        accuracy_drop_std=("accuracy_drop", "std"),
        projection_mse_mean=("projection_mse", "mean"),
        cosine_similarity_mean=("cosine_similarity_mean", "mean"),
        prediction_change_rate_mean=("prediction_change_rate", "mean"),
        correct_to_wrong_rate_mean=("correct_to_wrong_rate", "mean"),
        wrong_to_correct_rate_mean=("wrong_to_correct_rate", "mean"),
        weight_error_mse_mean=("weight_error_mse", "mean"),
        fan_pressure_mean=("fan_pressure", "mean"),
        effective_loss_mean=("effective_loss", "mean"),
        effective_crosstalk_mean=("effective_crosstalk", "mean"),
        runs=("seed", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def constrained_input_projection_rows(
    baseline: BaselineConfig,
    calibration_samples: int,
    scenarios: tuple[ConstrainedFixedScenario, ...],
) -> list[dict[str, object]]:
    baseline_metrics = run_baseline(baseline)
    weights_path = Path(str(baseline_metrics["artifacts"]["weights"]))
    weights = np.load(weights_path)
    params = {name: weights[name].astype(np.float32) for name in weights.files}
    train_images, _, test_images, test_labels = load_train_and_test(baseline)
    calibration_count = min(calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]

    rng = np.random.default_rng(baseline.seed + 90_000)
    ideal_test_projection = project(params, test_images, params["w_opt"], rng, detector_noise=0.0)
    ideal_calibration = project(
        params, calibration_images, params["w_opt"], rng, detector_noise=0.0
    )
    baseline_eval = evaluate_projection(params, test_images, test_labels, ideal_test_projection)
    baseline_accuracy = float(baseline_eval["accuracy"])

    rows: list[dict[str, object]] = [
        {
            "seed": baseline.seed,
            "target": "input_projection",
            "scenario": "digital_baseline",
            "mode": "baseline",
            "accuracy": baseline_accuracy,
            "accuracy_drop": 0.0,
            "loss": float(baseline_eval["loss"]),
            "projection_mse": 0.0,
            "cosine_similarity_mean": 1.0,
            "prediction_change_rate": 0.0,
            "correct_to_wrong_rate": 0.0,
            "wrong_to_correct_rate": 0.0,
            "weight_error_mse": 0.0,
            "fan_pressure": 0.0,
            "effective_loss": 0.0,
            "effective_crosstalk": 0.0,
        }
    ]

    for scenario in scenarios:
        scenario_rng = np.random.default_rng(scenario_seed(scenario.name, baseline.seed + 95_000))
        constrained_w, error_metrics = apply_constrained_fixed_weight_errors(
            params["w_opt"],
            scenario,
            scenario_rng,
        )
        raw_projection = project(
            params,
            test_images,
            constrained_w,
            scenario_rng,
            scenario.detector_noise,
        )
        observed_calibration = project(
            params,
            calibration_images,
            constrained_w,
            scenario_rng,
            scenario.detector_noise,
        )
        gain, bias = fit_affine_calibration(ideal_calibration, observed_calibration)
        calibrated_projection = raw_projection * gain + bias

        for mode, projection in (
            ("raw", raw_projection),
            ("calibrated", calibrated_projection),
        ):
            metrics = evaluate_projection(params, test_images, test_labels, projection)
            diagnostics = compare_projection(params, test_labels, ideal_test_projection, projection)
            projection_diag = diagnostics["projection"]
            prediction_diag = diagnostics["prediction"]
            rows.append(
                {
                    "seed": baseline.seed,
                    "target": "input_projection",
                    "scenario": scenario.name,
                    "mode": mode,
                    "accuracy": float(metrics["accuracy"]),
                    "accuracy_drop": baseline_accuracy - float(metrics["accuracy"]),
                    "loss": float(metrics["loss"]),
                    "projection_mse": projection_diag["projection_mse"],
                    "cosine_similarity_mean": projection_diag["cosine_similarity_mean"],
                    "prediction_change_rate": prediction_diag["prediction_change_rate"],
                    "correct_to_wrong_rate": prediction_diag["correct_to_wrong_rate"],
                    "wrong_to_correct_rate": prediction_diag["wrong_to_correct_rate"],
                    "weight_error_mse": error_metrics["weight_error_mse"],
                    "fan_pressure": error_metrics["routing"]["fan_pressure"],
                    "effective_loss": error_metrics["loss"]["effective"],
                    "effective_crosstalk": error_metrics["geometry"]["effective_crosstalk"],
                }
            )
    return rows


def constrained_mlp_rows(
    config: TransformerMLPConfig,
) -> list[dict[str, object]]:
    payload = run_transformer_mlp_target(config)
    baseline_accuracy = float(payload["baseline_test"]["accuracy"])  # type: ignore[index]
    rows: list[dict[str, object]] = [
        {
            "seed": config.seed,
            "target": "mlp_up_projection",
            "scenario": "digital_baseline",
            "mode": "baseline",
            "accuracy": baseline_accuracy,
            "accuracy_drop": 0.0,
            "loss": float(payload["baseline_test"]["loss"]),  # type: ignore[index]
            "projection_mse": 0.0,
            "cosine_similarity_mean": 1.0,
            "prediction_change_rate": 0.0,
            "correct_to_wrong_rate": 0.0,
            "wrong_to_correct_rate": 0.0,
            "weight_error_mse": 0.0,
            "fan_pressure": 0.0,
            "effective_loss": 0.0,
            "effective_crosstalk": 0.0,
        }
    ]
    for sweep in payload["constrained_fixed_sweeps"]:  # type: ignore[index]
        error_model = sweep["error_model"]
        for mode, metrics_key, drop_key, diagnostics_key in (
            ("raw", "raw_metrics", "raw_accuracy_drop", "raw_diagnostics"),
            (
                "calibrated",
                "calibrated_metrics",
                "calibrated_accuracy_drop",
                "calibrated_diagnostics",
            ),
        ):
            metrics = sweep[metrics_key]
            diagnostics = sweep[diagnostics_key]
            projection_diag = diagnostics["projection"]
            prediction_diag = diagnostics["prediction"]
            rows.append(
                {
                    "seed": config.seed,
                    "target": "mlp_up_projection",
                    "scenario": sweep["scenario"]["name"],
                    "mode": mode,
                    "accuracy": float(metrics["accuracy"]),
                    "accuracy_drop": float(sweep[drop_key]),
                    "loss": float(metrics["loss"]),
                    "projection_mse": projection_diag["projection_mse"],
                    "cosine_similarity_mean": projection_diag["cosine_similarity_mean"],
                    "prediction_change_rate": prediction_diag["prediction_change_rate"],
                    "correct_to_wrong_rate": prediction_diag["correct_to_wrong_rate"],
                    "wrong_to_correct_rate": prediction_diag["wrong_to_correct_rate"],
                    "weight_error_mse": error_model["weight_error_mse"],
                    "fan_pressure": error_model["routing"]["fan_pressure"],
                    "effective_loss": error_model["loss"]["effective"],
                    "effective_crosstalk": error_model["geometry"]["effective_crosstalk"],
                }
            )
    return rows


def run_constrained_comparison(config: ConstrainedComparisonConfig) -> dict[str, object]:
    output_dir = config.baseline.output_dir.parent / "constrained_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_baseline = replace(config.baseline, seed=seed, output_dir=seed_dir / "input_baseline")
        rows.extend(
            constrained_input_projection_rows(
                seed_baseline,
                config.calibration_samples,
                config.scenarios,
            )
        )
        mlp_config = TransformerMLPConfig(
            data_dir=config.baseline.data_dir,
            dataset=config.baseline.dataset,
            output_dir=seed_dir / "transformer_mlp_target",
            hidden_dim=config.baseline.hidden_dim,
            mlp_dim=config.mlp_dim,
            epochs=config.baseline.epochs,
            batch_size=config.baseline.batch_size,
            learning_rate=config.baseline.learning_rate,
            seed=seed,
            max_train_samples=config.baseline.max_train_samples,
            max_test_samples=config.baseline.max_test_samples,
            synthetic=config.baseline.synthetic,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.adaptation_epochs,
            adaptation_learning_rate=config.adaptation_learning_rate,
            constrained_fixed_scenarios=config.scenarios,
        )
        rows.extend(constrained_mlp_rows(mlp_config))
        seed_manifests.append({"seed": seed, "directory": str(seed_dir)})

    combined = pd.DataFrame(rows)
    summary = summarize_rows(combined)
    rows_csv = output_dir / "constrained_comparison_rows.csv"
    summary_csv = output_dir / "constrained_comparison_summary.csv"
    summary_md = output_dir / "constrained_comparison_summary.md"
    manifest_path = output_dir / "constrained_comparison_manifest.json"
    combined.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "baseline": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.baseline).items()
            },
            "seeds": list(config.seeds),
            "mlp_dim": config.mlp_dim,
            "calibration_samples": config.calibration_samples,
            "adaptation_epochs": config.adaptation_epochs,
            "adaptation_learning_rate": config.adaptation_learning_rate,
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
            "full_split": config.baseline.max_train_samples is None
            and config.baseline.max_test_samples is None,
        },
        "seed_manifests": seed_manifests,
        "rows": int(len(combined)),
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
