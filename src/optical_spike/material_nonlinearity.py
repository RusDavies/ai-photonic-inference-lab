"""Material-inspired nonlinear transfer checks for the MLP-up optical target."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from optical_spike.calibration_cost import train_transformer
from optical_spike.reporting import markdown_table
from optical_spike.transformer_mlp import (
    TransformerMLPConfig,
    embed_inputs,
    evaluate_mlp_up_projection,
    forward_from_mlp_up_projection,
    project_mlp_up,
)
from optical_spike.tunable import fit_affine_calibration


@dataclass(frozen=True)
class MaterialNonlinearityScenario:
    name: str
    material_family: str
    kind: str
    strength: float


DEFAULT_MATERIAL_NONLINEARITIES: tuple[MaterialNonlinearityScenario, ...] = (
    MaterialNonlinearityScenario(
        "saturable_absorber_mild",
        "saturable_absorber",
        "saturable_absorption",
        0.25,
    ),
    MaterialNonlinearityScenario(
        "reverse_saturable_absorber",
        "reverse_saturable_absorber",
        "reverse_saturable_absorption",
        0.20,
    ),
    MaterialNonlinearityScenario(
        "two_photon_absorption",
        "silicon_or_germanium_tpa",
        "two_photon_absorption",
        0.18,
    ),
    MaterialNonlinearityScenario(
        "kerr_phase_cubic",
        "kerr_or_carrier_phase",
        "cubic_phase",
        0.08,
    ),
    MaterialNonlinearityScenario(
        "electro_optic_sigmoid",
        "hybrid_electro_optic",
        "sigmoid_activation",
        1.50,
    ),
)


@dataclass(frozen=True)
class MaterialNonlinearityConfig:
    transformer: TransformerMLPConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    calibration_samples: int = 512
    scenarios: tuple[MaterialNonlinearityScenario, ...] = DEFAULT_MATERIAL_NONLINEARITIES


def apply_material_transfer(
    projection: np.ndarray,
    scenario: MaterialNonlinearityScenario,
) -> np.ndarray:
    scale = max(float(np.std(projection)), 1e-8)
    normalized = projection / scale
    if scenario.kind == "saturable_absorption":
        transferred = projection / (1.0 + scenario.strength * np.abs(normalized))
    elif scenario.kind == "reverse_saturable_absorption":
        attenuation = 1.0 + scenario.strength * np.abs(normalized)
        transferred = projection / attenuation
    elif scenario.kind == "two_photon_absorption":
        transferred = projection / (1.0 + scenario.strength * normalized * normalized)
    elif scenario.kind == "cubic_phase":
        transferred = projection + scenario.strength * projection * normalized * normalized
    elif scenario.kind == "sigmoid_activation":
        transferred = scale * (2.0 / (1.0 + np.exp(-scenario.strength * normalized)) - 1.0)
    else:
        raise ValueError(f"Unknown material nonlinearity kind: {scenario.kind}")
    return transferred.astype(np.float32)


def material_activation(
    projection: np.ndarray,
    scenario: MaterialNonlinearityScenario,
) -> np.ndarray:
    scale = max(float(np.std(projection)), 1e-8)
    normalized = projection / scale
    if scenario.kind == "saturable_absorption":
        activated = np.maximum(projection, 0.0) / (
            1.0 + scenario.strength * np.maximum(normalized, 0.0)
        )
    elif scenario.kind == "reverse_saturable_absorption":
        positive = np.maximum(projection, 0.0)
        activated = positive / (1.0 + scenario.strength * np.maximum(normalized, 0.0))
    elif scenario.kind == "two_photon_absorption":
        positive = np.maximum(projection, 0.0)
        activated = positive / (1.0 + scenario.strength * normalized * normalized)
    elif scenario.kind == "cubic_phase":
        positive = np.maximum(projection, 0.0)
        activated = positive + scenario.strength * positive * normalized * normalized
    elif scenario.kind == "sigmoid_activation":
        activated = scale / (1.0 + np.exp(-scenario.strength * normalized))
    else:
        raise ValueError(f"Unknown material nonlinearity kind: {scenario.kind}")
    return activated.astype(np.float32)


def evaluate_from_mlp_activation(
    params: dict[str, np.ndarray],
    inputs: np.ndarray,
    labels: np.ndarray,
    mlp_activation: np.ndarray,
) -> dict[str, object]:
    embed = embed_inputs(params, inputs)
    mlp_down_projection = mlp_activation @ params["w_mlp_down"] + params["b_mlp_down"]
    residual = np.maximum(embed + mlp_down_projection, 0.0)
    logits = residual @ params["w_head"] + params["b_head"]
    exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    predictions = np.argmax(probs, axis=1)
    clipped = np.clip(probs[np.arange(labels.shape[0]), labels], 1e-8, 1.0)
    return {
        "loss": float(-np.mean(np.log(clipped))),
        "accuracy": float(np.mean(predictions == labels)),
        "mlp_activation_mean": float(np.mean(mlp_activation)),
        "mlp_activation_std": float(np.std(mlp_activation)),
    }


def run_seed_material_nonlinearities(
    config: MaterialNonlinearityConfig,
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
        calibration_samples=config.calibration_samples,
        adaptation_epochs=config.transformer.adaptation_epochs,
        adaptation_learning_rate=config.transformer.adaptation_learning_rate,
    )
    params, train_images, _, test_images, test_labels, metrics = train_transformer(transformer)
    baseline_accuracy = float(metrics["baseline_test"]["accuracy"])  # type: ignore[index]
    rng = np.random.default_rng(seed + 120_000)
    calibration_count = min(config.calibration_samples, train_images.shape[0])
    calibration_images = train_images[:calibration_count]
    ideal_test_projection = project_mlp_up(params, test_images, params["w_mlp_up"], rng, 0.0)
    ideal_calibration_projection = project_mlp_up(
        params,
        calibration_images,
        params["w_mlp_up"],
        rng,
        0.0,
    )
    _, ideal_test_cache = forward_from_mlp_up_projection(
        params,
        test_images,
        ideal_test_projection,
    )
    _, ideal_calibration_cache = forward_from_mlp_up_projection(
        params,
        calibration_images,
        ideal_calibration_projection,
    )
    ideal_test_activation = ideal_test_cache["mlp_up"]
    ideal_calibration_activation = ideal_calibration_cache["mlp_up"]

    rows: list[dict[str, object]] = []
    for scenario in config.scenarios:
        distorted_test_projection = apply_material_transfer(ideal_test_projection, scenario)
        distorted_calibration_projection = apply_material_transfer(
            ideal_calibration_projection,
            scenario,
        )
        projection_gain, projection_bias = fit_affine_calibration(
            ideal_calibration_projection,
            distorted_calibration_projection,
        )
        calibrated_projection = distorted_test_projection * projection_gain + projection_bias
        for mode, projection in (
            ("distortion_raw", distorted_test_projection),
            ("distortion_calibrated", calibrated_projection),
        ):
            result = evaluate_mlp_up_projection(params, test_images, test_labels, projection)
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario.name,
                    "material_family": scenario.material_family,
                    "kind": scenario.kind,
                    "mode": mode,
                    "accuracy": float(result["accuracy"]),
                    "accuracy_drop": baseline_accuracy - float(result["accuracy"]),
                    "loss": float(result["loss"]),
                    "calibration_samples": calibration_count,
                    "projection_mse": float(np.mean((projection - ideal_test_projection) ** 2)),
                    "activation_mse": 0.0,
                }
            )

        activated_test = material_activation(ideal_test_projection, scenario)
        activated_calibration = material_activation(ideal_calibration_projection, scenario)
        activation_gain, activation_bias = fit_affine_calibration(
            ideal_calibration_activation,
            activated_calibration,
        )
        calibrated_activation = activated_test * activation_gain + activation_bias
        for mode, activation in (
            ("activation_raw", activated_test),
            ("activation_calibrated", calibrated_activation),
        ):
            result = evaluate_from_mlp_activation(params, test_images, test_labels, activation)
            rows.append(
                {
                    "seed": seed,
                    "scenario": scenario.name,
                    "material_family": scenario.material_family,
                    "kind": scenario.kind,
                    "mode": mode,
                    "accuracy": float(result["accuracy"]),
                    "accuracy_drop": baseline_accuracy - float(result["accuracy"]),
                    "loss": float(result["loss"]),
                    "calibration_samples": calibration_count,
                    "projection_mse": 0.0,
                    "activation_mse": float(np.mean((activation - ideal_test_activation) ** 2)),
                }
            )
    return rows, metrics


def summarize_material_rows(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["scenario", "material_family", "kind", "mode"], dropna=False)
    summary = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_drop_mean=("accuracy_drop", "mean"),
        accuracy_drop_std=("accuracy_drop", "std"),
        loss_mean=("loss", "mean"),
        projection_mse_mean=("projection_mse", "mean"),
        activation_mse_mean=("activation_mse", "mean"),
        runs=("seed", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def run_material_nonlinearity_sweeps(config: MaterialNonlinearityConfig) -> dict[str, object]:
    output_dir = config.transformer.output_dir.parent / "material_nonlinearity"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    seed_manifests: list[dict[str, object]] = []
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_rows, seed_metrics = run_seed_material_nonlinearities(config, seed, seed_dir)
        rows.extend(seed_rows)
        seed_manifests.append(
            {
                "seed": seed,
                "directory": str(seed_dir),
                "baseline_test": seed_metrics["baseline_test"],
            }
        )

    frame = pd.DataFrame(rows)
    summary = summarize_material_rows(frame)
    rows_csv = output_dir / "material_nonlinearity_rows.csv"
    summary_csv = output_dir / "material_nonlinearity_summary.csv"
    summary_md = output_dir / "material_nonlinearity_summary.md"
    manifest_path = output_dir / "material_nonlinearity_manifest.json"
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
            "calibration_samples": config.calibration_samples,
            "scenarios": [asdict(scenario) for scenario in config.scenarios],
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
