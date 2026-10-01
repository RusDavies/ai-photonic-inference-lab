"""Multi-seed runner for full-split optical projection spike checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import pandas as pd

from optical_spike.adaptation import AdaptationConfig, run_adaptation_modes
from optical_spike.baseline import BaselineConfig, run_baseline
from optical_spike.fixed import FixedSweepConfig, run_fixed_error_sweeps
from optical_spike.nonlinear import NonlinearSweepConfig, run_nonlinear_variant_sweeps
from optical_spike.quantization import QuantizationSweepConfig, run_quantized_projection_sweeps
from optical_spike.reporting import ReportConfig, generate_report, markdown_table
from optical_spike.tunable import TunableSweepConfig, run_tunable_error_sweeps


@dataclass(frozen=True)
class MultiSeedConfig:
    baseline: BaselineConfig
    seeds: tuple[int, ...] = (7, 11, 13)
    quantization_bits: tuple[int, ...] = (8, 6, 4, 3, 2)
    calibration_samples: int = 512
    adaptation_epochs: int = 8
    adaptation_learning_rate: float = 0.01


def parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def summarize_multi_seed(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["family", "scenario", "mode"], dropna=False)
    summary = grouped.agg(
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        accuracy_min=("accuracy", "min"),
        accuracy_max=("accuracy", "max"),
        accuracy_drop_mean=("accuracy_drop", "mean"),
        accuracy_drop_std=("accuracy_drop", "std"),
        loss_mean=("loss", "mean"),
        projection_mse_mean=("projection_mse", "mean"),
        cosine_similarity_mean=("cosine_similarity_mean", "mean"),
        feature_mean_shift_l2_mean=("feature_mean_shift_l2", "mean"),
        feature_std_shift_mean_abs_mean=("feature_std_shift_mean_abs", "mean"),
        confusion_delta_l1_mean=("confusion_delta_l1", "mean"),
        prediction_change_rate_mean=("prediction_change_rate", "mean"),
        correct_to_wrong_rate_mean=("correct_to_wrong_rate", "mean"),
        wrong_to_correct_rate_mean=("wrong_to_correct_rate", "mean"),
        runs=("seed", "count"),
    ).reset_index()
    return summary.fillna(0.0)


def run_single_seed(config: MultiSeedConfig, seed: int, seed_dir: Path) -> dict[str, object]:
    baseline = replace(
        config.baseline,
        seed=seed,
        output_dir=seed_dir / "baseline",
    )
    run_baseline(baseline)
    run_quantized_projection_sweeps(
        QuantizationSweepConfig(
            baseline=baseline,
            bits=config.quantization_bits,
        )
    )
    run_tunable_error_sweeps(
        TunableSweepConfig(
            baseline=baseline,
            calibration_samples=config.calibration_samples,
        )
    )
    run_fixed_error_sweeps(
        FixedSweepConfig(
            baseline=baseline,
            calibration_samples=config.calibration_samples,
        )
    )
    run_nonlinear_variant_sweeps(
        NonlinearSweepConfig(
            baseline=baseline,
            calibration_samples=config.calibration_samples,
        )
    )
    run_adaptation_modes(
        AdaptationConfig(
            baseline=baseline,
            calibration_samples=config.calibration_samples,
            adaptation_epochs=config.adaptation_epochs,
            adaptation_learning_rate=config.adaptation_learning_rate,
        )
    )
    return generate_report(ReportConfig(output_dir=seed_dir))


def run_multi_seed(config: MultiSeedConfig) -> dict[str, object]:
    output_dir = config.baseline.output_dir.parent / "multi_seed"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []
    for seed in config.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        manifest = run_single_seed(config, seed, seed_dir)
        manifests.append({"seed": seed, "manifest": manifest})
        summary_csv = Path(str(manifest["artifacts"]["summary_csv"]))  # type: ignore[index]
        frame = pd.read_csv(summary_csv)
        frame.insert(0, "seed", seed)
        frames.append(frame)

    combined_rows = pd.concat(frames, ignore_index=True)
    aggregate = summarize_multi_seed(combined_rows)
    combined_csv = output_dir / "multi_seed_rows.csv"
    aggregate_csv = output_dir / "multi_seed_summary.csv"
    aggregate_md = output_dir / "multi_seed_summary.md"
    manifest_path = output_dir / "multi_seed_manifest.json"
    combined_rows.to_csv(combined_csv, index=False)
    aggregate.to_csv(aggregate_csv, index=False)
    aggregate_md.write_text(markdown_table(aggregate), encoding="utf-8")

    payload: dict[str, object] = {
        "config": {
            "baseline": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(config.baseline).items()
            },
            "seeds": list(config.seeds),
            "quantization_bits": list(config.quantization_bits),
            "calibration_samples": config.calibration_samples,
            "adaptation_epochs": config.adaptation_epochs,
            "adaptation_learning_rate": config.adaptation_learning_rate,
            "full_split": config.baseline.max_train_samples is None
            and config.baseline.max_test_samples is None,
        },
        "seed_manifests": manifests,
        "rows": int(len(combined_rows)),
        "summary_rows": int(len(aggregate)),
        "artifacts": {
            "combined_rows_csv": str(combined_csv),
            "summary_csv": str(aggregate_csv),
            "summary_md": str(aggregate_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
