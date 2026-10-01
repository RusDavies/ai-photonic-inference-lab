"""Ablation-style reductions for source-coding calibration stress rows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd

from optical_spike.reporting import markdown_table


ACCURACY_FLOOR = 0.85


@dataclass(frozen=True)
class SourceCodingAblationReportConfig:
    source_coding_stress_rows: Path
    output_dir: Path
    accuracy_floor: float = ACCURACY_FLOOR


FACTOR_DEFINITIONS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("source_coding", ("source_coding", "source_bits", "source_redundant_phases")),
    ("thermal_drift", ("thermal_drift_rate", "thermal_excursion_k_per_step")),
    ("detector_noise", ("detector_noise", "detector_photons_per_sample")),
    ("output_readout_noise", ("output_readout_noise", "output_enob_equivalent")),
    ("recalibration_policy", ("calibration_samples", "recalibration_policy")),
    ("training_margin", ("seed", "trained_chain_accuracy")),
)


RUN_ID_COLUMNS = (
    "seed",
    "scenario",
    "chain_length",
    "physical_scenario",
    "source_coding",
    "calibration_samples",
    "recalibration_policy",
)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], path: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")


def _format_level(row: pd.Series, columns: tuple[str, ...]) -> str:
    return ", ".join(f"{column}={row[column]}" for column in columns)


def _summarize_factor(
    frame: pd.DataFrame,
    *,
    factor: str,
    columns: tuple[str, ...],
    accuracy_floor: float,
) -> pd.DataFrame:
    grouped = (
        frame.groupby(list(columns), dropna=False)
        .agg(
            rows=("seed", "count"),
            run_scenarios=("_run_id", "nunique"),
            calibrated_accuracy_mean=("calibrated_accuracy", "mean"),
            calibrated_accuracy_min=("calibrated_accuracy", "min"),
            drop_vs_trained_mean=("drop_vs_trained", "mean"),
            drop_vs_trained_max=("drop_vs_trained", "max"),
            projection_mse_mean=("projection_mse_mean", "mean"),
            projection_mse_max=("projection_mse_max", "max"),
            source_quantization_mse_mean=("source_quantization_mse_mean", "mean"),
            recalibrations_mean=("recalibration_count", "mean"),
            triggered_recalibrations_mean=("triggered_recalibration_count", "mean"),
            steps_below_floor=("below_floor", "sum"),
        )
        .reset_index()
    )
    grouped.insert(0, "factor", factor)
    grouped["accuracy_mean_percent"] = grouped["calibrated_accuracy_mean"] * 100.0
    grouped["accuracy_min_percent"] = grouped["calibrated_accuracy_min"] * 100.0
    grouped["min_margin_pp"] = (grouped["calibrated_accuracy_min"] - accuracy_floor) * 100.0
    grouped["drop_vs_trained_mean_pp"] = grouped["drop_vs_trained_mean"] * 100.0
    grouped["drop_vs_trained_max_pp"] = grouped["drop_vs_trained_max"] * 100.0
    report_columns = [
        "factor",
        *columns,
        "rows",
        "run_scenarios",
        "accuracy_mean_percent",
        "accuracy_min_percent",
        "min_margin_pp",
        "drop_vs_trained_mean_pp",
        "drop_vs_trained_max_pp",
        "projection_mse_mean",
        "projection_mse_max",
        "source_quantization_mse_mean",
        "recalibrations_mean",
        "triggered_recalibrations_mean",
        "steps_below_floor",
    ]
    return grouped[report_columns].sort_values(
        ["steps_below_floor", "min_margin_pp", "drop_vs_trained_max_pp"],
        ascending=[True, True, False],
    )


def _rank_factors(
    factor_summaries: dict[str, pd.DataFrame],
    factor_columns: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for factor, summary in factor_summaries.items():
        worst = summary.sort_values(
            ["steps_below_floor", "min_margin_pp", "drop_vs_trained_max_pp"],
            ascending=[False, True, False],
        ).iloc[0]
        rows.append(
            {
                "factor": factor,
                "levels": int(len(summary)),
                "accuracy_range_pp": float(
                    summary["accuracy_mean_percent"].max()
                    - summary["accuracy_mean_percent"].min()
                ),
                "worst_min_margin_pp": float(summary["min_margin_pp"].min()),
                "worst_drop_vs_trained_pp": float(summary["drop_vs_trained_max_pp"].max()),
                "steps_below_floor": int(summary["steps_below_floor"].sum()),
                "worst_level": _format_level(worst, factor_columns[factor]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["steps_below_floor", "worst_min_margin_pp", "accuracy_range_pp"],
        ascending=[False, True, False],
    )


def _write_markdown_report(
    path: Path,
    *,
    source_path: Path,
    rank: pd.DataFrame,
    factor_summaries: dict[str, pd.DataFrame],
    accuracy_floor: float,
) -> None:
    lines = [
        "# Source-Coding Calibration Ablation Report",
        "",
        f"Source rows: `{source_path}`",
        f"Accuracy floor: `{accuracy_floor * 100:.2f}%`",
        "",
        "## Factor Ranking",
        "",
        markdown_table(rank),
    ]
    for factor, summary in factor_summaries.items():
        lines.extend(
            [
                f"## {factor.replace('_', ' ').title()}",
                "",
                markdown_table(summary),
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_source_coding_ablation_report(
    config: SourceCodingAblationReportConfig,
) -> dict[str, object]:
    required_columns = (
        *RUN_ID_COLUMNS,
        "calibrated_accuracy",
        "drop_vs_trained",
        "trained_chain_accuracy",
        "projection_mse_mean",
        "projection_mse_max",
        "source_quantization_mse_mean",
        "recalibration_count",
        "triggered_recalibration_count",
    )
    header = pd.read_csv(config.source_coding_stress_rows, nrows=0)
    _require_columns(header, required_columns, config.source_coding_stress_rows)
    frame = pd.read_csv(config.source_coding_stress_rows)
    frame["below_floor"] = frame["calibrated_accuracy"] < config.accuracy_floor
    frame["_run_id"] = frame[list(RUN_ID_COLUMNS)].astype(str).agg("|".join, axis=1)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    factor_summaries: dict[str, pd.DataFrame] = {}
    factor_columns: dict[str, tuple[str, ...]] = {}
    artifacts: dict[str, str] = {}
    for factor, columns in FACTOR_DEFINITIONS:
        _require_columns(frame, columns, config.source_coding_stress_rows)
        summary = _summarize_factor(
            frame,
            factor=factor,
            columns=columns,
            accuracy_floor=config.accuracy_floor,
        )
        factor_summaries[factor] = summary
        factor_columns[factor] = columns
        summary_path = config.output_dir / f"{factor}_ablation_summary.csv"
        summary.to_csv(summary_path, index=False)
        artifacts[f"{factor}_summary_csv"] = str(summary_path)

    rank = _rank_factors(factor_summaries, factor_columns)
    rank_csv = config.output_dir / "factor_ablation_ranking.csv"
    report_md = config.output_dir / "source_coding_ablation_report.md"
    manifest_path = config.output_dir / "source_coding_ablation_manifest.json"
    rank.to_csv(rank_csv, index=False)
    _write_markdown_report(
        report_md,
        source_path=config.source_coding_stress_rows,
        rank=rank,
        factor_summaries=factor_summaries,
        accuracy_floor=config.accuracy_floor,
    )
    artifacts.update(
        {
            "factor_ranking_csv": str(rank_csv),
            "report_md": str(report_md),
            "manifest": str(manifest_path),
        }
    )
    payload: dict[str, object] = {
        "source_coding_stress_rows": str(config.source_coding_stress_rows),
        "rows": int(len(frame)),
        "run_scenarios": int(frame["_run_id"].nunique()),
        "accuracy_floor": config.accuracy_floor,
        "steps_below_floor": int(frame["below_floor"].sum()),
        "min_accuracy_percent": float(frame["calibrated_accuracy"].min() * 100.0),
        "mean_accuracy_percent": float(frame["calibrated_accuracy"].mean() * 100.0),
        "factor_ranking_rows": int(len(rank)),
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
