"""Diagnose persistent source-coding stress failures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from optical_spike.downstream_recoverability import RUN_KEYS, _max_consecutive
from optical_spike.reporting import markdown_table


@dataclass(frozen=True)
class PersistentFailureDiagnosisConfig:
    source_coding_stress_rows: Path
    output_dir: Path
    accuracy_floor: float = 0.85
    lift_scenarios: tuple[float, ...] = (0.0, 0.0025, 0.005, 0.01, 0.0125, 0.015, 0.02)


def build_run_diagnosis_rows(
    source_rows: pd.DataFrame,
    config: PersistentFailureDiagnosisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in source_rows.sort_values("time_step").groupby(RUN_KEYS):
        accuracies = group["calibrated_accuracy"].astype(float).tolist()
        below = [accuracy < config.accuracy_floor for accuracy in accuracies]
        baseline_accuracy = float(group["baseline_relu_accuracy"].iloc[0])
        trained_accuracy = float(group["trained_chain_accuracy"].iloc[0])
        min_accuracy = min(accuracies)
        rows.append(
            {
                **dict(zip(RUN_KEYS, key)),
                "baseline_relu_accuracy": baseline_accuracy,
                "trained_chain_accuracy": trained_accuracy,
                "trained_margin_vs_floor": trained_accuracy - config.accuracy_floor,
                "calibrated_accuracy_min": min_accuracy,
                "calibrated_accuracy_mean": sum(accuracies) / len(accuracies),
                "calibrated_margin_vs_floor": min_accuracy - config.accuracy_floor,
                "drop_trained_to_min": trained_accuracy - min_accuracy,
                "below_count": int(sum(below)),
                "max_consecutive_below": _max_consecutive(below),
                "persistent_below": _max_consecutive(below) >= 2,
                "all_steps_below": int(sum(below)) == len(below),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_seed(run_rows: pd.DataFrame) -> pd.DataFrame:
    return (
        run_rows.groupby("seed", dropna=False)
        .agg(
            baseline_relu_accuracy=("baseline_relu_accuracy", "first"),
            trained_chain_accuracy=("trained_chain_accuracy", "first"),
            trained_margin_vs_floor=("trained_margin_vs_floor", "first"),
            run_scenarios=("seed", "count"),
            scenarios_with_any_below=("below_count", lambda values: int((values > 0).sum())),
            persistent_scenarios=("persistent_below", "sum"),
            all_steps_below_scenarios=("all_steps_below", "sum"),
            worst_calibrated_accuracy=("calibrated_accuracy_min", "min"),
            mean_calibrated_accuracy=("calibrated_accuracy_mean", "mean"),
            worst_drop_trained_to_min=("drop_trained_to_min", "max"),
            median_drop_trained_to_min=("drop_trained_to_min", "median"),
        )
        .reset_index()
        .sort_values(["persistent_scenarios", "worst_calibrated_accuracy"], ascending=[False, True])
    )


def summarize_by_factor(run_rows: pd.DataFrame, factor: str) -> pd.DataFrame:
    return (
        run_rows.groupby(factor, dropna=False)
        .agg(
            run_scenarios=("seed", "count"),
            scenarios_with_any_below=("below_count", lambda values: int((values > 0).sum())),
            persistent_scenarios=("persistent_below", "sum"),
            worst_calibrated_accuracy=("calibrated_accuracy_min", "min"),
            mean_calibrated_accuracy=("calibrated_accuracy_mean", "mean"),
            worst_drop_trained_to_min=("drop_trained_to_min", "max"),
        )
        .reset_index()
        .sort_values(["persistent_scenarios", "worst_calibrated_accuracy"], ascending=[False, True])
    )


def build_lift_scenario_rows(
    source_rows: pd.DataFrame,
    config: PersistentFailureDiagnosisConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lift in config.lift_scenarios:
        adjusted = source_rows.copy()
        adjusted["calibrated_accuracy"] = adjusted["calibrated_accuracy"] + lift
        run_rows = build_run_diagnosis_rows(adjusted, config)
        rows.append(
            {
                "accuracy_lift": lift,
                "accuracy_lift_percent_points": lift * 100.0,
                "run_scenarios": int(len(run_rows)),
                "scenarios_with_any_below": int((run_rows["below_count"] > 0).sum()),
                "persistent_scenarios": int(run_rows["persistent_below"].sum()),
                "all_steps_below_scenarios": int(run_rows["all_steps_below"].sum()),
                "worst_calibrated_accuracy": float(run_rows["calibrated_accuracy_min"].min()),
                "minimum_margin_vs_floor": float(run_rows["calibrated_margin_vs_floor"].min()),
                "passes_strict_gate": int((run_rows["below_count"] == 0).sum())
                == len(run_rows),
            }
        )
    return pd.DataFrame(rows)


def run_persistent_failure_diagnosis(
    config: PersistentFailureDiagnosisConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = pd.read_csv(config.source_coding_stress_rows)
    run_rows = build_run_diagnosis_rows(source_rows, config)
    seed_summary = summarize_by_seed(run_rows)
    physical_summary = summarize_by_factor(run_rows, "physical_scenario")
    source_summary = summarize_by_factor(run_rows, "source_coding")
    policy_summary = summarize_by_factor(run_rows, "recalibration_policy")
    lift_summary = build_lift_scenario_rows(source_rows, config)

    artifacts = {
        "run_rows": config.output_dir / "persistent_failure_run_rows.csv",
        "seed_summary": config.output_dir / "persistent_failure_seed_summary.csv",
        "seed_summary_md": config.output_dir / "persistent_failure_seed_summary.md",
        "physical_summary": config.output_dir / "persistent_failure_physical_summary.csv",
        "source_summary": config.output_dir / "persistent_failure_source_summary.csv",
        "policy_summary": config.output_dir / "persistent_failure_policy_summary.csv",
        "lift_summary": config.output_dir / "persistent_failure_lift_summary.csv",
        "lift_summary_md": config.output_dir / "persistent_failure_lift_summary.md",
        "manifest": config.output_dir / "persistent_failure_diagnosis_manifest.json",
    }
    run_rows.to_csv(artifacts["run_rows"], index=False)
    seed_summary.to_csv(artifacts["seed_summary"], index=False)
    physical_summary.to_csv(artifacts["physical_summary"], index=False)
    source_summary.to_csv(artifacts["source_summary"], index=False)
    policy_summary.to_csv(artifacts["policy_summary"], index=False)
    lift_summary.to_csv(artifacts["lift_summary"], index=False)
    artifacts["seed_summary_md"].write_text(
        markdown_table(seed_summary),
        encoding="utf-8",
    )
    artifacts["lift_summary_md"].write_text(
        markdown_table(lift_summary),
        encoding="utf-8",
    )
    passing_lifts = lift_summary[lift_summary["passes_strict_gate"]]
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "run_scenarios": int(len(run_rows)),
        "scenarios_with_any_below": int((run_rows["below_count"] > 0).sum()),
        "persistent_scenarios": int(run_rows["persistent_below"].sum()),
        "all_steps_below_scenarios": int(run_rows["all_steps_below"].sum()),
        "dominant_failure_seed": int(seed_summary.iloc[0]["seed"]),
        "minimum_lift_to_clear_floor": (
            None
            if passing_lifts.empty
            else float(passing_lifts.iloc[0]["accuracy_lift"])
        ),
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }
    artifacts["manifest"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
