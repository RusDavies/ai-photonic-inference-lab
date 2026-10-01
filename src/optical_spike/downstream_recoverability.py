"""Downstream recoverability checks for source-coding accuracy dips."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import pandas as pd

from optical_spike.reporting import markdown_table


RUN_KEYS = [
    "source_coding",
    "calibration_samples",
    "recalibration_policy",
    "physical_scenario",
    "seed",
]


@dataclass(frozen=True)
class DownstreamRecoverabilityConfig:
    source_coding_stress_rows: Path
    output_dir: Path
    accuracy_floor: float = 0.85
    leaky_budget: float = 0.02
    leaky_decay: float = 0.5
    injected_dip_depths: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02)


def _max_consecutive(values: list[bool]) -> int:
    current = 0
    best = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _below_runs(values: list[bool]) -> int:
    runs = 0
    in_run = False
    for value in values:
        if value and not in_run:
            runs += 1
            in_run = True
        elif not value:
            in_run = False
    return runs


def _leaky_failure(
    accuracies: list[float],
    *,
    floor: float,
    budget: float,
    decay: float,
) -> tuple[bool, float]:
    state = 0.0
    peak = 0.0
    for accuracy in accuracies:
        state = max(0.0, decay * state + max(0.0, floor - accuracy))
        peak = max(peak, state)
    return peak >= budget, peak


def _classify_sequence(
    accuracies: list[float],
    *,
    floor: float,
    leaky_budget: float,
    leaky_decay: float,
) -> dict[str, object]:
    below = [accuracy < floor for accuracy in accuracies]
    leaky_failed, leaky_peak = _leaky_failure(
        accuracies,
        floor=floor,
        budget=leaky_budget,
        decay=leaky_decay,
    )
    below_count = int(sum(below))
    max_consecutive = _max_consecutive(below)
    return {
        "time_steps": len(accuracies),
        "below_count": below_count,
        "below_runs": _below_runs(below),
        "max_consecutive_below": max_consecutive,
        "min_accuracy": min(accuracies),
        "mean_accuracy": sum(accuracies) / len(accuracies),
        "hard_any_below_failure": below_count > 0,
        "two_step_persistence_failure": max_consecutive >= 2,
        "majority_below_failure": below_count > (len(accuracies) / 2.0),
        "leaky_integrator_peak_deficit": leaky_peak,
        "leaky_integrator_failure": leaky_failed,
    }


def build_observed_recoverability_rows(
    source_rows: pd.DataFrame,
    config: DownstreamRecoverabilityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, group in source_rows.sort_values("time_step").groupby(RUN_KEYS):
        accuracies = group["calibrated_accuracy"].astype(float).tolist()
        rows.append(
            {
                **dict(zip(RUN_KEYS, key)),
                "sequence_kind": "observed",
                "injected_dip_depth": 0.0,
                **_classify_sequence(
                    accuracies,
                    floor=config.accuracy_floor,
                    leaky_budget=config.leaky_budget,
                    leaky_decay=config.leaky_decay,
                ),
            }
        )
    return pd.DataFrame(rows)


def build_injected_dip_rows(
    observed_rows: pd.DataFrame,
    source_rows: pd.DataFrame,
    config: DownstreamRecoverabilityConfig,
) -> pd.DataFrame:
    stable_keys = observed_rows[observed_rows["below_count"] == 0][RUN_KEYS]
    if stable_keys.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    key_tuples = {tuple(row) for row in stable_keys.to_numpy().tolist()}
    for key, group in source_rows.sort_values("time_step").groupby(RUN_KEYS):
        if tuple(key) not in key_tuples:
            continue
        base = group["calibrated_accuracy"].astype(float).tolist()
        injection_step = len(base) // 2
        for dip_depth in config.injected_dip_depths:
            injected = list(base)
            injected[injection_step] = min(
                injected[injection_step],
                config.accuracy_floor - dip_depth,
            )
            rows.append(
                {
                    **dict(zip(RUN_KEYS, key)),
                    "sequence_kind": "injected_single_step_dip",
                    "injected_dip_depth": dip_depth,
                    "injection_step": injection_step,
                    **_classify_sequence(
                        injected,
                        floor=config.accuracy_floor,
                        leaky_budget=config.leaky_budget,
                        leaky_decay=config.leaky_decay,
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_recoverability(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby(["sequence_kind", "injected_dip_depth"], dropna=False)
    return (
        grouped.agg(
            runs=("seed", "count"),
            runs_with_below=("below_count", lambda values: int((values > 0).sum())),
            hard_any_below_failures=("hard_any_below_failure", "sum"),
            two_step_persistence_failures=("two_step_persistence_failure", "sum"),
            majority_below_failures=("majority_below_failure", "sum"),
            leaky_integrator_failures=("leaky_integrator_failure", "sum"),
            worst_min_accuracy=("min_accuracy", "min"),
            median_min_accuracy=("min_accuracy", "median"),
            max_consecutive_below=("max_consecutive_below", "max"),
            max_leaky_integrator_peak=(
                "leaky_integrator_peak_deficit",
                "max",
            ),
        )
        .reset_index()
        .sort_values(["sequence_kind", "injected_dip_depth"])
    )


def run_downstream_recoverability_report(
    config: DownstreamRecoverabilityConfig,
) -> dict[str, object]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = pd.read_csv(config.source_coding_stress_rows)
    observed = build_observed_recoverability_rows(source_rows, config)
    injected = build_injected_dip_rows(observed, source_rows, config)
    rows = (
        observed
        if injected.empty
        else pd.concat([observed, injected], ignore_index=True)
    )
    summary = summarize_recoverability(rows)
    rows_csv = config.output_dir / "downstream_recoverability_rows.csv"
    summary_csv = config.output_dir / "downstream_recoverability_summary.csv"
    summary_md = config.output_dir / "downstream_recoverability_summary.md"
    manifest_path = config.output_dir / "downstream_recoverability_manifest.json"
    rows.to_csv(rows_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(summary), encoding="utf-8")
    observed_failures = observed[
        observed["two_step_persistence_failure"]
        | observed["majority_below_failure"]
        | observed["leaky_integrator_failure"]
    ]
    payload: dict[str, object] = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "observed_runs": int(len(observed)),
        "injected_runs": int(len(injected)),
        "summary_rows": int(len(summary)),
        "observed_runs_with_any_below": int((observed["below_count"] > 0).sum()),
        "observed_runs_with_persistent_failure": int(len(observed_failures)),
        "artifacts": {
            "rows_csv": str(rows_csv),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
