"""Report tables and plots for the optical simulation spike."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class ReportConfig:
    output_dir: Path
    report_dir: Path | None = None


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(item: dict[str, Any], *path: str) -> float | None:
    value: Any = item
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return float(value)


def add_row(
    rows: list[dict[str, object]],
    *,
    family: str,
    scenario: str,
    mode: str,
    accuracy: float | None,
    accuracy_drop: float | None,
    loss: float | None = None,
    diagnostics: dict[str, Any] | None = None,
    extra: str = "",
) -> None:
    projection = diagnostics.get("projection", {}) if diagnostics else {}
    prediction = diagnostics.get("prediction", {}) if diagnostics else {}
    rows.append(
        {
            "family": family,
            "scenario": scenario,
            "mode": mode,
            "accuracy": accuracy,
            "accuracy_drop": accuracy_drop,
            "loss": loss,
            "projection_mse": projection.get("projection_mse"),
            "cosine_similarity_mean": projection.get("cosine_similarity_mean"),
            "feature_mean_shift_l2": projection.get("feature_mean_shift_l2"),
            "feature_std_shift_mean_abs": projection.get("feature_std_shift_mean_abs"),
            "confusion_delta_l1": prediction.get("confusion_delta_l1"),
            "prediction_change_rate": prediction.get("prediction_change_rate"),
            "correct_to_wrong_rate": prediction.get("correct_to_wrong_rate"),
            "wrong_to_correct_rate": prediction.get("wrong_to_correct_rate"),
            "extra": extra,
        }
    )


def collect_rows(output_dir: Path) -> tuple[list[dict[str, object]], dict[str, Any | None]]:
    payloads = {
        "baseline": read_json(output_dir / "baseline" / "baseline_metrics.json"),
        "quantized": read_json(
            output_dir / "quantized_sweeps" / "quantized_projection_sweeps.json"
        ),
        "tunable": read_json(output_dir / "tunable_errors" / "tunable_error_sweeps.json"),
        "fixed": read_json(output_dir / "fixed_errors" / "fixed_error_sweeps.json"),
        "nonlinear": read_json(
            output_dir / "nonlinear_variants" / "nonlinear_variant_sweeps.json"
        ),
        "adaptation": read_json(output_dir / "adaptation_modes" / "adaptation_mode_sweeps.json"),
    }
    rows: list[dict[str, object]] = []

    baseline = payloads["baseline"]
    if baseline:
        add_row(
            rows,
            family="baseline",
            scenario="digital",
            mode="train",
            accuracy=metric_value(baseline, "final_train", "accuracy"),
            accuracy_drop=0.0,
            loss=metric_value(baseline, "final_train", "loss"),
        )
        add_row(
            rows,
            family="baseline",
            scenario="digital",
            mode="test",
            accuracy=metric_value(baseline, "final_test", "accuracy"),
            accuracy_drop=0.0,
            loss=metric_value(baseline, "final_test", "loss"),
        )

    quantized = payloads["quantized"]
    if quantized:
        for item in quantized["sweeps"]:
            accuracy_drop = metric_value(item, "absolute_accuracy_drop")
            if accuracy_drop is None:
                accuracy_drop = metric_value(item, "accuracy_drop")
            add_row(
                rows,
                family="quantized_projection",
                scenario=f'{item["bits"]}bit',
                mode="quantized",
                accuracy=metric_value(item, "metrics", "accuracy"),
                accuracy_drop=accuracy_drop,
                loss=metric_value(item, "metrics", "loss"),
                diagnostics=item.get("diagnostics"),
                extra=f'bits={item["bits"]}',
            )

    for family in ("tunable", "fixed"):
        payload = payloads[family]
        if payload:
            for item in payload["sweeps"]:
                name = item["scenario"]["name"]
                add_row(
                    rows,
                    family=f"{family}_optical_error",
                    scenario=name,
                    mode="raw",
                    accuracy=metric_value(item, "raw_metrics", "accuracy"),
                    accuracy_drop=metric_value(item, "raw_accuracy_drop"),
                    loss=metric_value(item, "raw_metrics", "loss"),
                    diagnostics=item.get("raw_diagnostics"),
                )
                add_row(
                    rows,
                    family=f"{family}_optical_error",
                    scenario=name,
                    mode="calibrated",
                    accuracy=metric_value(item, "calibrated_metrics", "accuracy"),
                    accuracy_drop=metric_value(item, "calibrated_accuracy_drop"),
                    loss=metric_value(item, "calibrated_metrics", "loss"),
                    diagnostics=item.get("calibrated_diagnostics"),
                )

    nonlinear = payloads["nonlinear"]
    if nonlinear:
        for item in nonlinear["sweeps"]:
            name = item["scenario"]["name"]
            add_row(
                rows,
                family="nonlinear_variant",
                scenario=name,
                mode="raw",
                accuracy=metric_value(item, "raw_metrics", "accuracy"),
                accuracy_drop=metric_value(item, "raw_accuracy_drop"),
                loss=metric_value(item, "raw_metrics", "loss"),
                diagnostics=item.get("raw_diagnostics"),
            )
            add_row(
                rows,
                family="nonlinear_variant",
                scenario=name,
                mode="compensated",
                accuracy=metric_value(item, "compensated_metrics", "accuracy"),
                accuracy_drop=metric_value(item, "compensated_accuracy_drop"),
                loss=metric_value(item, "compensated_metrics", "loss"),
                diagnostics=item.get("compensated_diagnostics"),
            )

    adaptation = payloads["adaptation"]
    if adaptation:
        scenario = adaptation["config"]["scenario"]["name"]
        for item in adaptation["modes"]:
            add_row(
                rows,
                family="combined_stress_adaptation",
                scenario=scenario,
                mode=item["mode"],
                accuracy=metric_value(item, "metrics", "accuracy"),
                accuracy_drop=metric_value(item, "accuracy_drop"),
                loss=metric_value(item, "metrics", "loss"),
                diagnostics=item.get("diagnostics"),
                extra=f'best_mode={adaptation["best_mode"]}',
            )

    return rows, payloads


def markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def save_bar_plot(
    df: pd.DataFrame,
    report_dir: Path,
    filename: str,
    title: str,
    *,
    label_column: str = "scenario",
    mode_filter: str | None = None,
) -> str | None:
    plot_df = df.copy()
    if mode_filter is not None:
        plot_df = plot_df[plot_df["mode"] == mode_filter]
    plot_df = plot_df.dropna(subset=["accuracy_drop"])
    if plot_df.empty:
        return None

    labels = (plot_df[label_column].astype(str) + " / " + plot_df["mode"].astype(str)).tolist()
    values = plot_df["accuracy_drop"].astype(float).tolist()
    fig_width = max(7.0, min(16.0, 0.48 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 4.5))
    ax.bar(range(len(labels)), values, color="#3b82f6")
    ax.set_title(title)
    ax.set_ylabel("Accuracy drop vs digital baseline")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    fig.tight_layout()
    path = report_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def save_quantized_curve(df: pd.DataFrame, report_dir: Path) -> str | None:
    plot_df = df[df["family"] == "quantized_projection"].copy()
    if plot_df.empty:
        return None
    plot_df["bits"] = plot_df["scenario"].str.replace("bit", "", regex=False).astype(int)
    plot_df = plot_df.sort_values("bits", ascending=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(plot_df["bits"], plot_df["accuracy"], marker="o", color="#16a34a")
    ax.set_title("Quantized Projection Accuracy")
    ax.set_xlabel("Quantization bits")
    ax.set_ylabel("Accuracy")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = report_dir / "quantized_accuracy_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return str(path)


def generate_report(config: ReportConfig) -> dict[str, object]:
    report_dir = config.report_dir or config.output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    rows, payloads = collect_rows(config.output_dir)
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No spike metrics found under {config.output_dir}")

    summary_csv = report_dir / "summary.csv"
    summary_md = report_dir / "summary.md"
    df.to_csv(summary_csv, index=False)
    summary_md.write_text(markdown_table(df), encoding="utf-8")

    plots: dict[str, str] = {}
    maybe_plot = save_quantized_curve(df, report_dir)
    if maybe_plot:
        plots["quantized_accuracy_curve"] = maybe_plot

    plot_specs = [
        ("baseline", "baseline_accuracy_drop.png", "Baseline Accuracy Drop", None),
        ("tunable_optical_error", "tunable_error_tolerance.png", "Tunable Error Tolerance", None),
        ("fixed_optical_error", "fixed_error_tolerance.png", "Fixed Error Tolerance", None),
        ("nonlinear_variant", "nonlinear_variant_comparison.png", "Nonlinear Variant Comparison", None),
        (
            "combined_stress_adaptation",
            "adaptation_recovery.png",
            "Combined-Stress Adaptation Recovery",
            None,
        ),
    ]
    for family, filename, title, mode_filter in plot_specs:
        maybe_plot = save_bar_plot(
            df[df["family"] == family],
            report_dir,
            filename,
            title,
            mode_filter=mode_filter,
        )
        if maybe_plot:
            plots[filename.removesuffix(".png")] = maybe_plot

    present_inputs = [name for name, payload in payloads.items() if payload is not None]
    missing_inputs = [name for name, payload in payloads.items() if payload is None]
    payload: dict[str, object] = {
        "input_dir": str(config.output_dir),
        "report_dir": str(report_dir),
        "present_inputs": present_inputs,
        "missing_inputs": missing_inputs,
        "rows": int(len(df)),
        "artifacts": {
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
            "plots": plots,
        },
    }

    metrics_path = report_dir / "report_manifest.json"
    payload["artifacts"]["manifest"] = str(metrics_path)  # type: ignore[index]
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
