"""Feature and prediction diagnostics for projection transfer experiments."""

from __future__ import annotations

import numpy as np


def confusion_matrix(labels: np.ndarray, predictions: np.ndarray, classes: int = 10) -> np.ndarray:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(matrix, (labels.astype(int), predictions.astype(int)), 1)
    return matrix


def confusion_delta(
    baseline_confusion: np.ndarray,
    observed_confusion: np.ndarray,
) -> dict[str, object]:
    delta = observed_confusion - baseline_confusion
    samples = max(int(np.sum(baseline_confusion)), 1)
    return {
        "confusion_delta_l1": float(np.sum(np.abs(delta)) / samples),
        "confusion_delta_max_cell": int(np.max(np.abs(delta))),
        "confusion_delta_matrix": delta.astype(int).tolist(),
    }


def prediction_delta(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    observed_predictions: np.ndarray,
    baseline_confusion: np.ndarray,
    observed_confusion: np.ndarray,
) -> dict[str, object]:
    baseline_correct = baseline_predictions == labels
    observed_correct = observed_predictions == labels
    return {
        **confusion_delta(baseline_confusion, observed_confusion),
        "prediction_change_rate": float(np.mean(baseline_predictions != observed_predictions)),
        "correct_to_wrong_rate": float(np.mean(baseline_correct & ~observed_correct)),
        "wrong_to_correct_rate": float(np.mean(~baseline_correct & observed_correct)),
    }


def projection_diagnostics(
    ideal_projection: np.ndarray,
    observed_projection: np.ndarray,
) -> dict[str, float]:
    error = observed_projection - ideal_projection
    ideal_norm = np.linalg.norm(ideal_projection, axis=1)
    observed_norm = np.linalg.norm(observed_projection, axis=1)
    denominator = np.maximum(ideal_norm * observed_norm, 1e-8)
    cosine = np.sum(ideal_projection * observed_projection, axis=1) / denominator

    ideal_mean = np.mean(ideal_projection, axis=0)
    observed_mean = np.mean(observed_projection, axis=0)
    ideal_std = np.std(ideal_projection, axis=0)
    observed_std = np.std(observed_projection, axis=0)
    mean_shift = observed_mean - ideal_mean
    std_shift = observed_std - ideal_std
    std_ratio = observed_std / np.maximum(ideal_std, 1e-8)

    return {
        "projection_mse": float(np.mean(error * error)),
        "projection_rmse": float(np.sqrt(np.mean(error * error))),
        "projection_mae": float(np.mean(np.abs(error))),
        "projection_max_abs_error": float(np.max(np.abs(error))),
        "cosine_similarity_mean": float(np.mean(cosine)),
        "cosine_similarity_std": float(np.std(cosine)),
        "cosine_similarity_min": float(np.min(cosine)),
        "feature_mean_shift_l2": float(np.linalg.norm(mean_shift)),
        "feature_mean_shift_mean_abs": float(np.mean(np.abs(mean_shift))),
        "feature_mean_shift_max_abs": float(np.max(np.abs(mean_shift))),
        "feature_std_shift_mean_abs": float(np.mean(np.abs(std_shift))),
        "feature_std_shift_max_abs": float(np.max(np.abs(std_shift))),
        "feature_std_ratio_mean": float(np.mean(std_ratio)),
        "feature_std_ratio_std": float(np.std(std_ratio)),
    }
