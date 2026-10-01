from __future__ import annotations

import json
import pickle

import numpy as np

from optical_spike.cifar_feature_target import extract_cifar_spatial_features
from optical_spike.data import load_cifar10
from optical_spike.cli import main, planned_stages


def test_planned_stages_include_backlog_items() -> None:
    stages = planned_stages()

    assert "digital_baseline" in stages
    assert "tunable_optical_error_models" in stages
    assert "fixed_fabricated_error_models" in stages
    assert "nonlinear_optical_variants" in stages
    assert "calibration_and_adaptation_modes" in stages
    assert "multi_seed_full_split" in stages
    assert "transformer_mlp_block_target" in stages
    assert "cifar10_spatial_feature_projection_target" in stages
    assert "cifar10_spatial_feature_source_coding_stress" in stages
    assert "constrained_fixed_target_comparison" in stages
    assert "calibration_cost_model" in stages
    assert "material_nonlinearity_candidates" in stages
    assert "energy_latency_budget" in stages
    assert "material_activation_training" in stages
    assert "material_strength_constrained_transfer" in stages
    assert "chained_optical_block_budget" in stages
    assert "charge_bucket_readout_budget" in stages
    assert "functional_chained_block_simulation" in stages
    assert "chained_material_training" in stages
    assert "unshared_chained_material_training" in stages
    assert "calibration_bridge_physical_units" in stages
    assert "redundant_source_coding_validation" in stages
    assert "source_coding_calibration_stress" in stages
    assert "source_coding_ablation_report" in stages
    assert "downstream_recoverability_report" in stages
    assert "persistent_failure_diagnosis" in stages
    assert "architecture_energy_overlay_report" in stages
    assert "architecture_energy_stress_report" in stages
    assert "architecture_converter_stress_report" in stages
    assert "architecture_converter_target_report" in stages
    assert "architecture_converter_profile_report" in stages
    assert "architecture_dac_coding_sweep_report" in stages


def test_dry_run_outputs_json(capsys) -> None:
    exit_code = main(["--dry-run", "--output-dir", "artifacts/test"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["output_dir"] == "artifacts/test"


def test_synthetic_baseline_saves_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-baseline",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "64",
            "--max-test-samples",
            "32",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "baseline" / "baseline_metrics.json"
    weights_path = tmp_path / "baseline" / "baseline_weights.npz"
    assert payload["dataset"] == "synthetic"
    assert metrics_path.exists()
    assert weights_path.exists()
    assert 0.0 <= payload["final_test"]["accuracy"] <= 1.0


def test_source_coding_ablation_report_reduces_existing_rows(tmp_path, capsys) -> None:
    rows_path = tmp_path / "source_rows.csv"
    rows_path.write_text(
        "\n".join(
            [
                "seed,scenario,chain_length,physical_scenario,source_coding,"
                "source_bits,source_redundant_phases,calibration_samples,"
                "recalibration_policy,calibrated_accuracy,drop_vs_trained,"
                "trained_chain_accuracy,thermal_drift_rate,thermal_excursion_k_per_step,"
                "detector_noise,detector_photons_per_sample,output_readout_noise,"
                "output_enob_equivalent,projection_mse_mean,projection_mse_max,"
                "source_quantization_mse_mean,recalibration_count,"
                "triggered_recalibration_count",
                "7,reverse_saturable,4,nominal,redundant_6bit_2phase,6,2,32,"
                "scheduled_only,0.861,0.012,0.873,0.002,0.7,0.02,2500,0.005,"
                "7.35,1.2,2.1,0.14,1,0",
                "11,reverse_saturable,4,thermal_edge,redundant_7bit_2phase,7,2,256,"
                "scheduled_plus_projection_mse_trigger,0.852,0.021,0.873,0.005,1.7,"
                "0.03,1111,0.005,7.35,1.4,2.5,0.04,2,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-source-coding-ablation-report",
            "--source-coding-ablation-rows-path",
            str(rows_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    report_path = (
        tmp_path / "out" / "source_coding_ablation" / "source_coding_ablation_report.md"
    )
    assert payload["rows"] == 2
    assert payload["steps_below_floor"] == 0
    assert report_path.exists()


def test_cifar10_loader_reads_python_batches(tmp_path) -> None:
    cifar_dir = tmp_path / "cifar-10-batches-py"
    cifar_dir.mkdir()
    train_data = np.arange(2 * 3072, dtype=np.uint8).reshape(2, 3072)
    test_data = np.arange(3072, dtype=np.uint8).reshape(1, 3072)
    with (cifar_dir / "data_batch_1").open("wb") as handle:
        pickle.dump({"data": train_data, "labels": [1, 2]}, handle)
    for index in range(2, 6):
        with (cifar_dir / f"data_batch_{index}").open("wb") as handle:
            pickle.dump({"data": train_data[:0], "labels": []}, handle)
    with (cifar_dir / "test_batch").open("wb") as handle:
        pickle.dump({"data": test_data, "labels": [3]}, handle)

    train_images, train_labels, test_images, test_labels = load_cifar10(
        tmp_path,
        download=False,
    )

    assert train_images.shape == (2, 3072)
    assert test_images.shape == (1, 3072)
    assert train_images.dtype == np.float32
    assert train_images.max() <= 1.0
    assert train_labels.tolist() == [1, 2]
    assert test_labels.tolist() == [3]


def write_cifar_fixture(root, train_count: int = 8, test_count: int = 4) -> None:
    cifar_dir = root / "cifar-10-batches-py"
    cifar_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    train_data = rng.integers(0, 256, size=(train_count, 3072), dtype=np.uint8)
    test_data = rng.integers(0, 256, size=(test_count, 3072), dtype=np.uint8)
    with (cifar_dir / "data_batch_1").open("wb") as handle:
        pickle.dump({"data": train_data, "labels": list(range(train_count))}, handle)
    for index in range(2, 6):
        with (cifar_dir / f"data_batch_{index}").open("wb") as handle:
            pickle.dump({"data": train_data[:0], "labels": []}, handle)
    with (cifar_dir / "test_batch").open("wb") as handle:
        pickle.dump({"data": test_data, "labels": list(range(test_count))}, handle)


def test_cifar_spatial_features_are_standardized() -> None:
    rng = np.random.default_rng(11)
    images = rng.random((12, 3072), dtype=np.float32)

    features, mean, std = extract_cifar_spatial_features(images)
    transformed, _, _ = extract_cifar_spatial_features(images, mean=mean, std=std)

    assert features.shape == (12, 460)
    assert np.all(std > 0)
    np.testing.assert_allclose(features, transformed)
    np.testing.assert_allclose(features.mean(axis=0), 0.0, atol=1e-5)


def test_cifar_feature_target_saves_metrics(tmp_path, capsys) -> None:
    data_dir = tmp_path / "cifar"
    write_cifar_fixture(data_dir)

    exit_code = main(
        [
            "--run-cifar-feature-target",
            "--dataset",
            "cifar10",
            "--data-dir",
            str(data_dir),
            "--epochs",
            "1",
            "--hidden-dim",
            "16",
            "--quantization-bits",
            "8,4",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "out" / "cifar_feature_target" / "cifar_feature_target.json"
    weights_path = (
        tmp_path / "out" / "cifar_feature_target" / "cifar_feature_target_weights.npz"
    )
    assert payload["dataset"] == "cifar10"
    assert payload["feature_dim"] == 460
    assert payload["target"]["matrix_shape"] == [460, 16]
    assert [item["bits"] for item in payload["quantized_sweeps"]] == [8, 4]
    assert metrics_path.exists()
    assert weights_path.exists()


def test_cifar_feature_source_coding_stress_saves_metrics(tmp_path, capsys) -> None:
    data_dir = tmp_path / "cifar"
    write_cifar_fixture(data_dir)

    exit_code = main(
        [
            "--run-cifar-feature-source-coding-stress",
            "--dataset",
            "cifar10",
            "--data-dir",
            str(data_dir),
            "--epochs",
            "1",
            "--hidden-dim",
            "16",
            "--calibration-sample-counts",
            "4",
            "--constrained-fixed-scenario",
            "signed_tiled_mild",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    artifact_dir = tmp_path / "out" / "cifar_feature_source_coding_stress"
    assert payload["rows"] == 192
    assert payload["summary_rows"] == 24
    assert payload["config"]["constrained_scenario"]["name"] == "signed_tiled_mild"
    assert 0.0 <= payload["calibrated_accuracy_min"] <= payload["calibrated_accuracy_max"] <= 1.0
    assert (artifact_dir / "cifar_feature_source_coding_stress_rows.csv").exists()
    assert (artifact_dir / "cifar_feature_source_coding_stress_summary.csv").exists()
    assert (artifact_dir / "cifar_feature_source_coding_stress_summary.md").exists()
    assert (artifact_dir / "cifar_feature_source_coding_stress_manifest.json").exists()


def test_synthetic_quantized_sweeps_save_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-quantized-sweeps",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "64",
            "--max-test-samples",
            "32",
            "--quantization-bits",
            "8,4",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "quantized_sweeps" / "quantized_projection_sweeps.json"
    assert metrics_path.exists()
    assert [item["bits"] for item in payload["sweeps"]] == [8, 4]
    for item in payload["sweeps"]:
        assert 0.0 <= item["metrics"]["accuracy"] <= 1.0
        assert item["diagnostics"]["projection"]["projection_mse"] >= 0.0
        assert -1.0 <= item["diagnostics"]["projection"]["cosine_similarity_mean"] <= 1.0
        assert item["diagnostics"]["prediction"]["confusion_delta_l1"] >= 0.0


def test_synthetic_tunable_error_sweeps_save_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-tunable-errors",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "80",
            "--max-test-samples",
            "32",
            "--calibration-samples",
            "16",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "tunable_errors" / "tunable_error_sweeps.json"
    assert metrics_path.exists()
    assert {item["scenario"]["name"] for item in payload["sweeps"]} == {
        "quantized_8bit",
        "mild",
        "moderate",
        "severe",
    }
    for item in payload["sweeps"]:
        assert 0.0 <= item["raw_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= item["calibrated_metrics"]["accuracy"] <= 1.0


def test_synthetic_fixed_error_sweeps_save_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-fixed-errors",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "80",
            "--max-test-samples",
            "32",
            "--calibration-samples",
            "16",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "fixed_errors" / "fixed_error_sweeps.json"
    assert metrics_path.exists()
    assert {item["scenario"]["name"] for item in payload["sweeps"]} == {
        "near_ideal_fixed",
        "mild_fixed",
        "moderate_fixed",
        "severe_fixed",
    }
    for item in payload["sweeps"]:
        assert 0.0 <= item["raw_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= item["calibrated_metrics"]["accuracy"] <= 1.0


def test_synthetic_nonlinear_variant_sweeps_save_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-nonlinear-variants",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "80",
            "--max-test-samples",
            "32",
            "--calibration-samples",
            "16",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "nonlinear_variants" / "nonlinear_variant_sweeps.json"
    assert metrics_path.exists()
    assert {item["scenario"]["kind"] for item in payload["sweeps"]} == {
        "saturation",
        "clipping",
        "intensity_distortion",
    }
    for item in payload["sweeps"]:
        assert 0.0 <= item["raw_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= item["compensated_metrics"]["accuracy"] <= 1.0


def test_synthetic_adaptation_modes_save_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-adaptation-modes",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--calibration-samples",
            "16",
            "--adaptation-epochs",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "adaptation_modes" / "adaptation_mode_sweeps.json"
    assert metrics_path.exists()
    assert {item["mode"] for item in payload["modes"]} == {
        "no_adaptation",
        "calibration_only",
        "retrained_downstream_head",
        "hardware_aware_fine_tuning",
        "distillation",
    }
    assert payload["best_mode"] in {item["mode"] for item in payload["modes"]}
    for item in payload["modes"]:
        assert 0.0 <= item["metrics"]["accuracy"] <= 1.0


def write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_generate_report_from_existing_metrics(tmp_path, capsys) -> None:
    write_json(
        tmp_path / "baseline" / "baseline_metrics.json",
        {
            "final_train": {"accuracy": 0.9, "loss": 0.4},
            "final_test": {"accuracy": 0.8, "loss": 0.5},
        },
    )
    write_json(
        tmp_path / "quantized_sweeps" / "quantized_projection_sweeps.json",
        {
            "sweeps": [
                {
                    "bits": 8,
                    "metrics": {"accuracy": 0.79, "loss": 0.52},
                    "accuracy_drop": 0.01,
                    "diagnostics": {
                        "projection": {
                            "projection_mse": 0.001,
                            "cosine_similarity_mean": 0.99,
                            "feature_mean_shift_l2": 0.1,
                            "feature_std_shift_mean_abs": 0.02,
                        },
                        "prediction": {
                            "confusion_delta_l1": 0.03,
                            "prediction_change_rate": 0.02,
                            "correct_to_wrong_rate": 0.01,
                            "wrong_to_correct_rate": 0.0,
                        },
                    },
                }
            ]
        },
    )
    write_json(
        tmp_path / "tunable_errors" / "tunable_error_sweeps.json",
        {
            "sweeps": [
                {
                    "scenario": {"name": "mild"},
                    "raw_metrics": {"accuracy": 0.75, "loss": 0.6},
                    "calibrated_metrics": {"accuracy": 0.78, "loss": 0.55},
                    "raw_accuracy_drop": 0.05,
                    "calibrated_accuracy_drop": 0.02,
                }
            ]
        },
    )
    write_json(
        tmp_path / "fixed_errors" / "fixed_error_sweeps.json",
        {
            "sweeps": [
                {
                    "scenario": {"name": "moderate_fixed"},
                    "raw_metrics": {"accuracy": 0.7, "loss": 0.7},
                    "calibrated_metrics": {"accuracy": 0.76, "loss": 0.6},
                    "raw_accuracy_drop": 0.1,
                    "calibrated_accuracy_drop": 0.04,
                }
            ]
        },
    )
    write_json(
        tmp_path / "nonlinear_variants" / "nonlinear_variant_sweeps.json",
        {
            "sweeps": [
                {
                    "scenario": {"name": "saturation_mild"},
                    "raw_metrics": {"accuracy": 0.73, "loss": 0.65},
                    "compensated_metrics": {"accuracy": 0.77, "loss": 0.57},
                    "raw_accuracy_drop": 0.07,
                    "compensated_accuracy_drop": 0.03,
                }
            ]
        },
    )
    write_json(
        tmp_path / "adaptation_modes" / "adaptation_mode_sweeps.json",
        {
            "config": {"scenario": {"name": "moderate_fixed_adaptation"}},
            "best_mode": "hardware_aware_fine_tuning",
            "modes": [
                {
                    "mode": "no_adaptation",
                    "metrics": {"accuracy": 0.7, "loss": 0.7},
                    "accuracy_drop": 0.1,
                },
                {
                    "mode": "hardware_aware_fine_tuning",
                    "metrics": {"accuracy": 0.79, "loss": 0.52},
                    "accuracy_drop": 0.01,
                },
            ],
        },
    )

    exit_code = main(["--generate-report", "--output-dir", str(tmp_path)])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    report_dir = tmp_path / "reports"
    assert payload["missing_inputs"] == []
    assert (report_dir / "summary.csv").exists()
    assert (report_dir / "summary.md").exists()
    assert (report_dir / "report_manifest.json").exists()
    assert (report_dir / "quantized_accuracy_curve.png").exists()
    assert (report_dir / "adaptation_recovery.png").exists()
    summary_text = (report_dir / "summary.csv").read_text(encoding="utf-8")
    assert "projection_mse" in summary_text
    assert "confusion_delta_l1" in summary_text
    assert "0.001" in summary_text


def test_synthetic_multi_seed_runner_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-multi-seed",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "80",
            "--max-test-samples",
            "32",
            "--calibration-samples",
            "16",
            "--adaptation-epochs",
            "2",
            "--quantization-bits",
            "8,4",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "multi_seed"
    assert payload["config"]["seeds"] == [7, 11]
    assert payload["config"]["full_split"] is False
    assert payload["rows"] > 0
    assert (output_dir / "multi_seed_rows.csv").exists()
    assert (output_dir / "multi_seed_summary.csv").exists()
    assert (output_dir / "multi_seed_summary.md").exists()
    assert (output_dir / "multi_seed_manifest.json").exists()
    aggregate_header = (output_dir / "multi_seed_summary.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "projection_mse_mean" in aggregate_header
    assert "prediction_change_rate_mean" in aggregate_header


def test_synthetic_transformer_mlp_target_saves_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-transformer-mlp-target",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--quantization-bits",
            "8,4",
            "--calibration-samples",
            "16",
            "--adaptation-epochs",
            "2",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    metrics_path = tmp_path / "transformer_mlp_target" / "transformer_mlp_target.json"
    assert metrics_path.exists()
    assert payload["target"]["name"] == "mlp_up_projection"
    assert payload["target"]["matrix_shape"] == [16, 32]
    assert payload["config"]["calibration_samples"] == 16
    assert payload["adaptation"]["adaptation_epochs"] == 2
    assert [item["bits"] for item in payload["sweeps"]] == [8, 4]
    assert {item["scenario"]["name"] for item in payload["tunable_sweeps"]} == {
        "quantized_8bit",
        "mild",
        "moderate",
        "severe",
    }
    assert {item["scenario"]["name"] for item in payload["fixed_sweeps"]} == {
        "near_ideal_fixed",
        "mild_fixed",
        "moderate_fixed",
        "severe_fixed",
    }
    assert {item["scenario"]["name"] for item in payload["constrained_fixed_sweeps"]} == {
        "signed_tiled_mild",
        "signed_tiled_moderate",
        "signed_tiled_severe",
    }
    assert {item["mode"] for item in payload["adaptation"]["modes"]} == {
        "no_adaptation",
        "calibration_only",
        "hardware_aware_tail_fine_tuning",
    }
    for item in payload["sweeps"]:
        assert 0.0 <= item["metrics"]["accuracy"] <= 1.0
        assert item["diagnostics"]["projection"]["projection_mse"] >= 0.0
        assert item["diagnostics"]["prediction"]["prediction_change_rate"] >= 0.0
    for item in payload["fixed_sweeps"]:
        assert 0.0 <= item["raw_metrics"]["accuracy"] <= 1.0
        assert 0.0 <= item["calibrated_metrics"]["accuracy"] <= 1.0
    for item in payload["constrained_fixed_sweeps"]:
        assert item["error_model"]["routing"]["fan_in_tiles"] >= 1
        assert item["error_model"]["loss"]["effective"] >= 0.0


def test_synthetic_constrained_fixed_comparison_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-constrained-fixed-comparison",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--calibration-samples",
            "16",
            "--adaptation-epochs",
            "2",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "constrained_comparison"
    assert payload["config"]["seeds"] == [7, 11]
    assert payload["config"]["full_split"] is False
    assert payload["rows"] == 28
    assert (output_dir / "constrained_comparison_rows.csv").exists()
    assert (output_dir / "constrained_comparison_summary.csv").exists()
    assert (output_dir / "constrained_comparison_summary.md").exists()
    assert (output_dir / "constrained_comparison_manifest.json").exists()
    rows_text = (output_dir / "constrained_comparison_rows.csv").read_text(encoding="utf-8")
    assert "input_projection" in rows_text
    assert "mlp_up_projection" in rows_text
    assert "signed_tiled_severe" in rows_text


def test_synthetic_calibration_cost_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-calibration-cost",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--calibration-sample-counts",
            "8,16",
            "--recalibration-intervals",
            "1,4",
            "--drift-rates",
            "0.0,0.01",
            "--time-steps",
            "4",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "calibration_cost"
    assert payload["config"]["calibration_sample_counts"] == [8, 16]
    assert payload["config"]["recalibration_intervals"] == [1, 4]
    assert payload["config"]["drift_rates"] == [0.0, 0.01]
    assert payload["rows"] == 16
    assert (output_dir / "calibration_cost_rows.csv").exists()
    assert (output_dir / "calibration_cost_summary.csv").exists()
    assert (output_dir / "calibration_cost_summary.md").exists()
    assert (output_dir / "calibration_cost_manifest.json").exists()
    summary_text = (output_dir / "calibration_cost_summary.csv").read_text(encoding="utf-8")
    assert "calibration_examples_per_step_mean" in summary_text
    assert "digital_correction_ops_per_inference_mean" in summary_text


def test_synthetic_material_nonlinearities_save_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-material-nonlinearities",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--calibration-samples",
            "16",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "material_nonlinearity"
    assert payload["config"]["seeds"] == [7, 11]
    assert payload["rows"] == 40
    assert (output_dir / "material_nonlinearity_rows.csv").exists()
    assert (output_dir / "material_nonlinearity_summary.csv").exists()
    assert (output_dir / "material_nonlinearity_summary.md").exists()
    assert (output_dir / "material_nonlinearity_manifest.json").exists()
    rows_text = (output_dir / "material_nonlinearity_rows.csv").read_text(encoding="utf-8")
    assert "distortion_calibrated" in rows_text
    assert "activation_calibrated" in rows_text
    assert "saturable_absorber" in rows_text


def test_energy_budget_from_calibration_summary(tmp_path, capsys) -> None:
    summary_path = tmp_path / "calibration_cost_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "scenario,calibration_samples,recalibration_interval_steps,drift_rate,"
                "baseline_accuracy_mean,raw_accuracy_mean,calibrated_accuracy_mean,"
                "calibrated_accuracy_min_mean,calibrated_accuracy_drop_mean,"
                "calibrated_accuracy_drop_max_mean,total_calibration_examples_mean,"
                "calibration_examples_per_step_mean,affine_values_per_step_mean,"
                "digital_correction_ops_per_inference_mean,runs",
                "signed_tiled_severe,32,16,0.005,0.8686,0.8659,0.8675,0.8651,"
                "0.0011,0.0035,32,2,32,512,3",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-energy-budget",
            "--calibration-summary-path",
            str(summary_path),
            "--inferences-per-drift-step",
            "1000",
            "--hidden-dim",
            "64",
            "--mlp-dim",
            "256",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "energy_budget"
    assert payload["rows"] == 5
    assert payload["summary_rows"] == 5
    assert (output_dir / "energy_budget_rows.csv").exists()
    assert (output_dir / "energy_budget_summary.csv").exists()
    assert (output_dir / "energy_budget_summary.md").exists()
    rows_text = (output_dir / "energy_budget_rows.csv").read_text(encoding="utf-8")
    assert "published_273fj_low_io" in rows_text
    assert "energy_ratio_vs_digital" in rows_text
    assert "laser_energy_pj" in rows_text


def test_chained_block_budget_from_calibration_summary(tmp_path, capsys) -> None:
    summary_path = tmp_path / "calibration_cost_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "scenario,calibration_samples,recalibration_interval_steps,drift_rate,"
                "baseline_accuracy_mean,raw_accuracy_mean,calibrated_accuracy_mean,"
                "calibrated_accuracy_min_mean,calibrated_accuracy_drop_mean,"
                "calibrated_accuracy_drop_max_mean,total_calibration_examples_mean,"
                "calibration_examples_per_step_mean,affine_values_per_step_mean,"
                "digital_correction_ops_per_inference_mean,runs",
                "signed_tiled_severe,32,16,0.005,0.8686,0.8659,0.8675,0.8651,"
                "0.0011,0.0035,32,2,32,512,3",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-chained-block-budget",
            "--calibration-summary-path",
            str(summary_path),
            "--inferences-per-drift-step",
            "1000",
            "--hidden-dim",
            "64",
            "--mlp-dim",
            "256",
            "--chain-lengths",
            "1,4",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "chained_block_budget"
    assert payload["rows"] == 10
    assert payload["summary_rows"] == 10
    assert (output_dir / "chained_block_budget_rows.csv").exists()
    assert (output_dir / "chained_block_budget_summary.csv").exists()
    assert (output_dir / "chained_block_budget_summary.md").exists()
    rows_text = (output_dir / "chained_block_budget_rows.csv").read_text(encoding="utf-8")
    assert "energy_ratio_vs_digital_chain" in rows_text
    assert "converter_latency_saved_ns" in rows_text


def test_charge_readout_budget_from_calibration_summary(tmp_path, capsys) -> None:
    summary_path = tmp_path / "calibration_cost_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "scenario,calibration_samples,recalibration_interval_steps,drift_rate,"
                "baseline_accuracy_mean,raw_accuracy_mean,calibrated_accuracy_mean,"
                "calibrated_accuracy_min_mean,calibrated_accuracy_drop_mean,"
                "calibrated_accuracy_drop_max_mean,total_calibration_examples_mean,"
                "calibration_examples_per_step_mean,affine_values_per_step_mean,"
                "digital_correction_ops_per_inference_mean,runs",
                "signed_tiled_severe,32,16,0.005,0.8686,0.8659,0.8675,0.8651,"
                "0.0011,0.0035,32,2,32,512,3",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-charge-readout-budget",
            "--calibration-summary-path",
            str(summary_path),
            "--inferences-per-drift-step",
            "1000",
            "--hidden-dim",
            "64",
            "--mlp-dim",
            "256",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "charge_readout_budget"
    assert payload["rows"] == 20
    assert payload["summary_rows"] == 20
    assert (output_dir / "charge_readout_budget_rows.csv").exists()
    assert (output_dir / "charge_readout_budget_summary.csv").exists()
    assert (output_dir / "charge_readout_budget_summary.md").exists()
    rows_text = (output_dir / "charge_readout_budget_rows.csv").read_text(encoding="utf-8")
    assert "ccd_8_lane_low_power_bucket" in rows_text
    assert "charge_transfer_count" in rows_text


def test_synthetic_material_training_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-material-training",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "material_training"
    assert payload["config"]["seeds"] == [7, 11]
    assert payload["rows"] == 8
    assert (output_dir / "material_training_rows.csv").exists()
    assert (output_dir / "material_training_summary.csv").exists()
    assert (output_dir / "material_training_summary.md").exists()
    assert (output_dir / "material_training_manifest.json").exists()
    rows_text = (output_dir / "material_training_rows.csv").read_text(encoding="utf-8")
    assert "accuracy_delta_vs_relu" in rows_text
    assert "saturable_absorber_trained" in rows_text


def test_synthetic_material_strength_sweep_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-material-strength-sweep",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--material-strengths",
            "0.1,0.2",
            "--calibration-samples",
            "16",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "material_strength_sweep"
    assert payload["config"]["strengths"] == [0.1, 0.2]
    assert payload["rows"] == 8
    assert (output_dir / "material_strength_sweep_rows.csv").exists()
    assert (output_dir / "material_strength_sweep_summary.csv").exists()
    assert (output_dir / "material_strength_sweep_summary.md").exists()
    assert (output_dir / "material_strength_sweep_manifest.json").exists()
    rows_text = (output_dir / "material_strength_sweep_rows.csv").read_text(encoding="utf-8")
    assert "constrained_calibrated_accuracy" in rows_text
    assert "reverse_saturable_strength_0p2" in rows_text


def test_synthetic_functional_chain_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-functional-chain",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--chain-lengths",
            "1,2",
            "--calibration-samples",
            "16",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "functional_chain"
    assert payload["config"]["chain_lengths"] == [1, 2]
    assert payload["rows"] == 8
    assert (output_dir / "functional_chain_rows.csv").exists()
    assert (output_dir / "functional_chain_summary.csv").exists()
    assert (output_dir / "functional_chain_summary.md").exists()
    assert (output_dir / "functional_chain_manifest.json").exists()
    rows_text = (output_dir / "functional_chain_rows.csv").read_text(encoding="utf-8")
    assert "calibrated_constrained_accuracy" in rows_text
    assert "ideal_chain_delta_vs_one_block" in rows_text


def test_synthetic_chained_material_training_saves_aggregate(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-chained-material-training",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--chain-lengths",
            "2",
            "--calibration-samples",
            "16",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "chained_material_training"
    assert payload["config"]["chain_lengths"] == [2]
    assert payload["rows"] == 4
    assert (output_dir / "chained_material_training_rows.csv").exists()
    assert (output_dir / "chained_material_training_summary.csv").exists()
    assert (output_dir / "chained_material_training_summary.md").exists()
    assert (output_dir / "chained_material_training_manifest.json").exists()
    rows_text = (output_dir / "chained_material_training_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "trained_chain_accuracy" in rows_text
    assert "calibrated_drop_vs_trained" in rows_text


def test_synthetic_unshared_chained_material_training_saves_aggregate(
    tmp_path,
    capsys,
) -> None:
    exit_code = main(
        [
            "--run-unshared-chained-material-training",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--chain-lengths",
            "2",
            "--calibration-samples",
            "16",
            "--seeds",
            "7,11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "unshared_chained_material_training"
    assert payload["config"]["chain_lengths"] == [2]
    assert payload["rows"] == 2
    assert (output_dir / "unshared_chained_material_training_rows.csv").exists()
    assert (output_dir / "unshared_chained_material_training_summary.csv").exists()
    assert (output_dir / "unshared_chained_material_training_summary.md").exists()
    assert (output_dir / "unshared_chained_material_training_manifest.json").exists()
    rows_text = (
        output_dir / "unshared_chained_material_training_rows.csv"
    ).read_text(encoding="utf-8")
    assert "trained_chain_accuracy" in rows_text
    assert "calibrated_drop_vs_trained" in rows_text


def test_synthetic_redundant_source_validation_saves_metrics(tmp_path, capsys) -> None:
    exit_code = main(
        [
            "--run-redundant-source-validation",
            "--synthetic",
            "--seeds",
            "7",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--chain-lengths",
            "4",
            "--calibration-samples",
            "16",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "redundant_source_validation"
    assert payload["rows"] == 32
    assert payload["summary_rows"] == 4
    assert payload["config"]["source_coding"]["name"] == "redundant_6bit_2phase"
    assert (output_dir / "unshared_physical_drift_noise_rows.csv").exists()
    assert (output_dir / "unshared_physical_drift_noise_summary.md").exists()
    rows_text = (output_dir / "unshared_physical_drift_noise_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "source_coding" in rows_text
    assert "source_quantization_mse_mean" in rows_text
    assert "shared_boundary_source_dac_energy_pj" in rows_text
    assert "per_block_source_dac_energy_pj" in rows_text
    assert "drop_vs_trained" in rows_text


def test_synthetic_source_coding_calibration_stress_saves_metrics(
    tmp_path, capsys
) -> None:
    exit_code = main(
        [
            "--run-source-coding-calibration-stress",
            "--synthetic",
            "--seeds",
            "7",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--chain-lengths",
            "4",
            "--calibration-sample-counts",
            "16",
            "--source-coding-projection-mse-threshold",
            "0.18",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "source_coding_calibration_stress"
    assert payload["rows"] == 192
    assert payload["summary_rows"] == 24
    assert (output_dir / "source_coding_calibration_stress_rows.csv").exists()
    assert (output_dir / "source_coding_calibration_stress_summary.md").exists()
    rows_text = (output_dir / "source_coding_calibration_stress_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "redundant_6bit_2phase" in rows_text
    assert "direct_7bit" in rows_text
    assert "redundant_7bit_2phase" in rows_text
    assert "scheduled_plus_projection_mse_trigger" in rows_text
    assert "triggered_recalibration_count" in rows_text
    seed_slice = output_dir / "seed_slices" / "seed_7_rows.csv"
    assert seed_slice.exists()


def test_synthetic_source_coding_calibration_stress_resumes_seed_slices(
    tmp_path, capsys
) -> None:
    args = [
        "--run-source-coding-calibration-stress",
        "--synthetic",
        "--seeds",
        "7",
        "--epochs",
        "1",
        "--max-train-samples",
        "96",
        "--max-test-samples",
        "32",
        "--chain-lengths",
        "4",
        "--calibration-sample-counts",
        "16",
        "--source-coding-projection-mse-threshold",
        "0.18",
        "--output-dir",
        str(tmp_path),
    ]

    assert main(args) == 0
    capsys.readouterr()

    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)

    output_dir = tmp_path / "source_coding_calibration_stress"
    assert payload["rows"] == 192
    assert payload["summary_rows"] == 24
    assert payload["completed_seed_slices"] == []
    assert payload["skipped_seed_slices"] == [7]
    assert (
        output_dir / "seed_slices" / "seed_7_manifest.json"
    ).exists()


def test_downstream_recoverability_report_classifies_dips(tmp_path, capsys) -> None:
    rows_path = tmp_path / "source_coding_rows.csv"
    rows_path.write_text(
        "\n".join(
            [
                "source_coding,calibration_samples,recalibration_policy,"
                "physical_scenario,seed,time_step,calibrated_accuracy",
                "direct_7bit,128,scheduled_only,nominal,7,0,0.856",
                "direct_7bit,128,scheduled_only,nominal,7,1,0.857",
                "direct_7bit,128,scheduled_only,nominal,7,2,0.858",
                "direct_7bit,128,scheduled_only,nominal,7,3,0.856",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,0,0.844",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,1,0.843",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,2,0.842",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,3,0.841",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-downstream-recoverability-report",
            "--source-coding-stress-rows-path",
            str(rows_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "downstream_recoverability"
    assert payload["observed_runs"] == 2
    assert payload["injected_runs"] == 4
    assert payload["observed_runs_with_any_below"] == 1
    assert payload["observed_runs_with_persistent_failure"] == 1
    assert (output_dir / "downstream_recoverability_rows.csv").exists()
    assert (output_dir / "downstream_recoverability_summary.md").exists()
    rows_text = (output_dir / "downstream_recoverability_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "injected_single_step_dip" in rows_text
    assert "two_step_persistence_failure" in rows_text
    assert "leaky_integrator_peak_deficit" in rows_text


def test_persistent_failure_diagnosis_identifies_seed_and_lift(
    tmp_path, capsys
) -> None:
    rows_path = tmp_path / "source_coding_rows.csv"
    rows_path.write_text(
        "\n".join(
            [
                "source_coding,calibration_samples,recalibration_policy,"
                "physical_scenario,seed,time_step,calibrated_accuracy,"
                "baseline_relu_accuracy,trained_chain_accuracy",
                "direct_7bit,128,scheduled_only,nominal,7,0,0.856,0.866,0.861",
                "direct_7bit,128,scheduled_only,nominal,7,1,0.857,0.866,0.861",
                "direct_7bit,128,scheduled_only,nominal,7,2,0.858,0.866,0.861",
                "direct_7bit,128,scheduled_only,nominal,7,3,0.856,0.866,0.861",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,0,"
                "0.844,0.872,0.852",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,1,"
                "0.843,0.872,0.852",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,2,"
                "0.842,0.872,0.852",
                "redundant_6bit_2phase,128,scheduled_only,thermal,11,3,"
                "0.841,0.872,0.852",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-persistent-failure-diagnosis",
            "--source-coding-stress-rows-path",
            str(rows_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "persistent_failure_diagnosis"
    assert payload["run_scenarios"] == 2
    assert payload["persistent_scenarios"] == 1
    assert payload["dominant_failure_seed"] == 11
    assert payload["minimum_lift_to_clear_floor"] == 0.01
    assert (output_dir / "persistent_failure_seed_summary.md").exists()
    assert (output_dir / "persistent_failure_lift_summary.csv").exists()


def test_energy_accuracy_pareto_report_from_existing_summaries(tmp_path, capsys) -> None:
    budget_rows_path = tmp_path / "chained_block_budget_rows.csv"
    budget_rows_path.write_text(
        "\n".join(
            [
                "hardware_scenario,chain_length,calibration_samples,"
                "recalibration_interval_steps,drift_rate,single_block_accuracy_drop_mean,"
                "chain_error_growth_proxy,optical_compute_energy_pj,converter_energy_pj,"
                "detector_tia_energy_pj,laser_energy_pj,digital_correction_energy_pj,"
                "static_control_energy_pj,control_energy_per_inference_pj,"
                "calibration_energy_per_inference_pj,total_energy_per_chain_pj,"
                "energy_per_block_pj,digital_chain_energy_pj,"
                "energy_ratio_vs_digital_chain,isolated_optical_energy_pj,"
                "energy_ratio_vs_isolated_optical,converter_energy_saved_pj,"
                "estimated_latency_ns,latency_per_block_ns,digital_chain_latency_ns,"
                "latency_ratio_vs_digital_chain,isolated_optical_latency_ns,"
                "latency_ratio_vs_isolated_optical,converter_latency_saved_ns,"
                "inferences_per_drift_step",
                "low_io,2,32,1,0.0,0.001,0.0014,100,20,10,1,1,1,0,0,132,66,200,"
                "0.66,220,0.6,40,2.0,1.0,4.0,0.50,4.0,0.5,2.0,1000",
                "low_io,4,32,1,0.0,0.001,0.002,200,20,10,1,1,1,0,0,232,58,400,"
                "0.58,440,0.53,120,2.4,0.6,8.0,0.30,8.0,0.3,5.6,1000",
                "adc_bound,2,32,1,0.0,0.001,0.0014,100,120,80,1,1,1,0,0,302,151,200,"
                "1.51,400,0.76,200,20.0,10.0,4.0,5.00,30.0,0.67,10.0,1000",
                "adc_bound,4,32,1,0.0,0.001,0.002,200,120,80,1,1,1,0,0,402,100.5,400,"
                "1.005,800,0.50,600,21.0,5.25,8.0,2.625,60.0,0.35,39.0,1000",
            ]
        ),
        encoding="utf-8",
    )
    shared_path = tmp_path / "chained_material_training_summary.csv"
    shared_path.write_text(
        "\n".join(
            [
                "scenario,material_family,kind,strength,chain_length,"
                "trained_chain_accuracy_mean,raw_constrained_accuracy_mean,"
                "calibrated_constrained_accuracy_mean,trained_delta_vs_relu_mean,"
                "raw_drop_vs_trained_mean,calibrated_drop_vs_trained_mean,"
                "raw_projection_mse_mean,calibrated_projection_mse_mean,"
                "elapsed_seconds_mean,runs",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,2,0.862,0.854,0.861,-0.006,"
                "0.008,0.001,0.5,0.1,10,3",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,4,0.856,0.835,0.849,-0.012,"
                "0.021,0.008,0.5,0.1,10,3",
            ]
        ),
        encoding="utf-8",
    )
    unshared_path = tmp_path / "unshared_chained_material_training_summary.csv"
    unshared_path.write_text(
        "\n".join(
            [
                "scenario,material_family,kind,strength,chain_length,"
                "trained_chain_accuracy_mean,raw_constrained_accuracy_mean,"
                "calibrated_constrained_accuracy_mean,trained_delta_vs_relu_mean,"
                "raw_drop_vs_trained_mean,calibrated_drop_vs_trained_mean,"
                "raw_projection_mse_mean,calibrated_projection_mse_mean,"
                "elapsed_seconds_mean,runs",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,2,0.858,0.852,0.855,-0.010,"
                "0.007,0.003,0.5,0.1,10,3",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,4,0.866,0.853,0.857,-0.003,"
                "0.013,0.009,0.5,0.1,10,3",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-energy-accuracy-pareto-report",
            "--chained-budget-rows-path",
            str(budget_rows_path),
            "--shared-chained-training-summary-path",
            str(shared_path),
            "--unshared-chained-training-summary-path",
            str(unshared_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "energy_accuracy_pareto"
    assert payload["rows"] == 8
    assert payload["summary_rows"] == 4
    assert payload["pareto_efficient_rows"] >= 1
    assert (output_dir / "energy_accuracy_pareto_rows.csv").exists()
    assert (output_dir / "energy_accuracy_pareto_summary.csv").exists()
    assert (output_dir / "energy_accuracy_pareto_summary.md").exists()
    rows_text = (output_dir / "energy_accuracy_pareto_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "shared_2_block" in rows_text
    assert "single shared CMOS-like output boundary" in rows_text
    assert "pareto_efficient" in rows_text


def test_unshared_four_block_stress_report(tmp_path, capsys) -> None:
    budget_rows_path = tmp_path / "chained_block_budget_rows.csv"
    budget_rows_path.write_text(
        "\n".join(
            [
                "hardware_scenario,chain_length,calibration_samples,"
                "recalibration_interval_steps,drift_rate,single_block_accuracy_drop_mean,"
                "chain_error_growth_proxy,optical_compute_energy_pj,converter_energy_pj,"
                "detector_tia_energy_pj,laser_energy_pj,digital_correction_energy_pj,"
                "static_control_energy_pj,control_energy_per_inference_pj,"
                "calibration_energy_per_inference_pj,total_energy_per_chain_pj,"
                "energy_per_block_pj,digital_chain_energy_pj,"
                "energy_ratio_vs_digital_chain,isolated_optical_energy_pj,"
                "energy_ratio_vs_isolated_optical,converter_energy_saved_pj,"
                "estimated_latency_ns,latency_per_block_ns,digital_chain_latency_ns,"
                "latency_ratio_vs_digital_chain,isolated_optical_latency_ns,"
                "latency_ratio_vs_isolated_optical,converter_latency_saved_ns,"
                "inferences_per_drift_step",
                "published_273fj_low_io,4,32,1,0.0,0.001,0.002,200,288,128,1,1,"
                "1,0,0,618,154.5,400,1.545,800,0.77,600,2.5,0.625,8.0,0.3125,"
                "8.0,0.3125,5.5,1000",
            ]
        ),
        encoding="utf-8",
    )
    unshared_path = tmp_path / "unshared_chained_material_training_summary.csv"
    unshared_path.write_text(
        "\n".join(
            [
                "scenario,material_family,kind,strength,chain_length,"
                "trained_chain_accuracy_mean,raw_constrained_accuracy_mean,"
                "calibrated_constrained_accuracy_mean,trained_delta_vs_relu_mean,"
                "raw_drop_vs_trained_mean,calibrated_drop_vs_trained_mean,"
                "raw_projection_mse_mean,calibrated_projection_mse_mean,"
                "elapsed_seconds_mean,runs",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,4,0.866,0.853,0.857,-0.003,"
                "0.013,0.009,0.5,0.1,10,3",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-unshared-4block-stress-report",
            "--chained-budget-rows-path",
            str(budget_rows_path),
            "--unshared-chained-training-summary-path",
            str(unshared_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "unshared_4block_stress"
    assert payload["rows"] == 24
    assert payload["summary_rows"] == 24
    assert payload["passing_85_percent_rows"] >= 1
    assert (output_dir / "unshared_4block_stress_rows.csv").exists()
    assert (output_dir / "unshared_4block_stress_summary.csv").exists()
    assert (output_dir / "unshared_4block_stress_summary.md").exists()
    rows_text = (output_dir / "unshared_4block_stress_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "combined_drift4x_readout2x" in rows_text
    assert "ccd_8_lane_low_power_bucket" in rows_text
    assert "fails_85_percent_accuracy" in rows_text


def test_synthetic_unshared_physical_drift_noise_saves_aggregate(
    tmp_path,
    capsys,
) -> None:
    exit_code = main(
        [
            "--run-unshared-physical-drift-noise",
            "--synthetic",
            "--epochs",
            "1",
            "--max-train-samples",
            "96",
            "--max-test-samples",
            "32",
            "--hidden-dim",
            "16",
            "--mlp-dim",
            "32",
            "--calibration-samples",
            "16",
            "--seeds",
            "7,11",
            "--physical-architecture-profile",
            "athermal_compensated_mrr",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "unshared_physical_drift_noise"
    assert payload["config"]["seeds"] == [7, 11]
    assert payload["config"]["chain_length"] == 4
    assert payload["rows"] == 64
    assert payload["summary_rows"] == 4
    assert (output_dir / "unshared_physical_drift_noise_rows.csv").exists()
    assert (output_dir / "unshared_physical_drift_noise_summary.csv").exists()
    assert (output_dir / "unshared_physical_drift_noise_summary.md").exists()
    assert (output_dir / "unshared_physical_drift_noise_manifest.json").exists()
    assert (output_dir / "unshared_physical_drift_noise_accuracy.png").exists()
    assert (output_dir / "unshared_physical_drift_noise_projection_mse.png").exists()
    assert "plots" in payload["artifacts"]
    rows_text = (output_dir / "unshared_physical_drift_noise_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "thermal_drift_rate" in rows_text
    assert "output_readout_noise" in rows_text
    assert "athermal_compensated_mrr_nominal" in rows_text
    assert "thermal_excursion_k_per_step" in rows_text
    assert "output_enob_equivalent" in rows_text
    assert "below_85_percent" in rows_text


def test_device_tolerance_target_report(tmp_path, capsys) -> None:
    summary_path = tmp_path / "unshared_physical_drift_noise_summary.csv"
    summary_path.write_text(
        "\n".join(
            [
                "scenario,material_family,kind,strength,chain_length,"
                "physical_scenario,trained_chain_accuracy_mean,"
                "calibrated_accuracy_mean,calibrated_accuracy_min,"
                "final_step_accuracy_mean,mean_drop_vs_trained,"
                "worst_drop_vs_trained,projection_mse_mean,recalibrations_mean,"
                "steps_below_85_percent,runs,rows",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,4,warm_cmos_boundary,0.866,"
                "0.857,0.851,0.857,0.009,0.013,0.176,1.5,0,2,16",
                "reverse_saturable_strength_0p2,reverse_saturable_absorber,"
                "reverse_saturable_absorption,0.2,4,hot_noisy_boundary,0.866,"
                "0.856,0.849,0.856,0.010,0.014,0.192,1.5,1,2,16",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-device-tolerance-target-report",
            "--physical-drift-noise-summary-path",
            str(summary_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "device_tolerance_targets"
    assert payload["rows"] == 4
    assert (output_dir / "device_tolerance_targets.csv").exists()
    assert (output_dir / "device_tolerance_targets.md").exists()
    assert (output_dir / "device_tolerance_targets_manifest.json").exists()
    target_text = (output_dir / "device_tolerance_targets.csv").read_text(
        encoding="utf-8"
    )
    assert "recommended_operating_envelope" in target_text
    assert "projection_mse_early_warning" in target_text
    assert "0.192" in target_text


def test_calibration_bridge_report(tmp_path, capsys) -> None:
    targets_path = tmp_path / "device_tolerance_targets.csv"
    targets_path.write_text(
        "\n".join(
            [
                "target,max_thermal_drift_rate,max_detector_noise,"
                "max_output_readout_noise,max_recalibration_interval_steps,"
                "min_accuracy_floor,projection_mse_warning_threshold,status,"
                "evidence,notes",
                "recommended_operating_envelope,0.002,0.02,0.005,4,0.85,"
                "0.191,pass,warm_cmos_boundary,target",
                "warning_edge,0.005,0.03,0.01,4,0.85,0.191,marginal,"
                "hot_noisy_boundary,warning",
                "projection_mse_early_warning,,,,,0.85,0.191,monitor,mse,"
                "monitor",
                "recalibration_cadence,0.002,0.02,0.005,4,0.85,0.191,"
                "recommended,warm_cmos_boundary,cadence",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-calibration-bridge-report",
            "--device-tolerance-targets-path",
            str(targets_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "calibration_bridge"
    assert payload["rows"] == 7
    assert payload["sensitivity_rows"] == 4
    assert (output_dir / "calibration_bridge_rows.csv").exists()
    assert (output_dir / "calibration_bridge.md").exists()
    assert (output_dir / "calibration_bridge_sensitivity.csv").exists()
    assert (output_dir / "calibration_bridge_sensitivity.md").exists()
    assert (output_dir / "calibration_bridge_manifest.json").exists()
    bridge_text = (output_dir / "calibration_bridge_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "thermal_resonance_drift" in bridge_text
    assert ">= 2500 photons/sample" in bridge_text
    assert "ENOB equivalent" in bridge_text
    sensitivity_text = (
        output_dir / "calibration_bridge_sensitivity.csv"
    ).read_text(encoding="utf-8")
    assert "tight_mrr_weight_bank" in sensitivity_text
    assert "athermal_compensated_mrr" in sensitivity_text
    assert "recalibration_window_k" in sensitivity_text


def test_architecture_energy_overlay_report(tmp_path, capsys) -> None:
    pareto_path = tmp_path / "energy_accuracy_pareto_rows.csv"
    pareto_path.write_text(
        "\n".join(
            [
                "candidate,hardware_scenario,total_energy_per_chain_pj,"
                "energy_ratio_vs_digital_chain,estimated_latency_ns,"
                "latency_ratio_vs_digital_chain",
                "unshared_4_block,published_273fj_low_io,1000,0.25,2.0,0.5",
                "shared_2_block,published_273fj_low_io,900,0.20,1.5,0.4",
            ]
        ),
        encoding="utf-8",
    )
    physical_path = tmp_path / "unshared_physical_drift_noise_rows.csv"
    physical_path.write_text(
        "\n".join(
            [
                "seed,physical_scenario,architecture_profile,calibrated_accuracy,"
                "projection_mse_mean,below_85_percent,thermal_drift_rate,"
                "detector_noise,output_readout_noise,recalibration_interval_steps,"
                "resonance_shift_pm_per_step,thermal_excursion_k_per_step,"
                "detector_photons_per_sample,output_photons_per_sample,"
                "output_enob_equivalent",
                "7,athermal_compensated_mrr_nominal,athermal_compensated_mrr,"
                "0.858,0.176,False,0.002,0.02,0.005,4,4.8,0.704,"
                "2500,40000,7.35",
                "11,athermal_compensated_mrr_nominal,athermal_compensated_mrr,"
                "0.852,0.177,False,0.002,0.02,0.005,4,4.8,0.704,"
                "2500,40000,7.35",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-architecture-energy-overlay-report",
            "--energy-accuracy-pareto-rows-path",
            str(pareto_path),
            "--physical-drift-noise-rows-path",
            str(physical_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "architecture_energy_overlay"
    assert payload["rows"] == 1
    assert payload["summary_rows"] == 1
    assert (output_dir / "architecture_energy_overlay_rows.csv").exists()
    assert (output_dir / "architecture_energy_overlay_summary.md").exists()
    rows_text = (output_dir / "architecture_energy_overlay_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "profile_energy_overhead_pj" in rows_text
    assert "profile_output_readout_energy_pj" in rows_text
    assert "athermal_compensated_mrr_nominal" in rows_text


def test_architecture_energy_stress_report(tmp_path, capsys) -> None:
    pareto_path = tmp_path / "energy_accuracy_pareto_rows.csv"
    pareto_path.write_text(
        "\n".join(
            [
                "candidate,hardware_scenario,total_energy_per_chain_pj,"
                "energy_ratio_vs_digital_chain,estimated_latency_ns,"
                "latency_ratio_vs_digital_chain",
                "unshared_4_block,published_273fj_low_io,1000,0.25,2.0,0.5",
            ]
        ),
        encoding="utf-8",
    )
    physical_path = tmp_path / "unshared_physical_drift_noise_rows.csv"
    physical_path.write_text(
        "\n".join(
            [
                "seed,physical_scenario,architecture_profile,calibrated_accuracy,"
                "projection_mse_mean,below_85_percent,thermal_drift_rate,"
                "detector_noise,output_readout_noise,recalibration_interval_steps,"
                "resonance_shift_pm_per_step,thermal_excursion_k_per_step,"
                "detector_photons_per_sample,output_photons_per_sample,"
                "output_enob_equivalent",
                "7,athermal_compensated_mrr_nominal,athermal_compensated_mrr,"
                "0.858,0.176,False,0.002,0.02,0.005,4,4.8,0.704,"
                "2500,40000,7.35",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-architecture-energy-stress-report",
            "--energy-accuracy-pareto-rows-path",
            str(pareto_path),
            "--physical-drift-noise-rows-path",
            str(physical_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "architecture_energy_stress"
    assert payload["rows"] == 16
    assert payload["summary_rows"] == 16
    assert (output_dir / "architecture_energy_stress_rows.csv").exists()
    assert (output_dir / "architecture_energy_stress_summary.md").exists()
    rows_text = (output_dir / "architecture_energy_stress_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "thermal_control_pj_per_k_per_block" in rows_text
    assert "resonance_tracking_pj_per_block" in rows_text
    assert "passes_all_gates" in rows_text


def test_architecture_converter_stress_report(tmp_path, capsys) -> None:
    pareto_path = tmp_path / "energy_accuracy_pareto_rows.csv"
    pareto_path.write_text(
        "\n".join(
            [
                "candidate,hardware_scenario,total_energy_per_chain_pj,"
                "energy_ratio_vs_digital_chain,converter_energy_pj,"
                "detector_tia_energy_pj,estimated_latency_ns,"
                "latency_ratio_vs_digital_chain",
                "unshared_4_block,published_273fj_low_io,1000,0.25,"
                "288,128,2.0,0.5",
            ]
        ),
        encoding="utf-8",
    )
    physical_path = tmp_path / "unshared_physical_drift_noise_rows.csv"
    physical_path.write_text(
        "\n".join(
            [
                "seed,physical_scenario,architecture_profile,calibrated_accuracy,"
                "projection_mse_mean,below_85_percent,thermal_drift_rate,"
                "detector_noise,output_readout_noise,recalibration_interval_steps,"
                "resonance_shift_pm_per_step,thermal_excursion_k_per_step,"
                "detector_photons_per_sample,output_photons_per_sample,"
                "output_enob_equivalent",
                "7,athermal_compensated_mrr_nominal,athermal_compensated_mrr,"
                "0.858,0.176,False,0.002,0.02,0.005,4,4.8,0.704,"
                "2500,40000,7.35",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-architecture-converter-stress-report",
            "--energy-accuracy-pareto-rows-path",
            str(pareto_path),
            "--physical-drift-noise-rows-path",
            str(physical_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "architecture_converter_stress"
    assert payload["rows"] == 144
    assert payload["summary_rows"] == 144
    assert (output_dir / "architecture_converter_stress_rows.csv").exists()
    assert (output_dir / "architecture_converter_stress_summary.md").exists()
    rows_text = (output_dir / "architecture_converter_stress_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "converter_placement" in rows_text
    assert "adc_energy_multiplier" in rows_text
    assert "stressed_converter_readout_pj" in rows_text


def test_architecture_converter_target_report(tmp_path, capsys) -> None:
    stress_path = tmp_path / "architecture_converter_stress_rows.csv"
    stress_path.write_text(
        "\n".join(
            [
                "candidate,architecture_profile,physical_scenario,hardware_scenario,"
                "converter_placement,adc_energy_multiplier,dac_energy_multiplier,"
                "readout_energy_multiplier,input_dac_boundaries,output_adc_boundaries,"
                "detector_readout_boundaries,stressed_dac_energy_pj,"
                "stressed_adc_energy_pj,stressed_readout_energy_pj,"
                "stressed_converter_readout_pj,profile_energy_ratio_vs_digital,"
                "profile_latency_ratio_vs_digital,passes_all_gates",
                "unshared_4_block,athermal_compensated_mrr,nominal,"
                "published_273fj_low_io,shared_input_output_boundary,1,1,1,"
                "1,1,1,32,256,128,416,0.3,0.4,True",
                "unshared_4_block,athermal_compensated_mrr,nominal,"
                "published_273fj_low_io,shared_input_output_boundary,25,5,25,"
                "1,1,1,160,6400,3200,9760,0.9,0.4,True",
                "unshared_4_block,athermal_compensated_mrr,nominal,"
                "high_precision_adc_bound,shared_input_output_boundary,1,1,1,"
                "1,1,1,640,256000,2560,259200,16.0,3.0,False",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-architecture-converter-target-report",
            "--architecture-converter-stress-rows-path",
            str(stress_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "architecture_converter_target"
    assert payload["rows"] == 1
    assert payload["max_adc_pj_per_output"] == 25.0
    assert (output_dir / "architecture_converter_target_rows.csv").exists()
    rows_text = (output_dir / "architecture_converter_target_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "max_dac_pj_per_input" in rows_text
    assert "max_readout_tia_pj_per_output" in rows_text


def test_architecture_converter_profile_report(tmp_path, capsys) -> None:
    pareto_path = tmp_path / "energy_accuracy_pareto_rows.csv"
    pareto_path.write_text(
        "\n".join(
            [
                "candidate,hardware_scenario,total_energy_per_chain_pj,"
                "energy_ratio_vs_digital_chain,converter_energy_pj,"
                "detector_tia_energy_pj,estimated_latency_ns,"
                "latency_ratio_vs_digital_chain",
                "unshared_4_block,published_273fj_low_io,1000,0.25,"
                "288,128,2.0,0.5",
            ]
        ),
        encoding="utf-8",
    )
    physical_path = tmp_path / "unshared_physical_drift_noise_rows.csv"
    physical_path.write_text(
        "\n".join(
            [
                "seed,physical_scenario,architecture_profile,calibrated_accuracy,"
                "projection_mse_mean,below_85_percent,thermal_drift_rate,"
                "detector_noise,output_readout_noise,recalibration_interval_steps,"
                "resonance_shift_pm_per_step,thermal_excursion_k_per_step,"
                "detector_photons_per_sample,output_photons_per_sample,"
                "output_enob_equivalent",
                "7,athermal_compensated_mrr_nominal,athermal_compensated_mrr,"
                "0.858,0.176,False,0.002,0.02,0.005,4,4.8,0.704,"
                "2500,40000,7.35",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-architecture-converter-profile-report",
            "--energy-accuracy-pareto-rows-path",
            str(pareto_path),
            "--physical-drift-noise-rows-path",
            str(physical_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "architecture_converter_profile"
    assert payload["rows"] == 15
    assert payload["summary_rows"] == 15
    assert (output_dir / "architecture_converter_profile_rows.csv").exists()
    assert (output_dir / "architecture_converter_profile_summary.md").exists()
    rows_text = (output_dir / "architecture_converter_profile_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "converter_profile" in rows_text
    assert "inp_2022_10ghz_transceiver" in rows_text
    assert "ad9172_12gsps_rf_dac_reference" in rows_text
    assert "adc_source_chain" in rows_text
    assert "dac_bandwidth_gsps" in rows_text
    assert "dac_sfdr_db" in rows_text
    assert "dac_inl_lsb" in rows_text
    assert "dac_meets_source_chain_target" in rows_text
    assert "detector_component" in rows_text
    assert "tia_pj_per_output" in rows_text
    assert "profiled_converter_readout_pj" in rows_text


def test_architecture_dac_coding_sweep_report(tmp_path, capsys) -> None:
    pareto_path = tmp_path / "energy_accuracy_pareto_rows.csv"
    pareto_path.write_text(
        "\n".join(
            [
                "candidate,hardware_scenario,total_energy_per_chain_pj,"
                "energy_ratio_vs_digital_chain,converter_energy_pj,"
                "detector_tia_energy_pj,estimated_latency_ns,"
                "latency_ratio_vs_digital_chain",
                "unshared_4_block,published_273fj_low_io,1000,0.25,"
                "288,128,2.0,0.5",
            ]
        ),
        encoding="utf-8",
    )
    physical_path = tmp_path / "unshared_physical_drift_noise_rows.csv"
    physical_path.write_text(
        "\n".join(
            [
                "seed,physical_scenario,architecture_profile,calibrated_accuracy,"
                "projection_mse_mean,below_85_percent,thermal_drift_rate,"
                "detector_noise,output_readout_noise,recalibration_interval_steps,"
                "resonance_shift_pm_per_step,thermal_excursion_k_per_step,"
                "detector_photons_per_sample,output_photons_per_sample,"
                "output_enob_equivalent",
                "7,athermal_compensated_mrr_nominal,athermal_compensated_mrr,"
                "0.858,0.176,False,0.002,0.02,0.005,4,4.8,0.704,"
                "2500,40000,7.35",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--run-architecture-dac-coding-sweep-report",
            "--energy-accuracy-pareto-rows-path",
            str(pareto_path),
            "--physical-drift-noise-rows-path",
            str(physical_path),
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    output_dir = tmp_path / "architecture_dac_coding_sweep"
    assert payload["rows"] == 45
    assert payload["summary_rows"] == 45
    assert (output_dir / "architecture_dac_coding_sweep_rows.csv").exists()
    assert (output_dir / "architecture_dac_coding_sweep_summary.md").exists()
    rows_text = (output_dir / "architecture_dac_coding_sweep_rows.csv").read_text(
        encoding="utf-8"
    )
    assert "source_coding" in rows_text
    assert "source_bits" in rows_text
    assert "modeled_accuracy_penalty_percent" in rows_text
    assert "effective_min_accuracy_percent" in rows_text
    assert "passes_all_gates" in rows_text
