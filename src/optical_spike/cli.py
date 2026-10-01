"""Command-line entrypoint for the optical simulation spike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from optical_spike.adaptation import AdaptationConfig, run_adaptation_modes
from optical_spike.baseline import BaselineConfig, run_baseline
from optical_spike.calibration_cost import (
    CalibrationCostConfig,
    parse_float_tuple,
    parse_int_tuple,
    run_calibration_cost,
)
from optical_spike.calibration_bridge import (
    CalibrationBridgeConfig,
    run_calibration_bridge_report,
)
from optical_spike.constrained_compare import (
    ConstrainedComparisonConfig,
    run_constrained_comparison,
)
from optical_spike.cifar_feature_target import (
    CifarFeatureTargetConfig,
    CifarFeatureSourceCodingStressConfig,
    run_cifar_feature_source_coding_stress,
    run_cifar_feature_target,
)
from optical_spike.downstream_recoverability import (
    DownstreamRecoverabilityConfig,
    run_downstream_recoverability_report,
)
from optical_spike.energy_budget import (
    ArchitectureConverterStressConfig,
    ArchitectureConverterProfileConfig,
    ArchitectureConverterTargetConfig,
    ArchitectureDacCodingSweepConfig,
    ArchitectureEnergyOverlayConfig,
    ArchitectureEnergyStressConfig,
    ChainedBlockBudgetConfig,
    ChargeReadoutBudgetConfig,
    EnergyAccuracyParetoConfig,
    EnergyBudgetConfig,
    UnsharedFourBlockStressConfig,
    run_chained_block_budget,
    run_architecture_converter_profile_report,
    run_architecture_converter_stress_report,
    run_architecture_converter_target_report,
    run_architecture_dac_coding_sweep_report,
    run_architecture_energy_overlay_report,
    run_architecture_energy_stress_report,
    run_charge_readout_budget,
    run_energy_accuracy_pareto_report,
    run_energy_budget,
    run_unshared_four_block_stress_report,
)
from optical_spike.fabrication_constraints import DEFAULT_CONSTRAINED_FIXED_SCENARIOS
from optical_spike.fixed import FixedSweepConfig, run_fixed_error_sweeps
from optical_spike.material_nonlinearity import (
    MaterialNonlinearityConfig,
    run_material_nonlinearity_sweeps,
)
from optical_spike.material_training import (
    ChainedMaterialTrainingConfig,
    DIRECT_7BIT_SOURCE_CODING,
    DeviceToleranceTargetConfig,
    FunctionalChainConfig,
    MaterialStrengthSweepConfig,
    MaterialTrainingConfig,
    REDUNDANT_6BIT_SOURCE_CODING,
    REDUNDANT_7BIT_SOURCE_CODING,
    SourceCodingCalibrationStressConfig,
    UnsharedChainedMaterialTrainingConfig,
    UnsharedPhysicalDriftNoiseConfig,
    physical_scenarios_for_architecture,
    run_chained_material_training,
    run_device_tolerance_target_report,
    run_functional_chain_simulation,
    run_material_strength_sweep,
    run_material_training,
    run_source_coding_calibration_stress,
    run_unshared_physical_drift_noise,
    run_unshared_chained_material_training,
)
from optical_spike.multiseed import MultiSeedConfig, parse_seeds, run_multi_seed
from optical_spike.nonlinear import NonlinearSweepConfig, run_nonlinear_variant_sweeps
from optical_spike.persistent_failure_diagnosis import (
    PersistentFailureDiagnosisConfig,
    run_persistent_failure_diagnosis,
)
from optical_spike.quantization import (
    QuantizationSweepConfig,
    parse_bits,
    run_quantized_projection_sweeps,
)
from optical_spike.reporting import ReportConfig, generate_report
from optical_spike.source_coding_ablation import (
    SourceCodingAblationReportConfig,
    run_source_coding_ablation_report,
)
from optical_spike.transformer_mlp import (
    TransformerMLPConfig,
    parse_transformer_bits,
    run_transformer_mlp_target,
)
from optical_spike.tunable import TunableSweepConfig, run_tunable_error_sweeps


def source_coding_chain_length(chain_lengths_arg: str) -> int:
    chain_lengths = parse_int_tuple(chain_lengths_arg)
    return 4 if 4 in chain_lengths else chain_lengths[-1]


DEFAULT_OUTPUT_DIR = Path("artifacts/spike")


def parse_constrained_fixed_scenario(name: str):
    for scenario in DEFAULT_CONSTRAINED_FIXED_SCENARIOS:
        if scenario.name == name:
            return scenario
    valid = ", ".join(scenario.name for scenario in DEFAULT_CONSTRAINED_FIXED_SCENARIOS)
    raise argparse.ArgumentTypeError(
        f"unknown constrained scenario {name!r}; choose one of: {valid}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="optical-spike",
        description="Run or inspect the Fashion-MNIST optical projection simulation spike.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        type=Path,
        help="Directory for metrics, plots, and intermediate artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned spike stages without training or downloading data.",
    )
    parser.add_argument(
        "--run-baseline",
        action="store_true",
        help="Train the Fashion-MNIST digital baseline and save metrics.",
    )
    parser.add_argument(
        "--run-quantized-sweeps",
        action="store_true",
        help="Train the baseline and evaluate quantized versions of the optical projection.",
    )
    parser.add_argument(
        "--run-tunable-errors",
        action="store_true",
        help="Train the baseline and evaluate tunable optical projection error scenarios.",
    )
    parser.add_argument(
        "--run-fixed-errors",
        action="store_true",
        help="Train the baseline and evaluate fixed fabricated optical projection error scenarios.",
    )
    parser.add_argument(
        "--run-nonlinear-variants",
        action="store_true",
        help="Train the baseline and evaluate nonlinear optical projection variants.",
    )
    parser.add_argument(
        "--run-adaptation-modes",
        action="store_true",
        help="Train the baseline and evaluate optical transfer adaptation strategies.",
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Build CSV, Markdown, and PNG summaries from available spike metrics.",
    )
    parser.add_argument(
        "--run-multi-seed",
        action="store_true",
        help="Run all spike stages across multiple seeds and aggregate the results.",
    )
    parser.add_argument(
        "--run-transformer-mlp-target",
        action="store_true",
        help="Train a transformer-style MLP block target and quantize its expansion projection.",
    )
    parser.add_argument(
        "--run-cifar-feature-target",
        action="store_true",
        help="Train a CIFAR-10 spatial feature target with an optical-candidate projection.",
    )
    parser.add_argument(
        "--run-cifar-feature-source-coding-stress",
        action="store_true",
        help="Stress CIFAR-10 spatial feature projection under source coding and drift/noise.",
    )
    parser.add_argument(
        "--run-constrained-fixed-comparison",
        action="store_true",
        help="Compare constrained fixed-fabrication sweeps for input projection vs MLP-up target.",
    )
    parser.add_argument(
        "--run-calibration-cost",
        action="store_true",
        help="Sweep calibration sample count, recalibration cadence, drift, and control overhead.",
    )
    parser.add_argument(
        "--run-material-nonlinearities",
        action="store_true",
        help="Evaluate material-inspired nonlinearities as distortions and activation candidates.",
    )
    parser.add_argument(
        "--run-energy-budget",
        action="store_true",
        help="Estimate calibrated optical MLP-up energy and latency from calibration summary rows.",
    )
    parser.add_argument(
        "--run-chained-block-budget",
        action="store_true",
        help="Estimate energy and latency when several optical blocks share one ADC/DAC boundary.",
    )
    parser.add_argument(
        "--run-charge-readout-budget",
        action="store_true",
        help="Compare CMOS column readout with CCD-style shared charge-bucket readout.",
    )
    parser.add_argument(
        "--run-material-training",
        action="store_true",
        help="Train MLP-up models end-to-end with material-inspired activation functions.",
    )
    parser.add_argument(
        "--run-material-strength-sweep",
        action="store_true",
        help="Sweep material activation strengths and evaluate constrained fixed transfer.",
    )
    parser.add_argument(
        "--run-functional-chain",
        action="store_true",
        help="Simulate repeated material blocks with accumulated constrained optical error.",
    )
    parser.add_argument(
        "--run-chained-material-training",
        action="store_true",
        help="Train repeated material blocks end-to-end before constrained optical transfer.",
    )
    parser.add_argument(
        "--run-unshared-chained-material-training",
        action="store_true",
        help="Train separate material weights per chained block before constrained transfer.",
    )
    parser.add_argument(
        "--run-energy-accuracy-pareto-report",
        action="store_true",
        help="Join chained accuracy and hardware budgets into an energy/accuracy Pareto report.",
    )
    parser.add_argument(
        "--run-unshared-4block-stress-report",
        action="store_true",
        help="Stress-test the unshared 4-block chain across drift and readout assumptions.",
    )
    parser.add_argument(
        "--run-unshared-physical-drift-noise",
        action="store_true",
        help="Train and directly evaluate the unshared 4-block chain under physical drift/noise scenarios.",
    )
    parser.add_argument(
        "--run-redundant-source-validation",
        action="store_true",
        help="Validate the 6-bit redundant source-coding branch in the unshared 4-block transfer simulation.",
    )
    parser.add_argument(
        "--run-source-coding-calibration-stress",
        action="store_true",
        help=(
            "Compare 6-bit redundant, 7-bit direct, and 7-bit redundant source "
            "coding across calibration counts and MSE-triggered recalibration."
        ),
    )
    parser.add_argument(
        "--run-downstream-recoverability-report",
        action="store_true",
        help=(
            "Classify source-coding accuracy dips under downstream hard-floor, "
            "persistence, majority, and leaky-integrator failure models."
        ),
    )
    parser.add_argument(
        "--run-persistent-failure-diagnosis",
        action="store_true",
        help=(
            "Diagnose persistent under-floor source-coding runs by seed, "
            "physical scenario, source coding, policy, and lift scenario."
        ),
    )
    parser.add_argument(
        "--run-source-coding-ablation-report",
        action="store_true",
        help=(
            "Reduce source-coding stress rows into factor summaries for source "
            "coding, thermal drift, detector noise, readout noise, "
            "recalibration policy, and training margin."
        ),
    )
    parser.add_argument(
        "--run-device-tolerance-target-report",
        action="store_true",
        help="Convert physical drift/noise curves into explicit device tolerance targets.",
    )
    parser.add_argument(
        "--run-calibration-bridge-report",
        action="store_true",
        help="Map normalized tolerance targets to first-pass physical device quantities.",
    )
    parser.add_argument(
        "--run-architecture-energy-overlay-report",
        action="store_true",
        help="Overlay selected physical architecture overheads on energy/latency rows.",
    )
    parser.add_argument(
        "--run-architecture-energy-stress-report",
        action="store_true",
        help="Stress selected architecture energy across control/tracking overheads.",
    )
    parser.add_argument(
        "--run-architecture-converter-stress-report",
        action="store_true",
        help="Stress selected architecture energy across converter placement and ADC/DAC/readout assumptions.",
    )
    parser.add_argument(
        "--run-architecture-converter-target-report",
        action="store_true",
        help="Derive per-port ADC/DAC/readout targets from passing converter-stress rows.",
    )
    parser.add_argument(
        "--run-architecture-converter-profile-report",
        action="store_true",
        help="Evaluate named literature-backed converter/readout profiles across placement assumptions.",
    )
    parser.add_argument(
        "--run-architecture-dac-coding-sweep-report",
        action="store_true",
        help="Sweep lower-resolution DAC/source coding assumptions against accuracy, energy, and latency gates.",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        type=Path,
        help="Directory for generated summary tables and plots.",
    )
    parser.add_argument("--seeds", default="7,11,13")
    parser.add_argument("--quantization-bits", default="8,6,4,3,2")
    parser.add_argument("--calibration-samples", default=128, type=int)
    parser.add_argument("--calibration-sample-counts", default="32,128,512,2048")
    parser.add_argument(
        "--constrained-fixed-scenario",
        default="signed_tiled_severe",
        type=parse_constrained_fixed_scenario,
        help=(
            "Named constrained fixed-fabrication projection scenario for CIFAR feature "
            "source-coding stress."
        ),
    )
    parser.add_argument("--recalibration-intervals", default="1,4,16")
    parser.add_argument(
        "--source-coding-projection-mse-threshold",
        default=0.18,
        type=float,
        help="Projection-MSE threshold for triggered source-coding recalibration.",
    )
    parser.add_argument(
        "--source-coding-stress-rows-path",
        default=Path(
            "artifacts/source-coding-calibration-stress-full-2026-07-06/"
            "source_coding_calibration_stress/"
            "source_coding_calibration_stress_rows.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--source-coding-ablation-rows-path",
        default=Path(
            "artifacts/source-coding-calibration-sensitivity-12seed-20epoch-2026-08-11/"
            "source_coding_calibration_stress/"
            "source_coding_calibration_stress_rows.csv"
        ),
        type=Path,
        help="CSV rows to reduce with --run-source-coding-ablation-report.",
    )
    parser.add_argument("--downstream-accuracy-floor", default=0.85, type=float)
    parser.add_argument("--downstream-leaky-budget", default=0.02, type=float)
    parser.add_argument("--downstream-leaky-decay", default=0.5, type=float)
    parser.add_argument("--drift-rates", default="0.0,0.002,0.005")
    parser.add_argument("--time-steps", default=16, type=int)
    parser.add_argument("--material-strengths", default="0.1,0.2,0.35")
    parser.add_argument("--chain-lengths", default="1,2,4,8")
    parser.add_argument(
        "--calibration-summary-path",
        default=Path(
            "artifacts/calibration-cost-full-2026-06-21/calibration_cost/"
            "calibration_cost_summary.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--chained-budget-rows-path",
        default=Path(
            "artifacts/chained-block-budget-2026-06-21/chained_block_budget/"
            "chained_block_budget_rows.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--shared-chained-training-summary-path",
        default=Path(
            "artifacts/chained-material-training-2026-06-21/chained_material_training/"
            "chained_material_training_summary.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--unshared-chained-training-summary-path",
        default=Path(
            "artifacts/unshared-chained-material-training-2026-06-22/"
            "unshared_chained_material_training/"
            "unshared_chained_material_training_summary.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--physical-drift-noise-summary-path",
        default=Path(
            "artifacts/unshared-physical-drift-noise-2026-06-22/"
            "unshared_physical_drift_noise/"
            "unshared_physical_drift_noise_summary.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--physical-drift-noise-rows-path",
        default=Path(
            "artifacts/athermal-physical-drift-noise-2026-06-22/"
            "unshared_physical_drift_noise/"
            "unshared_physical_drift_noise_rows.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--energy-accuracy-pareto-rows-path",
        default=Path(
            "artifacts/energy-accuracy-pareto-2026-06-22/"
            "energy_accuracy_pareto/"
            "energy_accuracy_pareto_rows.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--architecture-converter-stress-rows-path",
        default=Path(
            "artifacts/athermal-converter-stress-2026-06-23/"
            "architecture_converter_stress/"
            "architecture_converter_stress_rows.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--device-tolerance-targets-path",
        default=Path(
            "artifacts/device-tolerance-targets-2026-06-22/"
            "device_tolerance_targets/"
            "device_tolerance_targets.csv"
        ),
        type=Path,
    )
    parser.add_argument(
        "--physical-architecture-profile",
        default=None,
        help=(
            "Use architecture-specific physical drift/noise scenarios, e.g. "
            "athermal_compensated_mrr."
        ),
    )
    parser.add_argument("--inferences-per-drift-step", default=10_000, type=int)
    parser.add_argument("--adaptation-epochs", default=8, type=int)
    parser.add_argument("--adaptation-learning-rate", default=0.01, type=float)
    parser.add_argument("--data-dir", default=Path("data/fashion-mnist"), type=Path)
    parser.add_argument(
        "--dataset",
        default="fashion_mnist",
        choices=("fashion_mnist", "cifar10"),
        help="Real dataset to load when --synthetic is not set.",
    )
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=0.003, type=float)
    parser.add_argument("--hidden-dim", default=64, type=int)
    parser.add_argument("--mlp-dim", default=256, type=int)
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--max-train-samples", default=None, type=int)
    parser.add_argument("--max-test-samples", default=None, type=int)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use deterministic synthetic data for fast smoke tests.",
    )
    return parser


def planned_stages() -> list[str]:
    return [
        "digital_baseline",
        "quantized_projection_sweeps",
        "tunable_optical_error_models",
        "fixed_fabricated_error_models",
        "nonlinear_optical_variants",
        "calibration_and_adaptation_modes",
        "tables_and_plots",
        "results_summary",
        "multi_seed_full_split",
        "transformer_mlp_block_target",
        "cifar10_spatial_feature_projection_target",
        "cifar10_spatial_feature_source_coding_stress",
        "constrained_fixed_target_comparison",
        "calibration_cost_model",
        "material_nonlinearity_candidates",
        "energy_latency_budget",
        "material_activation_training",
        "material_strength_constrained_transfer",
        "chained_optical_block_budget",
        "charge_bucket_readout_budget",
        "functional_chained_block_simulation",
        "chained_material_training",
        "unshared_chained_material_training",
        "energy_accuracy_pareto_report",
        "unshared_4block_stress_report",
        "unshared_physical_drift_noise_evaluation",
        "redundant_source_coding_validation",
        "source_coding_calibration_stress",
        "source_coding_ablation_report",
        "downstream_recoverability_report",
        "persistent_failure_diagnosis",
        "device_tolerance_target_report",
        "calibration_bridge_physical_units",
        "architecture_energy_overlay_report",
        "architecture_energy_stress_report",
        "architecture_converter_stress_report",
        "architecture_converter_target_report",
        "architecture_converter_profile_report",
        "architecture_dac_coding_sweep_report",
    ]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dataset == "cifar10" and args.data_dir == Path("data/fashion-mnist"):
        args.data_dir = Path("data/cifar10")

    baseline_config = BaselineConfig(
        data_dir=args.data_dir,
        dataset=args.dataset,
        output_dir=args.output_dir / "baseline",
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        synthetic=args.synthetic,
    )

    if args.run_quantized_sweeps:
        config = QuantizationSweepConfig(
            baseline=baseline_config,
            bits=parse_bits(args.quantization_bits),
        )
        metrics = run_quantized_projection_sweeps(config)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_tunable_errors:
        config = TunableSweepConfig(
            baseline=baseline_config,
            calibration_samples=args.calibration_samples,
        )
        metrics = run_tunable_error_sweeps(config)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_fixed_errors:
        config = FixedSweepConfig(
            baseline=baseline_config,
            calibration_samples=args.calibration_samples,
        )
        metrics = run_fixed_error_sweeps(config)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_nonlinear_variants:
        config = NonlinearSweepConfig(
            baseline=baseline_config,
            calibration_samples=args.calibration_samples,
        )
        metrics = run_nonlinear_variant_sweeps(config)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_adaptation_modes:
        config = AdaptationConfig(
            baseline=baseline_config,
            calibration_samples=args.calibration_samples,
            adaptation_epochs=args.adaptation_epochs,
            adaptation_learning_rate=args.adaptation_learning_rate,
        )
        metrics = run_adaptation_modes(config)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_baseline:
        metrics = run_baseline(baseline_config)
        print(json.dumps(metrics, indent=2))
        return 0

    if args.generate_report:
        metrics = generate_report(
            ReportConfig(
                output_dir=args.output_dir,
                report_dir=args.report_dir,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_multi_seed:
        metrics = run_multi_seed(
            MultiSeedConfig(
                baseline=baseline_config,
                seeds=parse_seeds(args.seeds),
                quantization_bits=parse_bits(args.quantization_bits),
                calibration_samples=args.calibration_samples,
                adaptation_epochs=args.adaptation_epochs,
                adaptation_learning_rate=args.adaptation_learning_rate,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_transformer_mlp_target:
        metrics = run_transformer_mlp_target(
            TransformerMLPConfig(
                data_dir=args.data_dir,
                dataset=args.dataset,
                output_dir=args.output_dir / "transformer_mlp_target",
                hidden_dim=args.hidden_dim,
                mlp_dim=args.mlp_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                synthetic=args.synthetic,
                quantization_bits=parse_transformer_bits(args.quantization_bits),
                calibration_samples=args.calibration_samples,
                adaptation_epochs=args.adaptation_epochs,
                adaptation_learning_rate=args.adaptation_learning_rate,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_cifar_feature_target:
        if args.dataset != "cifar10":
            parser.error("--run-cifar-feature-target requires --dataset cifar10")
        metrics = run_cifar_feature_target(
            CifarFeatureTargetConfig(
                data_dir=args.data_dir,
                output_dir=args.output_dir / "cifar_feature_target",
                hidden_dim=args.hidden_dim,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                quantization_bits=parse_bits(args.quantization_bits),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_cifar_feature_source_coding_stress:
        if args.dataset != "cifar10":
            parser.error("--run-cifar-feature-source-coding-stress requires --dataset cifar10")
        metrics = run_cifar_feature_source_coding_stress(
            CifarFeatureSourceCodingStressConfig(
                target=CifarFeatureTargetConfig(
                    data_dir=args.data_dir,
                    output_dir=args.output_dir / "cifar_feature_target",
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    quantization_bits=parse_bits(args.quantization_bits),
                ),
                output_dir=args.output_dir / "cifar_feature_source_coding_stress",
                calibration_sample_counts=parse_int_tuple(args.calibration_sample_counts),
                constrained_scenario=args.constrained_fixed_scenario,
                source_codings=(
                    REDUNDANT_6BIT_SOURCE_CODING,
                    DIRECT_7BIT_SOURCE_CODING,
                    REDUNDANT_7BIT_SOURCE_CODING,
                ),
                physical_scenarios=physical_scenarios_for_architecture(
                    args.physical_architecture_profile or "athermal_compensated_mrr"
                ),
                projection_mse_recalibration_threshold=(
                    args.source_coding_projection_mse_threshold
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_constrained_fixed_comparison:
        metrics = run_constrained_comparison(
            ConstrainedComparisonConfig(
                baseline=baseline_config,
                seeds=parse_seeds(args.seeds),
                mlp_dim=args.mlp_dim,
                calibration_samples=args.calibration_samples,
                adaptation_epochs=args.adaptation_epochs,
                adaptation_learning_rate=args.adaptation_learning_rate,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_calibration_cost:
        metrics = run_calibration_cost(
            CalibrationCostConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                calibration_sample_counts=parse_int_tuple(args.calibration_sample_counts),
                recalibration_intervals=parse_int_tuple(args.recalibration_intervals),
                drift_rates=parse_float_tuple(args.drift_rates),
                time_steps=args.time_steps,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_material_nonlinearities:
        metrics = run_material_nonlinearity_sweeps(
            MaterialNonlinearityConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                calibration_samples=args.calibration_samples,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_energy_budget:
        metrics = run_energy_budget(
            EnergyBudgetConfig(
                calibration_summary=args.calibration_summary_path,
                output_dir=args.output_dir / "energy_budget",
                input_dim=args.hidden_dim,
                output_dim=args.mlp_dim,
                inferences_per_drift_step=args.inferences_per_drift_step,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_chained_block_budget:
        metrics = run_chained_block_budget(
            ChainedBlockBudgetConfig(
                calibration_summary=args.calibration_summary_path,
                output_dir=args.output_dir / "chained_block_budget",
                input_dim=args.hidden_dim,
                output_dim=args.mlp_dim,
                inferences_per_drift_step=args.inferences_per_drift_step,
                chain_lengths=parse_int_tuple(args.chain_lengths),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_charge_readout_budget:
        metrics = run_charge_readout_budget(
            ChargeReadoutBudgetConfig(
                calibration_summary=args.calibration_summary_path,
                output_dir=args.output_dir / "charge_readout_budget",
                input_dim=args.hidden_dim,
                output_dim=args.mlp_dim,
                inferences_per_drift_step=args.inferences_per_drift_step,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_material_training:
        metrics = run_material_training(
            MaterialTrainingConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_material_strength_sweep:
        metrics = run_material_strength_sweep(
            MaterialStrengthSweepConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                strengths=parse_float_tuple(args.material_strengths),
                calibration_samples=args.calibration_samples,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_functional_chain:
        metrics = run_functional_chain_simulation(
            FunctionalChainConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                chain_lengths=parse_int_tuple(args.chain_lengths),
                calibration_samples=args.calibration_samples,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_chained_material_training:
        metrics = run_chained_material_training(
            ChainedMaterialTrainingConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                chain_lengths=parse_int_tuple(args.chain_lengths),
                calibration_samples=args.calibration_samples,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_unshared_chained_material_training:
        metrics = run_unshared_chained_material_training(
            UnsharedChainedMaterialTrainingConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                chain_lengths=parse_int_tuple(args.chain_lengths),
                calibration_samples=args.calibration_samples,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_energy_accuracy_pareto_report:
        metrics = run_energy_accuracy_pareto_report(
            EnergyAccuracyParetoConfig(
                chained_budget_rows=args.chained_budget_rows_path,
                shared_training_summary=args.shared_chained_training_summary_path,
                unshared_training_summary=args.unshared_chained_training_summary_path,
                output_dir=args.output_dir / "energy_accuracy_pareto",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_unshared_4block_stress_report:
        metrics = run_unshared_four_block_stress_report(
            UnsharedFourBlockStressConfig(
                chained_budget_rows=args.chained_budget_rows_path,
                unshared_training_summary=args.unshared_chained_training_summary_path,
                output_dir=args.output_dir / "unshared_4block_stress",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_unshared_physical_drift_noise:
        physical_scenarios = (
            physical_scenarios_for_architecture(args.physical_architecture_profile)
            if args.physical_architecture_profile
            else None
        )
        metrics = run_unshared_physical_drift_noise(
            UnsharedPhysicalDriftNoiseConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                calibration_samples=args.calibration_samples,
                physical_scenarios=(
                    physical_scenarios
                    if physical_scenarios is not None
                    else UnsharedPhysicalDriftNoiseConfig.physical_scenarios
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_redundant_source_validation:
        physical_scenarios = physical_scenarios_for_architecture(
            args.physical_architecture_profile or "athermal_compensated_mrr"
        )
        metrics = run_unshared_physical_drift_noise(
            UnsharedPhysicalDriftNoiseConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    calibration_samples=args.calibration_samples,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                seeds=parse_seeds(args.seeds),
                calibration_samples=args.calibration_samples,
                physical_scenarios=physical_scenarios,
                source_coding=REDUNDANT_6BIT_SOURCE_CODING,
                output_dir_name="redundant_source_validation",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_source_coding_calibration_stress:
        physical_scenarios = physical_scenarios_for_architecture(
            args.physical_architecture_profile or "athermal_compensated_mrr"
        )
        metrics = run_source_coding_calibration_stress(
            SourceCodingCalibrationStressConfig(
                transformer=TransformerMLPConfig(
                    data_dir=args.data_dir,
                    dataset=args.dataset,
                    output_dir=args.output_dir / "transformer_mlp_target",
                    hidden_dim=args.hidden_dim,
                    mlp_dim=args.mlp_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    seed=args.seed,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                    synthetic=args.synthetic,
                    adaptation_epochs=args.adaptation_epochs,
                    adaptation_learning_rate=args.adaptation_learning_rate,
                ),
                output_dir=args.output_dir / "source_coding_calibration_stress",
                seeds=parse_seeds(args.seeds),
                chain_length=source_coding_chain_length(args.chain_lengths),
                calibration_sample_counts=parse_int_tuple(args.calibration_sample_counts),
                source_codings=(
                    REDUNDANT_6BIT_SOURCE_CODING,
                    DIRECT_7BIT_SOURCE_CODING,
                    REDUNDANT_7BIT_SOURCE_CODING,
                ),
                physical_scenarios=physical_scenarios,
                projection_mse_recalibration_threshold=(
                    args.source_coding_projection_mse_threshold
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_source_coding_ablation_report:
        metrics = run_source_coding_ablation_report(
            SourceCodingAblationReportConfig(
                source_coding_stress_rows=args.source_coding_ablation_rows_path,
                output_dir=args.output_dir / "source_coding_ablation",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_downstream_recoverability_report:
        metrics = run_downstream_recoverability_report(
            DownstreamRecoverabilityConfig(
                source_coding_stress_rows=args.source_coding_stress_rows_path,
                output_dir=args.output_dir / "downstream_recoverability",
                accuracy_floor=args.downstream_accuracy_floor,
                leaky_budget=args.downstream_leaky_budget,
                leaky_decay=args.downstream_leaky_decay,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_persistent_failure_diagnosis:
        metrics = run_persistent_failure_diagnosis(
            PersistentFailureDiagnosisConfig(
                source_coding_stress_rows=args.source_coding_stress_rows_path,
                output_dir=args.output_dir / "persistent_failure_diagnosis",
                accuracy_floor=args.downstream_accuracy_floor,
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_device_tolerance_target_report:
        metrics = run_device_tolerance_target_report(
            DeviceToleranceTargetConfig(
                physical_summary=args.physical_drift_noise_summary_path,
                output_dir=args.output_dir / "device_tolerance_targets",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_calibration_bridge_report:
        metrics = run_calibration_bridge_report(
            CalibrationBridgeConfig(
                target_csv=args.device_tolerance_targets_path,
                output_dir=args.output_dir / "calibration_bridge",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_architecture_energy_overlay_report:
        metrics = run_architecture_energy_overlay_report(
            ArchitectureEnergyOverlayConfig(
                pareto_rows=args.energy_accuracy_pareto_rows_path,
                physical_rows=args.physical_drift_noise_rows_path,
                output_dir=args.output_dir / "architecture_energy_overlay",
                architecture_profile=(
                    args.physical_architecture_profile or "athermal_compensated_mrr"
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_architecture_energy_stress_report:
        metrics = run_architecture_energy_stress_report(
            ArchitectureEnergyStressConfig(
                pareto_rows=args.energy_accuracy_pareto_rows_path,
                physical_rows=args.physical_drift_noise_rows_path,
                output_dir=args.output_dir / "architecture_energy_stress",
                architecture_profile=(
                    args.physical_architecture_profile or "athermal_compensated_mrr"
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_architecture_converter_stress_report:
        metrics = run_architecture_converter_stress_report(
            ArchitectureConverterStressConfig(
                pareto_rows=args.energy_accuracy_pareto_rows_path,
                physical_rows=args.physical_drift_noise_rows_path,
                output_dir=args.output_dir / "architecture_converter_stress",
                architecture_profile=(
                    args.physical_architecture_profile or "athermal_compensated_mrr"
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_architecture_converter_target_report:
        metrics = run_architecture_converter_target_report(
            ArchitectureConverterTargetConfig(
                converter_stress_rows=args.architecture_converter_stress_rows_path,
                output_dir=args.output_dir / "architecture_converter_target",
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_architecture_converter_profile_report:
        metrics = run_architecture_converter_profile_report(
            ArchitectureConverterProfileConfig(
                pareto_rows=args.energy_accuracy_pareto_rows_path,
                physical_rows=args.physical_drift_noise_rows_path,
                output_dir=args.output_dir / "architecture_converter_profile",
                architecture_profile=(
                    args.physical_architecture_profile or "athermal_compensated_mrr"
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    if args.run_architecture_dac_coding_sweep_report:
        metrics = run_architecture_dac_coding_sweep_report(
            ArchitectureDacCodingSweepConfig(
                pareto_rows=args.energy_accuracy_pareto_rows_path,
                physical_rows=args.physical_drift_noise_rows_path,
                output_dir=args.output_dir / "architecture_dac_coding_sweep",
                architecture_profile=(
                    args.physical_architecture_profile or "athermal_compensated_mrr"
                ),
            )
        )
        print(json.dumps(metrics, indent=2))
        return 0

    payload = {
        "output_dir": str(args.output_dir),
        "dry_run": args.dry_run,
        "stages": planned_stages(),
    }

    print(json.dumps(payload, indent=2))

    if not args.dry_run:
        parser.exit(
            2,
            "Implementation not wired yet. Run with --dry-run for the scaffold check.\n",
        )

    return 0
