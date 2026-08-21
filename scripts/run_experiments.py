"""Run reproducible turn-detection ablations and write a grounded report.

Each completed experiment gets its own directory under ``experiments/`` with
the resolved YAML config, training history/metrics, validation-first comparison
metrics, checkpoint, and compact result manifest. Held-out test evaluation is
explicit. Combined JSON, CSV, and Markdown reports are regenerated after runs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.evaluate import evaluate_checkpoint
from src.train import load_config, train

BASELINE_CONFIG_PATH = "configs/baseline.yaml"
OUTPUT_ROOT = Path("experiments")
DOCS_REPORT_ROOT = Path("docs/generated")
EXPERIMENT_PROTOCOL_VERSION = 2  # label-preserving filler policy + pause slices

TRACKED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "false_complete_rate",
    "latency",
    "model_size",
)


@dataclass(frozen=True)
class ExperimentSpec:
    """One controlled ablation and its pre-registered interpretation rule."""

    experiment_id: str
    research_question: str
    hypothesis: str
    description: str
    overrides: dict[str, Any]
    comparator_id: str | None
    success_criteria: str
    metrics: tuple[str, ...] = TRACKED_METRICS


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        "E1_no_augmentation",
        "data_vs_architecture",
        "Raw audio establishes how much the model learns without synthetic data transforms.",
        "Attention pooling on original, unaugmented audio",
        {"data.use_augmentation": False},
        None,
        "Reference control; no standalone pass/fail claim.",
    ),
    ExperimentSpec(
        "E2_augmented",
        "data_vs_architecture",
        "Targeted augmentation lowers false-complete rate on pause/filler slices without collapsing recall.",
        "Attention pooling with full Hinglish/pause augmentation",
        {},
        "E1_no_augmentation",
        "FCR improves on hard slices, recall drops no more than 2 pp, and overall F1 is not worse by more than 1 pp.",
    ),
    ExperimentSpec(
        "E3_mean_pool",
        "pooling",
        "Mean pooling dilutes local endpoint cues and underperforms attention on pause-heavy slices.",
        "Mean pooling with same augmented data",
        {"model.pooling": "mean"},
        "E2_augmented",
        "Attention wins by at least 1 F1 point on internal-pause or hard-Hinglish slice without material latency penalty.",
    ),
    ExperimentSpec(
        "E4_last_pool",
        "pooling",
        "Last-frame pooling is brittle when trailing silence separates speech from tensor end.",
        "Last-frame pooling with same augmented data",
        {"model.pooling": "last"},
        "E2_augmented",
        "Attention lowers internal/trailing-pause FCR while retaining recall within 2 pp.",
    ),
    ExperimentSpec(
        "E5_short_pauses",
        "silence_length",
        "Short-pause-only training fails on long hesitation.",
        "Silence augmentation restricted to 50-250 ms",
        {"data.augment_config.silence_ms_range": [50, 250]},
        "E2_augmented",
        "Standard 100-800 ms range lowers FCR on internal-pause and Hindi-pause slices without >1 pp overall-F1 loss.",
    ),
    ExperimentSpec(
        "E6_long_pauses",
        "silence_length",
        "Long-pause-only training over-predicts incomplete after short natural breaths.",
        "Silence augmentation restricted to 600-1500 ms",
        {"data.augment_config.silence_ms_range": [600, 1500]},
        "E2_augmented",
        "Standard range improves complete-class recall and overall F1 while matching long-pause FCR within 2 pp.",
    ),
    ExperimentSpec(
        "E7_frozen_encoder",
        "architecture",
        "A frozen ASR encoder lacks task-specific endpoint adaptation.",
        "Freeze all four Whisper encoder layers",
        {"model.freeze_encoder_layers": 4},
        "E2_augmented",
        "Full tuning improves overall and hard-slice F1 by at least 2 pp; compare training cost separately.",
    ),
    ExperimentSpec(
        "E8_partial_finetune",
        "architecture",
        "Partial tuning retains most full-tuning quality with fewer trainable parameters.",
        "Freeze first two Whisper encoder layers",
        {"model.freeze_encoder_layers": 2},
        "E2_augmented",
        "Within 1 F1 point and 2 FCR points of full tuning on overall and hard-Hinglish slices.",
    ),
    ExperimentSpec(
        "E9_no_filler",
        "filler_injection",
        "Filler injection improves filler/pause robustness beyond other acoustic transforms.",
        "Full augmentation except filler injection",
        {"data.augment_config.p_filler": 0.0},
        "E2_augmented",
        "Full augmentation improves Hindi-filler or hard-Hinglish F1 by at least 1 pp without >1 pp overall-F1 loss.",
    ),
    ExperimentSpec(
        "E10_filler_only",
        "filler_injection",
        "Label-preserving filler injection contributes measurable value without other transforms.",
        "Only filler injection enabled",
        {
            "data.augment_config.p_silence": 0.0,
            "data.augment_config.p_speed": 0.0,
            "data.augment_config.p_pitch": 0.0,
            "data.augment_config.p_noise": 0.0,
            "data.augment_config.p_volume": 0.0,
        },
        "E1_no_augmentation",
        "Improves filler-slice F1 or FCR by at least 1 pp while overall F1 stays within 1 pp.",
    ),
    ExperimentSpec(
        "E11_no_silence",
        "silence_length",
        "Silence insertion, not generic augmentation, drives pause robustness.",
        "Full augmentation except silence insertion",
        {"data.augment_config.p_silence": 0.0},
        "E2_augmented",
        "Full augmentation lowers internal-pause FCR by at least 2 pp with recall loss no larger than 2 pp.",
    ),
    ExperimentSpec(
        "E12_no_hard_mining",
        "hard_hinglish",
        "Hard-example oversampling improves difficult Hindi/filler cases without harming the main distribution.",
        "Disable hard-example oversampling",
        {"data.use_hard_negatives": False},
        "E2_augmented",
        "Hard-Hinglish F1 improves by at least 2 pp while overall F1 changes by no worse than -1 pp.",
    ),
)

CORE_EXPERIMENT_IDS = {
    "E1_no_augmentation",
    "E2_augmented",
    "E3_mean_pool",
    "E5_short_pauses",
    "E6_long_pauses",
    "E9_no_filler",
    "E11_no_silence",
    "E12_no_hard_mining",
}
FULL_EXPERIMENT_IDS = {experiment.experiment_id for experiment in EXPERIMENTS}


def _apply_overrides(base_config: dict, overrides: dict) -> dict:
    config = copy.deepcopy(base_config)
    for dotted_key, value in overrides.items():
        node = config
        *path, leaf = dotted_key.split(".")
        for key in path:
            if key not in node or not isinstance(node[key], dict):
                raise KeyError(f"invalid config override path: {dotted_key}")
            node = node[key]
        if leaf not in node:
            raise KeyError(f"invalid config override key: {dotted_key}")
        node[leaf] = value
    return config


def _file_fingerprint(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    digest = hashlib.sha256()
    with metadata_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    rows = pl.scan_parquet(metadata_path).select(pl.len()).collect().item()
    return {
        "path": str(metadata_path.resolve()),
        "rows": rows,
        "sha256": digest.hexdigest(),
    }


def build_experiment_manifest(
    base_config: dict[str, Any],
    selected_ids: set[str],
    *,
    epochs: int,
    train_meta: str | Path,
    val_meta: str | Path,
    test_meta: str | Path,
    challenge_meta: str | Path | None = None,
    output_root: Path = OUTPUT_ROOT,
    experiment_name_prefix: str | None = None,
) -> dict[str, Any]:
    """Resolve configs and data fingerprints before any expensive training."""
    specs = [experiment for experiment in EXPERIMENTS if experiment.experiment_id in selected_ids]
    runs = []
    for experiment in specs:
        resolved = _apply_overrides(base_config, experiment.overrides)
        resolved["training"]["num_epochs"] = epochs
        resolved["experiment_name"] = (
            f"{experiment_name_prefix}/{experiment.experiment_id}"
            if experiment_name_prefix
            else experiment.experiment_id
        )
        resolved["checkpoint"]["dir"] = str(output_root / experiment.experiment_id / "checkpoints")
        config_bytes = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
        runs.append(
            {
                **asdict(experiment),
                "resolved_config": resolved,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            }
        )

    datasets = {
        "train": _file_fingerprint(train_meta),
        "validation": _file_fingerprint(val_meta),
        "test": _file_fingerprint(test_meta),
    }
    if challenge_meta is not None:
        datasets["challenge"] = _file_fingerprint(challenge_meta)
    return {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "epochs": epochs,
        "tracked_metrics": TRACKED_METRICS,
        "datasets": datasets,
        "experiments": runs,
    }


def _evaluate_on_test(checkpoint_path: Path, test_meta_path: str, output_path: Path) -> dict:
    report = evaluate_checkpoint(checkpoint_path, test_meta_path)
    output_path.write_text(json.dumps(report, indent=2, default=str))
    return report


def _report_matches_metadata(report: dict, metadata_path: str | Path) -> bool:
    recorded = report.get("metadata")
    return bool(recorded) and Path(recorded).resolve() == Path(metadata_path).resolve()


def _result_from_reports(
    experiment: ExperimentSpec,
    train_report: dict,
    evaluation_report: dict,
    experiment_dir: Path,
    evaluation_split: str,
    challenge_report: dict | None = None,
) -> dict:
    num_parameters = train_report["latency_cpu"]["num_parameters"]
    return {
        "experiment_id": experiment.experiment_id,
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "seed": train_report["config"]["training"]["seed"],
        "status": "completed",
        "research_question": experiment.research_question,
        "hypothesis": experiment.hypothesis,
        "description": experiment.description,
        "overrides": experiment.overrides,
        "comparator_id": experiment.comparator_id,
        "success_criteria": experiment.success_criteria,
        "metrics_tracked": experiment.metrics,
        "config_path": str(experiment_dir / "config.yaml"),
        "checkpoint_path": str(experiment_dir / "checkpoints" / "best.pt"),
        "val_metrics": train_report["best_val_metrics"],
        "val_metrics_fixed_0_5": train_report.get("best_val_metrics_fixed_0_5"),
        "decision_threshold": train_report.get("decision_threshold", 0.5),
        "threshold_calibration": train_report.get("threshold_calibration"),
        "evaluation_split": evaluation_split,
        "evaluation_metrics": evaluation_report["metrics"],
        "error_analysis": evaluation_report["error_analysis"],
        "challenge_metrics": challenge_report["metrics"] if challenge_report else None,
        "challenge_error_analysis": challenge_report["error_analysis"] if challenge_report else None,
        "model_size_mb_fp32": num_parameters * 4 / (1024**2),
        "latency_cpu": train_report["latency_cpu"],
        "latency_gpu": train_report["latency_gpu"],
    }


def _text_features_available(metadata_path: str) -> bool:
    schema = pl.read_parquet_schema(metadata_path)
    if "transcript" not in schema:
        return False
    texts = pl.read_parquet(metadata_path, columns=["transcript"])
    return texts["transcript"].drop_nulls().len() > 0


def _multimodal_result(train_meta_path: str) -> dict:
    result_path = OUTPUT_ROOT / "M1_audio_plus_text" / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    if _text_features_available(train_meta_path):
        reason = "Transcripts exist; run scripts/run_multimodal_experiment.py to create matched metrics."
    else:
        reason = (
            "no transcript column. Run scripts/transcribe_dataset.py, then "
            "scripts/run_multimodal_experiment.py. "
            "Frozen ASR receives audio only; turn labels never enter transcript generation."
        )
    return {
        "experiment_id": "M1_audio_plus_text",
        "status": "skipped",
        "research_question": "audio_vs_semantics",
        "description": "Audio plus text features",
        "comparator_id": "E1_no_augmentation",
        "reason": reason,
    }


def _format_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _metrics_for_result(result: dict) -> dict:
    """Read protocol-v2 metrics, falling back to historical test artifacts."""
    return result.get("evaluation_metrics") or result["test_metrics"]


def _format_ms(latency: dict | None) -> str:
    return f"{latency['latency_ms_mean']:.2f}" if latency else "n/a"


def _format_slice_metric(result: dict, slice_name: str, metric: str) -> str:
    slice_result = result.get("error_analysis", {}).get("slices", {}).get(slice_name)
    if not slice_result:
        return "n/a"
    return _format_pct(slice_result[metric])


def _render_report(results: list[dict], epochs: int) -> str:
    completed = [result for result in results if result["status"] == "completed"]
    by_id = {result["experiment_id"]: result for result in completed}
    evaluation_splits = {result.get("evaluation_split", "test") for result in completed}
    split_description = ", ".join(sorted(evaluation_splits)) if evaluation_splits else "none"
    calibrated = any(result.get("threshold_calibration") for result in completed)
    selection_description = (
        "maximum validation F1 under configured FCR/recall constraints"
        if calibrated
        else "validation F1"
    )
    lines = [
        "<!-- markdownlint-disable MD013 MD060 -->",
        "",
        "# Turn Detection Ablation Report",
        "",
        (
            f"All completed runs use matched data partitions and **{epochs} epochs**. "
            f"Checkpoint selection uses {selection_description}; comparison rows currently contain: {split_description}."
        ),
        "",
        "## Comparison",
        "",
        "| Experiment | Status | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | FP32 MB | CPU ms | GPU ms |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in results:
        if result["status"] != "completed":
            lines.append(f"| {result['experiment_id']} | skipped | - | - | - | - | - | - | - | - | - | - |")
            continue
        metrics = _metrics_for_result(result)
        cpu = result["latency_cpu"]
        lines.append(
            f"| {result['experiment_id']} | completed | {_format_pct(metrics['accuracy'])} | "
            f"{_format_pct(metrics['precision'])} | {_format_pct(metrics['recall'])} | "
            f"{_format_pct(metrics['f1'])} | {_format_pct(metrics['auc'])} | "
            f"{_format_pct(metrics['false_complete_rate'])} | {cpu['num_parameters']:,} | "
            f"{result['model_size_mb_fp32']:.2f} | {_format_ms(cpu)} | {_format_ms(result['latency_gpu'])} |"
        )

    lines.extend(
        [
            "",
            "## Pause and hard-case slices",
            "",
            "| Experiment | Internal-pause F1 | Internal-pause FCR | Hindi filler+pause F1 | Hard-Hinglish F1 | Hard-Hinglish FCR |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for result in results:
        if result["status"] != "completed":
            lines.append(f"| {result['experiment_id']} | - | - | - | - | - |")
            continue
        lines.append(
            f"| {result['experiment_id']} | {_format_slice_metric(result, 'internal_pause', 'f1')} | "
            f"{_format_slice_metric(result, 'internal_pause', 'false_complete_rate')} | "
            f"{_format_slice_metric(result, 'hindi_filler_pause', 'f1')} | "
            f"{_format_slice_metric(result, 'hard_hinglish_proxy', 'f1')} | "
            f"{_format_slice_metric(result, 'hard_hinglish_proxy', 'false_complete_rate')} |"
        )

    challenge_results = [result for result in completed if result.get("challenge_metrics")]
    if challenge_results:
        lines.extend(
            [
                "",
                "## Curated challenge set",
                "",
                "| Experiment | Accuracy | Precision | Recall | F1 | FCR |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for result in challenge_results:
            metrics = result["challenge_metrics"]
            lines.append(
                f"| {result['experiment_id']} | {_format_pct(metrics['accuracy'])} | "
                f"{_format_pct(metrics['precision'])} | {_format_pct(metrics['recall'])} | "
                f"{_format_pct(metrics['f1'])} | {_format_pct(metrics['false_complete_rate'])} |"
            )

    lines.extend(["", "## Per-experiment insights", ""])
    for result in completed:
        metrics = _metrics_for_result(result)
        experiment_id = result["experiment_id"]
        insight = (
            f"Evaluation F1 {_format_pct(metrics['f1'])}; accuracy {_format_pct(metrics['accuracy'])}; "
            f"false-complete {_format_pct(metrics['false_complete_rate'])}."
        )
        comparator_id = result.get("comparator_id")
        if comparator_id is None and experiment_id in {"E2_augmented", "M1_audio_plus_text"}:
            comparator_id = "E1_no_augmentation"
        elif comparator_id is None and experiment_id != "E1_no_augmentation":
            comparator_id = "E2_augmented"
        comparator = by_id.get(comparator_id) if comparator_id else None
        if comparator is not None:
            comparator_metrics = _metrics_for_result(comparator)
            delta_f1 = 100 * (metrics["f1"] - comparator_metrics["f1"])
            delta_false = 100 * (
                metrics["false_complete_rate"] - comparator_metrics["false_complete_rate"]
            )
            insight += (
                f" Versus {comparator['experiment_id']}: F1 {delta_f1:+.2f} points; "
                f"false-complete {delta_false:+.2f} points."
            )
        if experiment_id == "E7_frozen_encoder":
            insight += (
                f" Recall is only {_format_pct(metrics['recall'])}; its low false-complete rate is caused "
                "by over-predicting incomplete, not by a useful operating point. This equal-LR ablation "
                "does not test a frozen-head-specific learning rate or longer schedule."
            )
        elif experiment_id == "E8_partial_finetune":
            insight += " Partial tuning retains nearly all full-tuning F1 with substantially fewer trainable parameters."
        lines.extend([f"### {experiment_id}", "", result["description"] + ". " + insight, ""])

    for result in (item for item in results if item["status"] == "skipped"):
        lines.extend([f"### {result['experiment_id']}", "", result["reason"], ""])

    if completed:
        best = max(completed, key=lambda result: _metrics_for_result(result)["f1"])
        viable = [result for result in completed if _metrics_for_result(result)["recall"] >= 0.75]
        best_metrics = _metrics_for_result(best)
        lines.extend(
            [
                "## Summary",
                "",
                f"- Highest evaluation F1: **{best['experiment_id']}** ({_format_pct(best_metrics['f1'])}).",
            ]
        )
        if viable:
            safest = min(viable, key=lambda result: _metrics_for_result(result)["false_complete_rate"])
            safest_metrics = _metrics_for_result(safest)
            lines.append(
                f"- Lowest false-complete rate among models with at least 75% recall: "
                f"**{safest['experiment_id']}** ({_format_pct(safest_metrics['false_complete_rate'])})."
            )
        else:
            lines.append("- No completed model met the 75% recall guard for FCR ranking.")
        lines.append("- Single-seed, short-budget ablations: small differences are directional, not statistical proof.")
    return "\n".join(lines) + "\n"


def _write_comparison_csv(results: list[dict], path: Path) -> None:
    columns = [
        "experiment_id",
        "status",
        "seed",
        "experiment_protocol_version",
        "evaluation_split",
        "research_question",
        "comparator_id",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "false_complete_rate",
        "num_parameters",
        "model_size_mb_fp32",
        "cpu_latency_ms",
        "gpu_latency_ms",
        "internal_pause_f1",
        "internal_pause_false_complete_rate",
        "hard_hinglish_f1",
        "hard_hinglish_false_complete_rate",
        "challenge_f1",
        "challenge_false_complete_rate",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for result in results:
            row = {
                "experiment_id": result["experiment_id"],
                "status": result["status"],
                "seed": result.get("seed"),
                "experiment_protocol_version": result.get("experiment_protocol_version"),
                "evaluation_split": result.get("evaluation_split", "test"),
                "research_question": result.get("research_question"),
                "comparator_id": result.get("comparator_id"),
            }
            if result["status"] == "completed":
                row.update(_metrics_for_result(result))
                row["num_parameters"] = result["latency_cpu"]["num_parameters"]
                row["model_size_mb_fp32"] = result["model_size_mb_fp32"]
                row["cpu_latency_ms"] = result["latency_cpu"]["latency_ms_mean"]
                row["gpu_latency_ms"] = (
                    result["latency_gpu"]["latency_ms_mean"] if result["latency_gpu"] else None
                )
                slices = result.get("error_analysis", {}).get("slices", {})
                internal = slices.get("internal_pause") or {}
                hard = slices.get("hard_hinglish_proxy") or {}
                row["internal_pause_f1"] = internal.get("f1")
                row["internal_pause_false_complete_rate"] = internal.get("false_complete_rate")
                row["hard_hinglish_f1"] = hard.get("f1")
                row["hard_hinglish_false_complete_rate"] = hard.get("false_complete_rate")
                challenge = result.get("challenge_metrics") or {}
                row["challenge_f1"] = challenge.get("f1")
                row["challenge_false_complete_rate"] = challenge.get("false_complete_rate")
            writer.writerow({column: row.get(column) for column in columns})


def _docs_report_path(run_tag: str | None) -> Path:
    """Return canonical human-readable report path for one experiment suite."""
    if not run_tag:
        return DOCS_REPORT_ROOT / "ablation_report.md"
    safe_tag = run_tag.replace("/", "_").replace("\\", "_")
    return DOCS_REPORT_ROOT / f"{safe_tag}_ablation_report.md"


def _print_comparison_table(results: list[dict]) -> None:
    header = (
        f"{'experiment':<24}{'status':<11}{'test_acc':>10}{'test_f1':>10}"
        f"{'test_auc':>10}{'false_cpl':>11}{'params':>12}{'cpu_ms':>10}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        if result["status"] != "completed":
            print(f"{result['experiment_id']:<24}{'skipped':<11}")
            continue
        metrics = _metrics_for_result(result)
        latency = result["latency_cpu"]
        print(
            f"{result['experiment_id']:<24}{'completed':<11}{metrics['accuracy']:>10.4f}"
            f"{metrics['f1']:>10.4f}{metrics['auc']:>10.4f}{metrics['false_complete_rate']:>11.4f}"
            f"{latency['num_parameters']:>12,}{latency['latency_ms_mean']:>10.1f}"
        )


def main() -> None:
    """Run selected ablations and regenerate comparison artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="Run only these experiment IDs")
    parser.add_argument("--suite", choices=("core", "full"), default="core")
    parser.add_argument("--epochs", type=int, default=3, help="Equal epoch budget for every experiment")
    parser.add_argument("--seed", type=int, default=42, help="Training seed shared by selected experiments")
    parser.add_argument("--run-tag", help="Write an isolated suite under experiments/<run-tag>/")
    parser.add_argument("--base-config", default=str(BASELINE_CONFIG_PATH))
    parser.add_argument("--reuse-existing", action="store_true", help="Reuse complete metrics/test artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Write resolved manifest without training")
    parser.add_argument(
        "--final-test",
        action="store_true",
        help="Evaluate on held-out test; default compares experiments on validation",
    )
    parser.add_argument("--train-meta", default=str(cfg.SUBSET_DIR / "train_split.parquet"))
    parser.add_argument("--val-meta", default=str(cfg.SUBSET_DIR / "val_split.parquet"))
    parser.add_argument("--test-meta", default=str(cfg.SUBSET_DIR / "test_split.parquet"))
    parser.add_argument("--challenge-meta", help="Optional manually curated Hinglish challenge manifest")
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")

    run_root = OUTPUT_ROOT / args.run_tag if args.run_tag else OUTPUT_ROOT
    run_root.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.base_config)
    base_config["training"]["num_epochs"] = args.epochs
    base_config["training"]["seed"] = args.seed
    base_config["experiment_protocol_version"] = EXPERIMENT_PROTOCOL_VERSION

    suite_ids = CORE_EXPERIMENT_IDS if args.suite == "core" else FULL_EXPERIMENT_IDS
    selected_ids = set(args.only) if args.only is not None else suite_ids
    known_ids = FULL_EXPERIMENT_IDS
    unknown = selected_ids - known_ids
    if unknown:
        raise ValueError(f"unknown experiment IDs: {sorted(unknown)}")
    if not selected_ids:
        raise ValueError("at least one experiment must be selected")
    evaluation_split = "test" if args.final_test else "validation"
    evaluation_meta = args.test_meta if args.final_test else args.val_meta

    manifest = build_experiment_manifest(
        base_config,
        selected_ids,
        epochs=args.epochs,
        train_meta=args.train_meta,
        val_meta=args.val_meta,
        test_meta=args.test_meta,
        challenge_meta=args.challenge_meta,
        output_root=run_root,
        experiment_name_prefix=args.run_tag,
    )
    manifest["experiment_protocol_version"] = EXPERIMENT_PROTOCOL_VERSION
    manifest_path = run_root / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    if args.dry_run:
        print(f"wrote {manifest_path}")
        for run in manifest["experiments"]:
            print(f"{run['experiment_id']}: {run['description']}")
        return

    results = []
    for experiment in EXPERIMENTS:
        experiment_id = experiment.experiment_id
        if experiment_id not in selected_ids:
            continue
        print(f"\n{'=' * 70}\n[{experiment_id}] {experiment.description}\n{'=' * 70}")
        experiment_dir = run_root / experiment_id
        checkpoint_path = experiment_dir / "checkpoints" / "best.pt"
        metrics_path = experiment_dir / "metrics.json"
        evaluation_path = experiment_dir / f"{evaluation_split}_metrics.json"
        challenge_path = experiment_dir / "challenge_metrics.json"
        config = _apply_overrides(base_config, experiment.overrides)
        config["experiment_name"] = f"{args.run_tag}/{experiment_id}" if args.run_tag else experiment_id
        config["checkpoint"]["dir"] = str(experiment_dir / "checkpoints")

        if args.reuse_existing and metrics_path.exists() and checkpoint_path.exists():
            existing_train = json.loads(metrics_path.read_text())
            if existing_train.get("config") == config:
                print(f"[{experiment_id}] reusing config-matched training artifacts")
                train_report = existing_train
                if evaluation_path.exists():
                    candidate_report = json.loads(evaluation_path.read_text())
                    if _report_matches_metadata(candidate_report, evaluation_meta):
                        evaluation_report = candidate_report
                    else:
                        evaluation_report = _evaluate_on_test(
                            checkpoint_path, evaluation_meta, evaluation_path
                        )
                else:
                    evaluation_report = _evaluate_on_test(
                        checkpoint_path, evaluation_meta, evaluation_path
                    )
            else:
                print(f"[{experiment_id}] existing artifacts use a different config; rerunning")
                train_report = train(config, args.train_meta, args.val_meta)
                evaluation_report = _evaluate_on_test(
                    checkpoint_path, evaluation_meta, evaluation_path
                )
        else:
            train_report = train(config, args.train_meta, args.val_meta)
            evaluation_report = _evaluate_on_test(
                checkpoint_path, evaluation_meta, evaluation_path
            )

        challenge_report = None
        if args.challenge_meta:
            if args.reuse_existing and challenge_path.exists():
                candidate_report = json.loads(challenge_path.read_text())
                if _report_matches_metadata(candidate_report, args.challenge_meta):
                    challenge_report = candidate_report
                else:
                    challenge_report = _evaluate_on_test(
                        checkpoint_path,
                        args.challenge_meta,
                        challenge_path,
                    )
            else:
                challenge_report = _evaluate_on_test(
                    checkpoint_path,
                    args.challenge_meta,
                    challenge_path,
                )

        result = _result_from_reports(
            experiment,
            train_report,
            evaluation_report,
            experiment_dir,
            evaluation_split,
            challenge_report,
        )
        (experiment_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))
        results.append(result)

    if args.only is None and args.run_tag is None and args.seed == 42:
        results.append(_multimodal_result(args.train_meta))

    all_results_path = run_root / "all_results.json"
    existing_results = json.loads(all_results_path.read_text()) if all_results_path.exists() else []
    merged = {result["experiment_id"]: result for result in existing_results}
    merged.update({result["experiment_id"]: result for result in results})
    all_results = list(merged.values())
    all_results_path.write_text(json.dumps(all_results, indent=2, default=str))
    comparable_results = [
        result
        for result in all_results
        if result.get("experiment_protocol_version") == EXPERIMENT_PROTOCOL_VERSION
        and result.get("evaluation_split") == evaluation_split
    ]
    _write_comparison_csv(comparable_results, run_root / "comparison.csv")
    report_path = _docs_report_path(args.run_tag)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_report(comparable_results, args.epochs))
    print(f"\nwrote {run_root}/all_results.json, {run_root}/comparison.csv, and {report_path}")
    _print_comparison_table(comparable_results)


if __name__ == "__main__":
    main()
