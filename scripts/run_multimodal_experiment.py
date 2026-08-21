"""Train matched audio+text ablation and report deltas against audio-only E1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.run_experiments import (
    DOCS_REPORT_ROOT,
    EXPERIMENT_PROTOCOL_VERSION,
    _render_report,
    _write_comparison_csv,
)
from src.evaluate import evaluate_checkpoint
from src.train import load_config, train


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _delta(value: float, reference: float) -> str:
    return f"{100 * (value - reference):+.2f} pp"


def _render_multimodal_report(
    baseline: dict,
    multimodal: dict,
    transcript_summary: dict | None,
    live_latency: dict | None,
) -> str:
    base_metrics = baseline["test_metrics"]
    multi_metrics = multimodal["test_metrics"]
    lines = [
        "<!-- markdownlint-disable MD013 MD060 -->",
        "",
        "# Audio-only vs Audio+Text",
        "",
        (
            "Controlled comparison: same unaugmented train/validation/test splits, seed 42, "
            "optimizer, attention pooling, and 3-epoch budget. M1 adds frozen Whisper-tiny "
            "ASR transcripts, a learned 64-dimensional token embedding, and feature concatenation."
        ),
        "",
        "| Model | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | GPU classifier ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in (baseline, multimodal):
        metrics = result["test_metrics"]
        latency = result.get("latency_gpu")
        latency_text = f"{latency['latency_ms_mean']:.2f}" if latency else "n/a"
        lines.append(
            f"| {result['experiment_id']} | {_pct(metrics['accuracy'])} | "
            f"{_pct(metrics['precision'])} | {_pct(metrics['recall'])} | "
            f"{_pct(metrics['f1'])} | {_pct(metrics['auc'])} | "
            f"{_pct(metrics['false_complete_rate'])} | "
            f"{result['latency_cpu']['num_parameters']:,} | {latency_text} |"
        )
    lines.extend(
        [
            "",
            "## Overall difference",
            "",
            (
                f"M1 versus E1: accuracy {_delta(multi_metrics['accuracy'], base_metrics['accuracy'])}, "
                f"F1 {_delta(multi_metrics['f1'], base_metrics['f1'])}, AUC "
                f"{_delta(multi_metrics['auc'], base_metrics['auc'])}, and false-complete "
                f"{_delta(multi_metrics['false_complete_rate'], base_metrics['false_complete_rate'])}."
            ),
            "",
            "## Hindi / Hinglish-proxy slices",
            "",
            (
                "Dataset has Hindi language tags but no verified code-switch/Hinglish annotation. "
                "These Hindi and Hindi+filler slices are proxies, not a claimed Hinglish benchmark."
            ),
            "",
            "| Slice | n | E1 F1 | M1 F1 | F1 delta | E1 false-complete | M1 false-complete | Delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    base_slices = baseline["error_analysis"]["slices"]
    multi_slices = multimodal["error_analysis"]["slices"]
    for key, label in (
        ("hindi", "Hindi"),
        ("hindi_midfiller", "Hindi + mid-filler"),
        ("hindi_endfiller", "Hindi + end-filler"),
    ):
        base_slice = base_slices[key]
        multi_slice = multi_slices[key]
        has_both_classes = multi_slice["n_complete"] > 0 and multi_slice["n_incomplete"] > 0
        base_f1 = _pct(base_slice["f1"]) if has_both_classes else "n/a"
        multi_f1 = _pct(multi_slice["f1"]) if has_both_classes else "n/a"
        f1_delta = _delta(multi_slice["f1"], base_slice["f1"]) if has_both_classes else "n/a"
        lines.append(
            f"| {label} | {multi_slice['n']} | {base_f1} | {multi_f1} | {f1_delta} | "
            f"{_pct(base_slice['false_complete_rate'])} | "
            f"{_pct(multi_slice['false_complete_rate'])} | "
            f"{_delta(multi_slice['false_complete_rate'], base_slice['false_complete_rate'])} |"
        )
    lines.extend(
        [
            "",
            "`n/a` means slice contains only one ground-truth class; binary F1 is not meaningful there.",
            "",
            "## Latency scope",
            "",
            (
                "Classifier latency excludes transcript generation. Cached text is appropriate for this "
                "offline ablation; live deployment must add autoregressive ASR latency and therefore loses "
                "the tiny audio-only model's streaming-speed advantage."
            ),
        ]
    )
    if transcript_summary:
        total_rows = sum(item["rows"] for item in transcript_summary["splits"])
        empty_rows = sum(item["empty_transcripts"] for item in transcript_summary["splits"])
        generation_seconds = sum(item["generation_seconds"] for item in transcript_summary["splits"])
        newly_transcribed = sum(item["newly_transcribed"] for item in transcript_summary["splits"])
        amortized_ms = 1000 * generation_seconds / max(1, newly_transcribed)
        lines.extend(
            [
                "",
                (
                    f"Batched ASR cache run: {total_rows:,} rows; {empty_rows:,} empty transcripts; "
                    f"{amortized_ms:.2f} ms/clip amortized generation time on "
                    f"{transcript_summary['device']} (not single-request latency)."
                ),
            ]
        )
    if live_latency:
        audio = live_latency["audio_only"]
        fused = live_latency["audio_plus_text"]
        lines.extend(
            [
                "",
                "| End-to-end batch-1 path | Mean | p50 | p95 |",
                "| --- | ---: | ---: | ---: |",
                f"| Audio-only | {audio['mean_ms']:.2f} ms | {audio['p50_ms']:.2f} ms | {audio['p95_ms']:.2f} ms |",
                f"| Audio+text | {fused['mean_ms']:.2f} ms | {fused['p50_ms']:.2f} ms | {fused['p95_ms']:.2f} ms |",
                "",
                (
                    f"Warm live M1 is {fused['mean_ms'] / audio['mean_ms']:.1f}x slower across "
                    f"{fused['n']} held-out clips. Model loading is excluded; feature extraction, "
                    "ASR generation, tokenization, and classification are included."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Train, evaluate, and report the configured audio-text experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/multimodal.yaml")
    parser.add_argument("--baseline-result", type=Path, default=Path("experiments/E1_no_augmentation/result.json"))
    parser.add_argument("--train-meta", type=Path, default=Path("data/subset/transcribed/train_split.parquet"))
    parser.add_argument("--val-meta", type=Path, default=Path("data/subset/transcribed/val_split.parquet"))
    parser.add_argument("--test-meta", type=Path, default=Path("data/subset/transcribed/test_split.parquet"))
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    for path in (args.baseline_result, args.train_meta, args.val_meta, args.test_meta):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (args.train_meta, args.val_meta, args.test_meta):
        schema = pl.read_parquet_schema(path)
        if "transcript" not in schema:
            raise ValueError(f"missing transcript column: {path}")

    config = load_config(args.config)
    experiment_dir = Path("experiments") / config["experiment_name"]
    checkpoint = experiment_dir / "checkpoints" / "best.pt"
    metrics_path = experiment_dir / "metrics.json"
    if args.reuse_existing:
        if not checkpoint.is_file() or not metrics_path.is_file():
            raise FileNotFoundError("--reuse-existing requires completed M1 checkpoint and metrics")
        train_report = json.loads(metrics_path.read_text())
    else:
        train_report = train(config, args.train_meta, args.val_meta)
    multimodal_test = evaluate_checkpoint(checkpoint, args.test_meta)
    (experiment_dir / "test_metrics.json").write_text(json.dumps(multimodal_test, indent=2, default=str))

    baseline = json.loads(args.baseline_result.read_text())
    baseline_test = evaluate_checkpoint(Path(baseline["checkpoint_path"]), args.test_meta)
    baseline["research_question"] = "audio_vs_semantics"
    baseline["comparator_id"] = None
    baseline["experiment_protocol_version"] = EXPERIMENT_PROTOCOL_VERSION
    baseline["evaluation_split"] = "test"
    baseline["evaluation_metrics"] = baseline_test["metrics"]
    baseline["test_metrics"] = baseline_test["metrics"]
    baseline["error_analysis"] = baseline_test["error_analysis"]
    multimodal = {
        "experiment_id": config["experiment_name"],
        "experiment_protocol_version": EXPERIMENT_PROTOCOL_VERSION,
        "status": "completed",
        "research_question": "audio_vs_semantics",
        "description": "Whisper audio embedding plus cached Whisper transcript embedding",
        "overrides": {"model.multimodal": True, "data.use_augmentation": False},
        "comparator_id": "E1_no_augmentation",
        "success_criteria": (
            "Improve Hindi/hard-Hinglish F1 or FCR by at least 2 pp with recall preserved, "
            "and justify end-to-end ASR latency."
        ),
        "config_path": str(experiment_dir / "config.yaml"),
        "checkpoint_path": str(checkpoint),
        "val_metrics": train_report["best_val_metrics"],
        "evaluation_split": "test",
        "evaluation_metrics": multimodal_test["metrics"],
        "test_metrics": multimodal_test["metrics"],
        "error_analysis": multimodal_test["error_analysis"],
        "model_size_mb_fp32": train_report["latency_cpu"]["num_parameters"] * 4 / (1024**2),
        "latency_cpu": train_report["latency_cpu"],
        "latency_gpu": train_report["latency_gpu"],
    }
    (experiment_dir / "result.json").write_text(json.dumps(multimodal, indent=2, default=str))

    summary_path = args.train_meta.parent / "transcription_summary.json"
    transcript_summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
    latency_path = Path("experiments/multimodal_latency.json")
    live_latency = json.loads(latency_path.read_text()) if latency_path.exists() else None
    comparison = {"audio_only": baseline, "audio_plus_text": multimodal}
    Path("experiments/multimodal_comparison.json").write_text(json.dumps(comparison, indent=2, default=str))
    DOCS_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    multimodal_report_path = DOCS_REPORT_ROOT / "multimodal_report.md"
    multimodal_report_path.write_text(
        _render_multimodal_report(baseline, multimodal, transcript_summary, live_latency)
    )

    all_results_path = Path("experiments/all_results.json")
    all_results = json.loads(all_results_path.read_text()) if all_results_path.exists() else []
    refreshed_ids = {baseline["experiment_id"], multimodal["experiment_id"]}
    all_results = [item for item in all_results if item["experiment_id"] not in refreshed_ids]
    all_results.extend([baseline, multimodal])
    all_results_path.write_text(json.dumps(all_results, indent=2, default=str))
    comparable = [
        item
        for item in all_results
        if item.get("experiment_protocol_version") == EXPERIMENT_PROTOCOL_VERSION
        and item.get("evaluation_split") == "test"
    ]
    _write_comparison_csv(comparable, Path("experiments/comparison.csv"))
    (DOCS_REPORT_ROOT / "multimodal_ablation_report.md").write_text(_render_report(comparable, 3))
    print(multimodal_report_path.read_text())


if __name__ == "__main__":
    main()
