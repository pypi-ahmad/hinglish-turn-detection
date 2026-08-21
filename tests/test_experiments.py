import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from scripts.run_experiments import (
    EXPERIMENTS,
    _apply_overrides,
    _docs_report_path,
    _multimodal_result,
    _render_report,
    build_experiment_manifest,
)
from src.evaluate import error_analysis


def _completed(experiment_id: str, f1: float = 0.8) -> dict:
    metrics = {
        "accuracy": 0.81,
        "precision": 0.79,
        "recall": 0.82,
        "f1": f1,
        "auc": 0.9,
        "false_complete_rate": 0.1,
    }
    latency = {"latency_ms_mean": 20.0, "num_parameters": 8_000_000}
    return {
        "experiment_id": experiment_id,
        "status": "completed",
        "description": "test run",
        "test_metrics": metrics,
        "model_size_mb_fp32": latency["num_parameters"] * 4 / (1024**2),
        "latency_cpu": latency,
        "latency_gpu": {"latency_ms_mean": 5.0, "num_parameters": 8_000_000},
    }


class ExperimentFrameworkTests(unittest.TestCase):
    def test_human_reports_live_under_docs(self):
        self.assertEqual(_docs_report_path(None), Path("docs/generated/ablation_report.md"))
        self.assertEqual(
            _docs_report_path("seed/43"), Path("docs/generated/seed_43_ablation_report.md")
        )

    def test_overrides_do_not_mutate_baseline(self):
        baseline = {"model": {"pooling": "attention"}}

        changed = _apply_overrides(baseline, {"model.pooling": "mean"})

        self.assertEqual(changed["model"]["pooling"], "mean")
        self.assertEqual(baseline["model"]["pooling"], "attention")

    def test_report_contains_comparison_insights_and_skip_reason(self):
        skipped = {
            "experiment_id": "M1_audio_plus_text",
            "status": "skipped",
            "reason": "No transcripts.",
        }

        report = _render_report([_completed("E1_no_augmentation", 0.7), _completed("E2_augmented"), skipped], 3)

        self.assertIn("## Comparison", report)
        self.assertIn("## Pause and hard-case slices", report)
        self.assertIn("## Per-experiment insights", report)
        self.assertIn("F1 +10.00 points", report)
        self.assertIn("No transcripts.", report)

    def test_report_does_not_rank_degenerate_low_recall_model_as_safest(self):
        baseline = _completed("E2_augmented")
        degenerate = _completed("E7_frozen_encoder", 0.3)
        degenerate["test_metrics"]["recall"] = 0.2
        degenerate["test_metrics"]["false_complete_rate"] = 0.01

        report = _render_report([baseline, degenerate], 3)

        self.assertIn("at least 75% recall: **E2_augmented**", report)
        self.assertIn("over-predicting incomplete", report)

    def test_multimodal_result_is_compared_with_unaugmented_audio_control(self):
        baseline = _completed("E1_no_augmentation", 0.8)
        augmented = _completed("E2_augmented", 0.7)
        multimodal = _completed("M1_audio_plus_text", 0.81)

        report = _render_report([baseline, augmented, multimodal], 3)

        multimodal_section = report.split("### M1_audio_plus_text", maxsplit=1)[1]
        self.assertIn("Versus E1_no_augmentation", multimodal_section)

    def test_missing_transcripts_record_multimodal_as_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "metadata.parquet"
            pl.DataFrame({"id": ["sample"]}).write_parquet(metadata)

            with patch("scripts.run_experiments.OUTPUT_ROOT", Path(directory)):
                result = _multimodal_result(str(metadata))

        self.assertEqual(result["status"], "skipped")
        self.assertIn("no transcript", result["reason"])

    def test_plan_contains_isolated_filler_silence_and_hard_mining_controls(self):
        by_id = {experiment.experiment_id: experiment for experiment in EXPERIMENTS}

        self.assertEqual(by_id["E9_no_filler"].overrides["data.augment_config.p_filler"], 0.0)
        self.assertEqual(by_id["E11_no_silence"].overrides["data.augment_config.p_silence"], 0.0)
        self.assertFalse(by_id["E12_no_hard_mining"].overrides["data.use_hard_negatives"])

    def test_manifest_fingerprints_data_and_resolves_config(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "split.parquet"
            pl.DataFrame({"id": ["sample"]}).write_parquet(metadata)
            base = {
                "experiment_name": "base",
                "model": {"pooling": "attention"},
                "data": {
                    "use_augmentation": True,
                    "augment_config": {"p_filler": 0.35},
                },
                "training": {"num_epochs": 3},
                "checkpoint": {"dir": "checkpoints/base"},
            }

            manifest = build_experiment_manifest(
                base,
                {"E9_no_filler"},
                epochs=2,
                train_meta=metadata,
                val_meta=metadata,
                test_meta=metadata,
            )

        run = manifest["experiments"][0]
        self.assertEqual(run["resolved_config"]["training"]["num_epochs"], 2)
        self.assertEqual(run["resolved_config"]["data"]["augment_config"]["p_filler"], 0.0)
        self.assertEqual(manifest["datasets"]["train"]["rows"], 1)
        self.assertEqual(len(manifest["datasets"]["train"]["sha256"]), 64)

    def test_error_analysis_reports_pause_and_hard_hinglish_slices(self):
        result = {
            "labels": np.array([0, 1]),
            "preds": np.array([1, 1]),
            "probs": np.array([0.8, 0.9]),
            "meta": [
                {
                    "id": "hard",
                    "language": "hin",
                    "midfiller": True,
                    "endfiller": False,
                    "synthetic": False,
                    "duration_s": 5.0,
                    "internal_pause": True,
                    "trailing_pause": False,
                },
                {
                    "id": "easy",
                    "language": "eng",
                    "midfiller": False,
                    "endfiller": False,
                    "synthetic": False,
                    "duration_s": 1.0,
                    "internal_pause": False,
                    "trailing_pause": True,
                },
            ],
        }

        slices = error_analysis(result)["slices"]

        self.assertEqual(slices["internal_pause"]["n"], 1)
        self.assertEqual(slices["hard_hinglish_proxy"]["n"], 1)
        self.assertEqual(slices["hard_hinglish_proxy"]["false_complete_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
