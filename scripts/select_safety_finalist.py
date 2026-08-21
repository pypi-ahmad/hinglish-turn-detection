"""Select one deployable checkpoint from repeated validation-only safety runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.evaluate import evaluate_checkpoint


def select_finalist(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Select architecture by median constrained F1, then median seed run."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "completed" or result.get("evaluation_split") != "validation":
            raise ValueError("all finalist inputs must be completed validation results")
        grouped.setdefault(result["experiment_id"], []).append(result)
    if not grouped or len({len(runs) for runs in grouped.values()}) != 1:
        raise ValueError("each architecture must have the same non-zero seed count")

    summaries = []
    for experiment_id, runs in grouped.items():
        summaries.append(
            {
                "experiment_id": experiment_id,
                "seed_count": len(runs),
                "all_seeds_feasible": all(
                    bool(run.get("threshold_calibration", {}).get("feasible")) for run in runs
                ),
                "median_val_f1": median(run["val_metrics"]["f1"] for run in runs),
                "median_val_fcr": median(
                    run["val_metrics"]["false_complete_rate"] for run in runs
                ),
                "runs": runs,
            }
        )
    eligible = [summary for summary in summaries if summary["all_seeds_feasible"]]
    if not eligible:
        raise RuntimeError("no architecture met safety constraints across every seed")
    winner = min(
        eligible,
        key=lambda item: (-item["median_val_f1"], item["median_val_fcr"], item["experiment_id"]),
    )
    ordered_runs = sorted(winner["runs"], key=lambda run: (run["val_metrics"]["f1"], run["seed"]))
    deployed_run = ordered_runs[len(ordered_runs) // 2]
    return {
        "selection_policy": "highest median constrained validation F1; tie lower median FCR",
        "architectures": [{key: value for key, value in item.items() if key != "runs"} for item in summaries],
        "winning_experiment_id": winner["experiment_id"],
        "deployed_seed": deployed_run["seed"],
        "deployed_validation_metrics": deployed_run["val_metrics"],
        "decision_threshold": deployed_run["decision_threshold"],
        "source_checkpoint": deployed_run["checkpoint_path"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_roots", nargs="+", help="Seed run directories containing result.json files")
    parser.add_argument("--output-dir", default="checkpoints/safety_finalist")
    parser.add_argument("--test-meta", default=str(cfg.SUBSET_DIR / "test_split.parquet"))
    args = parser.parse_args()

    results = []
    for root_text in args.run_roots:
        root = Path(root_text)
        for result_path in sorted(root.glob("*/result.json")):
            results.append(json.loads(result_path.read_text()))
    selection = select_finalist(results)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "best.pt"
    shutil.copy2(selection["source_checkpoint"], destination)
    selection["deployed_checkpoint"] = str(destination)
    selection["heldout_test"] = evaluate_checkpoint(destination, args.test_meta)
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2, default=str))
    print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
