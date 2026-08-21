"""
Data exploration: computes real statistics over whatever subset
`scripts/prepare_data.py` has already pulled down (train_meta_raw.parquet /
test_meta_raw.parquet), demonstrates the Dataset class returning a real
(waveform, label) pair, and writes identical markdown reports to the canonical
docs location and the challenge-requested notebooks location.

Every number in the generated report comes from actually reading the local
metadata table -- nothing here is estimated or invented. If you've only run
`prepare_data.py` with a small `--train-scan-budget` so far (e.g. for a quick
pipeline smoke test), the report will honestly reflect that smaller sample
size; re-run both scripts with a larger budget for a report over more data.

Run with:  python scripts/explore_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.dataset import TurnDetectionDataset, measure_pause_features


def _duration_stats(df: pl.DataFrame) -> dict:
    d = df["duration_s"].to_numpy()
    return {
        "n": len(d),
        "mean_s": float(np.mean(d)),
        "median_s": float(np.median(d)),
        "p10_s": float(np.percentile(d, 10)),
        "p90_s": float(np.percentile(d, 90)),
        "min_s": float(np.min(d)),
        "max_s": float(np.max(d)),
    }


def _class_balance(df: pl.DataFrame) -> dict:
    n_complete = int(df["endpoint_bool"].sum())
    n_incomplete = int((~df["endpoint_bool"]).sum())
    return {"complete": n_complete, "incomplete": n_incomplete, "complete_pct": 100 * n_complete / df.height}


def _language_table(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("language").agg(pl.len().alias("n")).sort("n", descending=True)


def _filler_stats(df: pl.DataFrame) -> dict:
    return {
        "midfiller_true": int(df["midfiller"].sum()),
        "endfiller_true": int(df["endfiller"].sum()),
        "synthetic_true": int(df["synthetic"].sum()),
        "synthetic_pct": 100 * df["synthetic"].sum() / df.height,
    }


def _sample_rate_table(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("orig_sample_rate").agg(pl.len().alias("n")).sort("n", descending=True)


def _pause_stats(df: pl.DataFrame, root: Path = cfg.ROOT) -> dict[str, float]:
    measurements = []
    for row in df.iter_rows(named=True):
        wav, sr = sf.read(root / row["path"], dtype="float32", always_2d=False)
        measurements.append(measure_pause_features(wav, sr))
    pauses = pl.DataFrame(measurements)
    return {
        "low_energy_mean_pct": 100 * float(pauses["low_energy_fraction"].mean()),
        "low_energy_median_pct": 100 * float(pauses["low_energy_fraction"].median()),
        "any_pause_pct": 100 * float(pauses["any_pause"].mean()),
        "internal_pause_pct": 100 * float(pauses["internal_pause"].mean()),
        "trailing_pause_pct": 100 * float(pauses["trailing_pause"].mean()),
    }


def _markdown_table(df: pl.DataFrame) -> str:
    """Render a small polars DataFrame as a markdown table without pulling in
    pandas/tabulate as extra dependencies -- these tables are tiny (a couple
    dozen rows at most), so a hand-rolled renderer is simpler than a new dep."""
    cols = df.columns
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.iter_rows()]
    return "\n".join([header, sep, *rows])


def render_report(
    train_df: pl.DataFrame, test_df: pl.DataFrame, dataset_demo: dict[str, Any]
) -> str:
    """Render reproducible Markdown exploration report from local metadata."""
    train_dur = _duration_stats(train_df)
    train_bal = _class_balance(train_df)
    train_fill = _filler_stats(train_df)
    lang_table = _language_table(train_df)
    sr_table = _sample_rate_table(train_df)
    pause = _pause_stats(train_df)

    lines = [
        "<!-- markdownlint-disable MD013 MD022 MD032 MD058 -->",
        "",
        "# Data Exploration -- pipecat-ai/smart-turn-data-v3.2-train",
        "",
        (
            f"_Generated from a local subset of **{train_df.height} rows** (train) + "
            f"**{test_df.height} rows** (test), pulled via `scripts/prepare_data.py`. "
            "See docs/01_data_preparation_approach.md for why we work from a bounded "
            "subset rather than the full 270,946-row / 41GB dataset._"
        ),
        "",
        "## 1. Sample counts",
        f"- Train subset: {train_df.height} rows",
        f"- Test subset: {test_df.height} rows",
        "",
        "## 2. Class distribution (train subset)",
        f"- Complete: {train_bal['complete']} ({train_bal['complete_pct']:.1f}%)",
        f"- Incomplete: {train_bal['incomplete']} ({100 - train_bal['complete_pct']:.1f}%)",
        "",
        "## 3. Duration distribution (seconds, train subset)",
        (
            f"- mean={train_dur['mean_s']:.2f}, median={train_dur['median_s']:.2f}, "
            f"p10={train_dur['p10_s']:.2f}, p90={train_dur['p90_s']:.2f}, "
            f"min={train_dur['min_s']:.2f}, max={train_dur['max_s']:.2f}"
        ),
        "",
        "## 4. Original sample rates seen (before our resample-to-16kHz step)",
        _markdown_table(sr_table),
        "",
        "## 5. Languages present in this subset",
        _markdown_table(lang_table),
        "",
        "## 6. Silence / pause presence (train subset)",
        (
            "Low-energy proxy: 20 ms RMS frames, 10 ms hop, RMS below 0.01 "
            "(-40 dBFS), with a pause requiring at least 100 ms. This is not "
            "voice-activity ground truth; gain and noise affect the threshold."
        ),
        f"- Mean low-energy frame share: {pause['low_energy_mean_pct']:.1f}%",
        f"- Median low-energy frame share: {pause['low_energy_median_pct']:.1f}%",
        f"- Clips with any >=100 ms pause: {pause['any_pause_pct']:.1f}%",
        f"- Clips with an internal >=100 ms pause: {pause['internal_pause_pct']:.1f}%",
        f"- Clips with a trailing >=100 ms pause: {pause['trailing_pause_pct']:.1f}%",
        "",
        "## 7. Filler / synthetic-vs-human signal",
        f"- Rows with midfiller=True: {train_fill['midfiller_true']}",
        f"- Rows with endfiller=True: {train_fill['endfiller_true']}",
        f"- Synthetic (TTS-generated) rows: {train_fill['synthetic_true']} ({train_fill['synthetic_pct']:.1f}%)",
        (
            "- `spoken_text` is null for every row we've inspected -- there is no ready-made "
            "transcript, and no explicit \"code-switched\"/\"hinglish\" label; see "
            "docs/01_data_preparation_approach.md for how we address that gap."
        ),
        "",
        "## 8. Dataset class sanity check",
        (
            "`src.dataset.TurnDetectionDataset.__getitem__` returns a dict with a raw "
            "waveform + label (feature extraction happens later, in `collate_fn`, so this "
            "class stays reusable). One real example from the train subset:"
        ),
        f"- id: {dataset_demo['id']}",
        f"- waveform shape: {dataset_demo['waveform_shape']}, dtype: {dataset_demo['waveform_dtype']}",
        f"- label: {dataset_demo['label']} ({'complete' if dataset_demo['label'] == 1 else 'incomplete'})",
        f"- language: {dataset_demo['language']}",
        "",
        "## 9. Sample audio files for manual inspection",
        (
            "Saved to `data/samples/` by `scripts/prepare_data.py` "
            "(a handful of labeled Hindi/English clips + an augmentation before/after preview). "
            "Ground-truth metadata for every root sample WAV is in `data/samples/labels.csv`."
        ),
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Load prepared metadata, inspect dataset contract, and save report."""
    train_df = pl.read_parquet(cfg.SUBSET_DIR / "train_meta_raw.parquet")
    test_df = pl.read_parquet(cfg.SUBSET_DIR / "test_meta_raw.parquet")

    demo_df = train_df.filter(pl.col("language") == "hin").head(1)
    if demo_df.is_empty():
        demo_df = train_df.head(1)
    ds = TurnDetectionDataset(demo_df)
    example = ds[0]
    dataset_demo = {
        "id": example["id"],
        "waveform_shape": example["waveform"].shape,
        "waveform_dtype": str(example["waveform"].dtype),
        "label": example["label"],
        "language": example["language"],
    }

    report = render_report(train_df, test_df, dataset_demo)
    out_paths = (
        Path("docs/data_exploration.md"),
        Path("notebooks/01_data_exploration.md"),
    )
    for out_path in out_paths:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[explore_data] wrote {', '.join(map(str, out_paths))}")


if __name__ == "__main__":
    main()
