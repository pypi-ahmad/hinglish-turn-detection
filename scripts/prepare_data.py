"""
Data preparation pipeline: pulls a filtered subset of Pipecat's smart-turn
train + test datasets, splits the train subset into train/val, and produces
a before/after augmentation preview (a handful of real examples run through
the complete augmentation stack, saved as audio pairs + a stats diff)
for manual inspection.

Run with:  python scripts/prepare_data.py
(add --train-scan-budget / --test-scan-budget to control how much of the
41GB dataset gets scanned -- see configs/config.py's ROW_SCAN_BUDGET_* for
the bandwidth/time trade-off this controls)

WHY WE DON'T MATERIALIZE A SEPARATE "AUGMENTED DATASET" DIRECTORY
----------------------------------------------------------------------
The brief (and common practice elsewhere) sometimes frames augmentation as
"apply it once, save the augmented dataset to disk". We deliberately do NOT
do that as the main path: our augmentations (dataset.py's
`TurnDetectionDataset`) run ON THE FLY, once per epoch, with fresh random
choices each time. Baking a single static augmented copy to disk would mean
the model sees the exact same (silence-length, filler-word, noise-level)
combination every single epoch, which is strictly less regularization than
re-rolling it every epoch -- and it would multiply our disk usage for no
benefit. What we DO materialize here is a small, fixed PREVIEW set (see
`preview_augmentations` below) purely so a human can listen to / read
before-vs-after examples and see the augmentation pipeline actually doing
something sensible -- that's a diagnostic artifact, not the training data.
The reusable processed representation is 16 kHz WAV audio plus Parquet
metadata/splits consumed by `TurnDetectionDataset`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.dataset import (
    FillerBank,
    add_background_noise,
    build_hard_negative_indices,
    inject_filler,
    insert_silence,
    load_noise_bank,
    pitch_shift,
    speed_perturb,
    stratified_split,
    stream_filtered_subset,
    volume_perturb,
)


def preview_augmentations(train_meta: pl.DataFrame, n_examples: int, out_dir: Path) -> dict:
    """Take a few real COMPLETE utterances, run them through silence-insertion
    and filler-injection, and save original+augmented pairs to `out_dir` so a
    human can listen to (or eyeball the waveform/duration of) what the
    augmentation pipeline is actually doing -- not just trust the code."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale_preview in out_dir.glob("*.wav"):
        stale_preview.unlink()
    if n_examples <= 0:
        return {"n_examples": 0, "examples": []}

    filler_bank = FillerBank(cache_dir=cfg.DATA_DIR / "fillers")
    filler_bank.build()
    noise_bank = load_noise_bank(cfg.DATA_DIR / "noise")

    # Preview primary/bridge languages instead of arbitrary row order so the
    # artifacts demonstrate the Hinglish-focused path they are meant to audit.
    n_primary = (n_examples + 1) // 2
    n_bridge = n_examples // 2
    preferred = pl.concat(
        [
            train_meta.filter((pl.col("language") == cfg.PRIMARY_LANGUAGE) & pl.col("endpoint_bool")).head(n_primary),
            train_meta.filter((pl.col("language") == cfg.BRIDGE_LANGUAGE) & pl.col("endpoint_bool")).head(n_bridge),
        ]
    )
    if preferred.height < n_examples:
        remainder = train_meta.filter(pl.col("endpoint_bool") & ~pl.col("id").is_in(preferred["id"]))
        complete_rows = pl.concat([preferred, remainder.head(n_examples - preferred.height)])
    else:
        complete_rows = preferred
    before_after = []

    for row in complete_rows.iter_rows(named=True):
        source_wav, sr = sf.read(cfg.ROOT / row["path"], dtype="float32")
        wav = source_wav[-cfg.MAX_REAL_SAMPLES :]
        before_path = out_dir / f"{row['id']}__before.wav"
        sf.write(before_path, wav, sr)

        aug_filler = inject_filler(wav, sr, filler_bank, position="mid")
        after_filler_path = out_dir / f"{row['id']}__after_midfiller.wav"
        sf.write(after_filler_path, aug_filler, sr)

        aug_silence = insert_silence(wav, sr, 300, 800, position="end")
        after_silence_path = out_dir / f"{row['id']}__after_endsilence.wav"
        sf.write(after_silence_path, aug_silence, sr)

        # Full deterministic preview: every configured augmentation is applied
        # once, in the same order as TurnDetectionDataset. Training itself
        # still samples each transform probabilistically on every epoch.
        aug_all = inject_filler(wav, sr, filler_bank, position="mid")
        aug_all = insert_silence(aug_all, sr, 100, 800, position="end")
        aug_all = speed_perturb(aug_all, sr)
        aug_all = pitch_shift(aug_all, sr)
        aug_all = add_background_noise(aug_all, sr, noise_bank)
        aug_all = volume_perturb(aug_all)
        aug_all = aug_all[-cfg.MAX_REAL_SAMPLES :]
        after_all_path = out_dir / f"{row['id']}__after_all.wav"
        sf.write(after_all_path, aug_all, sr)

        before_after.append(
            {
                "id": row["id"],
                "language": row["language"],
                "original_label": "complete",
                "source_duration_s": len(source_wav) / sr,
                "original_duration_s": len(wav) / sr,
                "after_midfiller_duration_s": len(aug_filler) / sr,
                "after_midfiller_new_label": "complete (label preserved)",
                "after_endsilence_duration_s": len(aug_silence) / sr,
                "after_endsilence_new_label": "complete (label preserved)",
                "after_all_duration_s": len(aug_all) / sr,
                "after_all_new_label": "complete (label preserved)",
                "after_all_augmentations": ["mid_filler", "end_silence", "speed", "pitch", "noise", "volume"],
            }
        )

    # Preview mirrors the training policy: filler insertion is internal and
    # label-preserving, and complete utterances only receive trailing silence.
    # We do not infer incompleteness from filler presence alone.
    return {"n_examples": len(before_after), "examples": before_after}


def print_before_after_stats(train_meta_raw: pl.DataFrame, preview: dict) -> None:
    """Print class and duration changes for human-auditable previews."""
    print("\n=== BEFORE (raw streamed subset) vs AFTER (augmentation preview) ===")
    print(f"Raw subset size: {train_meta_raw.height} rows")
    print(
        "Raw class balance: "
        f"{train_meta_raw['endpoint_bool'].sum()} complete / "
        f"{(~train_meta_raw['endpoint_bool']).sum()} incomplete"
    )
    print(f"Augmentation preview: {preview['n_examples']} complete examples run through the pipeline")
    if preview["examples"]:
        before_mean = np.mean([ex["original_duration_s"] for ex in preview["examples"]])
        after_mean = np.mean([ex["after_all_duration_s"] for ex in preview["examples"]])
        print(
            f"Preview class balance: before={preview['n_examples']} complete / 0 incomplete; "
            f"after full pipeline={preview['n_examples']} complete / 0 incomplete"
        )
        print(f"Preview mean model-window duration: before={before_mean:.2f}s, after={after_mean:.2f}s")
    for ex in preview["examples"][:5]:
        print(
            f"  id={ex['id'][:8]}... lang={ex['language']} "
            f"orig={ex['original_duration_s']:.2f}s -> "
            f"+midfiller={ex['after_midfiller_duration_s']:.2f}s, "
            f"+endsilence={ex['after_endsilence_duration_s']:.2f}s, "
            f"+all={ex['after_all_duration_s']:.2f}s"
        )


def main() -> None:
    """Stream bounded datasets, split metadata, and write inspection artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-scan-budget", type=int, default=cfg.ROW_SCAN_BUDGET_TRAIN)
    parser.add_argument("--test-scan-budget", type=int, default=cfg.ROW_SCAN_BUDGET_TEST)
    parser.add_argument("--n-preview-examples", type=int, default=10)
    parser.add_argument("--val-frac", type=float, default=0.12)
    args = parser.parse_args()

    print(f"=== Step 1/4: streaming + filtering TRAIN split (budget={args.train_scan_budget} rows) ===")
    train_meta_raw = stream_filtered_subset(
        cfg.TRAIN_DATASET_REPO,
        "train",
        args.train_scan_budget,
        cfg.RAW_CACHE_DIR / "train",
        cfg.SUBSET_DIR / "train_meta_raw.parquet",
    )

    print(f"\n=== Step 2/4: streaming + filtering TEST split (budget={args.test_scan_budget} rows) ===")
    test_meta = stream_filtered_subset(
        cfg.TEST_DATASET_REPO,
        "train",  # HF convention: this dataset's only split is still named "train" even though it's semantically our held-out test set
        args.test_scan_budget,
        cfg.RAW_CACHE_DIR / "test",
        cfg.SUBSET_DIR / "test_meta_raw.parquet",
    )

    print("\n=== Step 3/4: stratified train/val split ===")
    train_df, val_df, _ = stratified_split(train_meta_raw, val_frac=args.val_frac, test_frac=0.0)
    train_df.write_parquet(cfg.SUBSET_DIR / "train_split.parquet")
    val_df.write_parquet(cfg.SUBSET_DIR / "val_split.parquet")
    test_meta.write_parquet(cfg.SUBSET_DIR / "test_split.parquet")
    print(f"train={train_df.height} rows, val={val_df.height} rows, test={test_meta.height} rows")

    hard_indices = build_hard_negative_indices(train_df)
    hard_df = (
        train_df.with_row_index("_row_index")
        .filter(pl.col("_row_index").is_in(hard_indices.tolist()))
        .drop("_row_index")
        .with_columns(
            pl.when(pl.col("endpoint_bool"))
            .then(pl.lit("short_complete"))
            .otherwise(pl.lit("long_incomplete_with_filler"))
            .alias("hard_case")
        )
    )
    hard_path = cfg.SUBSET_DIR / "hard_negative_split.parquet"
    hard_df.write_parquet(hard_path)
    hard_complete = int(hard_df["endpoint_bool"].sum())
    print(
        f"hard cases={hard_df.height} ({hard_complete} short complete / "
        f"{hard_df.height - hard_complete} long incomplete-with-filler) -> {hard_path}"
    )

    print("\n=== Step 4/4: augmentation before/after preview ===")
    preview = preview_augmentations(train_df, args.n_preview_examples, cfg.DATA_DIR / "samples" / "augmentation_preview")
    print_before_after_stats(train_meta_raw, preview)
    (cfg.DATA_DIR / "samples" / "augmentation_preview.json").write_text(json.dumps(preview, indent=2))

    # A few plain labeled samples (no augmentation) for the Gradio demo's
    # gr.Examples and for manual listen-through -- one per language present,
    # preferring Hindi/English and a mix of complete/incomplete.
    sample_dir = cfg.DATA_DIR / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    # Root-level demo samples are generated artifacts. Remove stale samples
    # from older split runs so labels.csv always describes every clip shown
    # by the demo. Augmentation previews live in a subdirectory and remain.
    for old_sample in sample_dir.glob("*.wav"):
        old_sample.unlink()

    picks = (
        pl.concat(
            [
                train_df.filter((pl.col("language") == "hin") & (pl.col("endpoint_bool"))).head(3),
                train_df.filter((pl.col("language") == "hin") & (~pl.col("endpoint_bool"))).head(3),
                train_df.filter((pl.col("language") == "eng") & (pl.col("endpoint_bool"))).head(2),
                train_df.filter((pl.col("language") == "eng") & (~pl.col("endpoint_bool"))).head(2),
            ]
        )
    )
    sample_records = []
    for row in picks.iter_rows(named=True):
        src = cfg.ROOT / row["path"]
        label = "complete" if row["endpoint_bool"] else "incomplete"
        dst = sample_dir / f"{row['language']}_{label}_{row['id'][:8]}.wav"
        wav, sr = sf.read(src, dtype="float32")
        sf.write(dst, wav, sr)
        sample_records.append(
            {
                "file": dst.name,
                "id": row["id"],
                "language": row["language"],
                "label": label,
                "endpoint_bool": bool(row["endpoint_bool"]),
                "duration_s": float(row["duration_s"]),
                "source_dataset": row["source_dataset"],
            }
        )
    labels_path = sample_dir / "labels.csv"
    pl.DataFrame(sample_records).write_csv(labels_path)
    print(f"wrote {picks.height} labeled example clips + {labels_path} for manual inspection and Gradio")


if __name__ == "__main__":
    main()
