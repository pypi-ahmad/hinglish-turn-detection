"""
One-off diagnostic: the real baseline training run hung mid-epoch-2 with GPU
idle but CPU actively burning for several minutes on a single batch -- a
pathological slowdown/hang inside the augmentation pipeline (dataset.py's
TurnDetectionDataset.__getitem__), not the model itself. This script times
EVERY row's augmented __getitem__ call individually, with a per-item hard
timeout, to find exactly which row(s) are pathological and why -- rather
than guessing and patching blindly.

Run with:  python scripts/diagnose_slow_augment.py
"""

from __future__ import annotations

import signal
import sys
import time
from pathlib import Path
from types import FrameType

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.dataset import AugmentConfig, FillerBank, TurnDetectionDataset


class Timeout(Exception):
    """Raised when POSIX augmentation diagnostic exceeds timeout."""


def _handle_alarm(signum: int, frame: FrameType | None) -> None:
    raise Timeout()


def main() -> None:
    """Scan prepared training rows and report slow augmentation calls."""
    train_meta = pl.read_parquet(cfg.SUBSET_DIR / "train_split.parquet")
    filler_bank = FillerBank(cache_dir=cfg.DATA_DIR / "fillers")  # already built, just reads cache

    augment_cfg = AugmentConfig(
        enabled=True, p_silence=0.3, p_speed=0.3, p_pitch=0.2, p_noise=0.2, p_volume=0.3, p_filler=0.35,
        silence_ms_range=(100, 800),
    )
    ds = TurnDetectionDataset(train_meta, augment=augment_cfg, filler_bank=filler_bank)

    print(f"scanning {len(ds)} rows for slow/hung augmentation calls (>3s = suspicious, 20s timeout)...")

    # SIGALRM is POSIX-only, but this environment's Bash tool runs Git Bash
    # (MSYS) on top of Windows Python, where `signal.alarm` is NOT available
    # (Windows Python doesn't implement SIGALRM). Fall back to plain timing
    # without a hard kill -- we can't interrupt a hung call on Windows from
    # pure Python without threads/subprocess, so we just log slow ones and
    # manually interrupt (Ctrl+C) if something truly never returns.
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        signal.signal(signal.SIGALRM, _handle_alarm)

    slow = []
    for i in range(len(ds)):
        row = train_meta.row(i, named=True)
        t0 = time.time()
        try:
            if has_alarm:
                signal.alarm(20)
            ds[i]
            if has_alarm:
                signal.alarm(0)
        except Timeout:
            print(f"TIMEOUT at row {i} id={row['id']} lang={row['language']} duration_s={row['duration_s']:.2f}")
            slow.append((i, row["id"], row["duration_s"], "TIMEOUT"))
            continue
        elapsed = time.time() - t0
        if elapsed > 3.0:
            print(
                f"SLOW row {i} id={row['id']} lang={row['language']} "
                f"duration_s={row['duration_s']:.2f} synthetic={row['synthetic']} took {elapsed:.1f}s"
            )
            slow.append((i, row["id"], row["duration_s"], elapsed))
        if i % 500 == 0:
            print(f"  ...scanned {i}/{len(ds)}")

    print(f"\ndone. {len(slow)} slow/timeout rows out of {len(ds)}")
    for s in slow:
        print(s)


if __name__ == "__main__":
    main()
