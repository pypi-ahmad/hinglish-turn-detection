"""
Data pipeline for Hinglish-focused turn detection.

WHAT LIVES IN THIS FILE
------------------------
1. `stream_filtered_subset(...)`  -- pulls a bounded, filtered slice of
   Pipecat's 41GB/270k-row `smart-turn-data-v3.2-train` dataset down to local
   disk (we never download the whole thing -- see the "WHY A BOUNDED SCAN"
   section below), decoding + resampling audio and writing a metadata table.

2. `stratified_split(...)` -- splits that local metadata table into
   train/val/test.

3. Augmentation primitives (`insert_silence`, `speed_perturb`, `pitch_shift`,
   `add_background_noise`, `volume_perturb`, `inject_filler`) -- each takes a
   waveform (+ sample rate) and returns a modified waveform, so they compose
   freely.

4. `FillerBank` -- synthesizes a small library of Hinglish filler-word audio
   clips (via Microsoft Edge's free neural TTS, in Hindi + Indian-English
   voices) that `inject_filler` splices into existing clips.

5. `TurnDetectionDataset` -- a torch Dataset that returns
   `{"waveform": np.ndarray, "label": int, ...metadata}` per the brief's
   literal requirement ("a clean Dataset class that returns audio waveform +
   label"). Feature extraction (log-mel spectrogram) is deliberately NOT done
   here -- see `collate_fn` -- so this class stays reusable for anything that
   wants raw waveforms (e.g. playback, other feature extractors), not just
   our specific Whisper-tiny model.

6. `collate_fn(...)` -- the model-specific step: left-pads a batch of
   variable-length waveforms to the model's 8s window, builds a frame-level attention
   mask, and runs them through `WhisperFeatureExtractor` to get the log-mel
   spectrograms our model actually consumes (see models.py).

WHY A BOUNDED SCAN INSTEAD OF DOWNLOADING THE WHOLE DATASET
--------------------------------------------------------------
The full train split is 270,946 rows / 41.4GB, spread across 23 languages.
Hindi ("hin") is only ~4.5% of rows, with (as far as we can tell) no sort
order we could exploit to fetch just those rows cheaply -- HF's streaming
parquet reader still has to materialize each row (including its audio bytes)
before we get a chance to inspect its `language` field in Python. So
"guarantee we get every Hindi row" effectively means "read close to all 41GB".

Instead we set a hard ROW-COUNT budget (`ROW_SCAN_BUDGET_TRAIN`, see
configs/config.py) on how many rows we're willing to *stream past* and keep
whatever Hindi ("hin") and English ("eng", the language Hinglish code-switches
with) rows show up inside that budget, plus a small opportunistic sample of
whatever other languages happen to appear (free, since we already paid the
bandwidth cost for those rows while scanning past them). This is a deliberate,
documented time/bandwidth trade-off -- see docs/01_data_preparation_approach.md
for the reasoning -- not an oversight.

WHY WE BUILD OUR OWN HINGLISH EXAMPLES INSTEAD OF ONLY USING "hin"-TAGGED ROWS
---------------------------------------------------------------------------------
Critically, the dataset's `language` column is a single per-row tag -- there
is no "hinglish" or "code-switched" label, and `spoken_text` is null in the
published schema, so code-switching cannot be measured from metadata. Real
Hinglish speech mixes Hindi and English *within one utterance* and uses
fillers like "matlab", "actually", and "yaar". On top of the real "hin"/"eng"
rows, we therefore splice Hindi/Indian-English filler audio into both classes
as a label-preserving robustness augmentation. We deliberately avoid treating
an isolated filler as proof of incompleteness: words such as "bas", "haan",
and "wait" can themselves close a turn.
"""

from __future__ import annotations

import asyncio
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg

# ---------------------------------------------------------------------------
# 1. Streaming download + language filtering
# ---------------------------------------------------------------------------


def _resample_if_needed(array: np.ndarray, orig_sr: int, target_sr: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """Resample to our target sample rate (16kHz, what Whisper expects) if the
    source clip wasn't already at that rate. Different contributors to the
    dataset (human recordings, several different TTS engines) may not all
    ship audio at exactly 16kHz, so we can't assume it."""
    if isinstance(orig_sr, bool) or not isinstance(orig_sr, (int, np.integer)) or orig_sr <= 0:
        raise ValueError(f"source sample rate must be a positive integer, got {orig_sr!r}")
    if isinstance(target_sr, bool) or not isinstance(target_sr, (int, np.integer)) or target_sr <= 0:
        raise ValueError(f"target sample rate must be a positive integer, got {target_sr!r}")
    if orig_sr == target_sr:
        return array.astype(np.float32, copy=False)
    import librosa

    return librosa.resample(array.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio_waveform(
    array: np.ndarray, orig_sr: int, target_sr: int = cfg.SAMPLE_RATE
) -> np.ndarray:
    """Validate decoded audio, collapse mono/stereo layouts, and resample.

    Hugging Face decoders commonly return ``[samples]`` or
    ``[channels, samples]`` while other decoders use ``[samples, channels]``.
    Accept either stereo layout, but reject ambiguous higher-dimensional data
    instead of silently writing malformed WAV files.
    """
    waveform = np.asarray(array)
    if waveform.ndim == 2:
        if waveform.shape[0] <= 8 and waveform.shape[1] > waveform.shape[0]:
            waveform = waveform.mean(axis=0)
        elif waveform.shape[1] <= 8 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.mean(axis=1)
        else:
            raise ValueError(f"ambiguous two-dimensional audio shape: {waveform.shape}")
    elif waveform.ndim != 1:
        raise ValueError(f"audio must be mono or stereo, got shape {waveform.shape}")
    if waveform.size == 0:
        raise ValueError("audio waveform is empty")
    waveform = waveform.astype(np.float32, copy=False)
    if not np.isfinite(waveform).all():
        raise ValueError("audio waveform contains NaN or infinite values")
    return np.asarray(_resample_if_needed(waveform, orig_sr, target_sr), dtype=np.float32)


def _ensure_annotation_provenance(metadata: pl.DataFrame) -> pl.DataFrame:
    """Add nullable provenance columns to metadata created by older versions.

    Legacy caches already collapsed source nulls to ``False``. Their original
    annotation availability cannot be reconstructed safely, so mark it unknown
    (null) rather than guessing from ``synthetic`` or source-batch fields.
    """
    missing = [
        name
        for name in ("midfiller_annotation_known", "endfiller_annotation_known")
        if name not in metadata.columns
    ]
    if not missing:
        return metadata
    return metadata.with_columns(pl.lit(None, dtype=pl.Boolean).alias(name) for name in missing)


def measure_pause_features(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    rms_threshold: float = 0.01,
    min_pause_ms: int = 100,
) -> dict[str, float | bool]:
    """Return low-energy pause proxies using 20 ms RMS frames and 10 ms hops.

    ``rms_threshold=0.01`` is -40 dBFS for unit-scale float audio. These are
    reproducible acoustic slices, not ground-truth voice activity labels.
    """
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if wav.ndim != 1 or wav.size == 0 or not np.isfinite(wav).all():
        raise ValueError("waveform must be non-empty, finite mono/stereo audio")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    frame = max(1, round(0.020 * sample_rate))
    hop = max(1, round(0.010 * sample_rate))
    min_frames = max(1, round(min_pause_ms / 10))
    if len(wav) < frame:
        wav = np.pad(wav, (0, frame - len(wav)))

    starts = np.arange(0, len(wav) - frame + 1, hop)
    power = np.square(wav, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    rms = np.sqrt((cumulative[starts + frame] - cumulative[starts]) / frame)
    low = rms < rms_threshold
    run_starts = np.flatnonzero(low & np.r_[True, ~low[:-1]])
    run_ends = np.flatnonzero(low & np.r_[~low[1:], True]) + 1
    long_runs = [(int(start), int(end)) for start, end in zip(run_starts, run_ends) if end - start >= min_frames]

    return {
        "low_energy_fraction": float(low.mean()),
        "any_pause": bool(long_runs),
        "internal_pause": any(start > 0 and end < len(low) for start, end in long_runs),
        "trailing_pause": any(end == len(low) for _, end in long_runs),
    }


def stream_filtered_subset(
    repo_id: str,
    split: str,
    row_scan_budget: int,
    out_audio_dir: Path,
    out_meta_path: Path,
    max_primary_rows: int = cfg.MAX_PRIMARY_ROWS,
    max_bridge_rows: int = cfg.MAX_BRIDGE_ROWS,
    max_rows_per_other_lang: int = cfg.MAX_ROWS_PER_OTHER_LANG,
) -> pl.DataFrame:
    """Stream `repo_id`'s `split`, keep up to `row_scan_budget` rows worth of
    network traffic, and save every Hindi/English row (up to their caps) plus
    an opportunistic sample of other languages to `out_audio_dir` as 16kHz
    mono WAV files. Writes + returns a metadata table (one row per saved clip).

    This function is idempotent-ish: if `out_meta_path` already exists we load
    and return it directly rather than re-downloading (re-running data prep
    during development shouldn't re-pull gigabytes of audio every time).
    """
    if row_scan_budget <= 0:
        raise ValueError("row_scan_budget must be positive")
    if out_meta_path.exists():
        print(f"[dataset] found existing metadata at {out_meta_path}, skipping re-download")
        cached = pl.read_parquet(out_meta_path)
        upgraded = _ensure_annotation_provenance(cached)
        if upgraded.columns != cached.columns:
            upgraded.write_parquet(out_meta_path)
            print("[dataset] marked legacy filler-annotation provenance as unknown")
        return upgraded

    out_audio_dir.mkdir(parents=True, exist_ok=True)

    # `streaming=True` gives us a generator over rows without ever writing the
    # full 41GB to disk -- rows are decoded (including their audio) one at a
    # time as we iterate.
    ds = load_dataset(repo_id, split=split, streaming=True)
    # Force audio decoding at whatever native sample rate each clip has; we
    # resample ourselves in `_resample_if_needed` so we can log when a clip
    # *wasn't* already 16kHz (useful for the exploration report).
    ds = ds.cast_column("audio", Audio(decode=True))

    records: list[dict] = []
    other_lang_counts: dict[str, int] = {}
    primary_count = 0
    bridge_count = 0

    pbar = tqdm(total=row_scan_budget, desc=f"scanning {repo_id}:{split}")
    for i, row in enumerate(ds):
        pbar.update(1)
        if i >= row_scan_budget:
            break

        row_id = row["id"]
        lang = row["language"]
        endpoint = row["endpoint_bool"]
        if not isinstance(row_id, str) or not row_id:
            raise ValueError(f"row {i} has invalid id: {row_id!r}")
        if not isinstance(lang, str) or not lang:
            raise ValueError(f"row {row_id} has invalid language: {lang!r}")
        if not isinstance(endpoint, (bool, np.bool_)):
            raise TypeError(f"row {row_id} has non-binary endpoint_bool: {endpoint!r}")
        keep = False

        if lang == cfg.PRIMARY_LANGUAGE and primary_count < max_primary_rows:
            keep, primary_count = True, primary_count + 1
        elif lang == cfg.BRIDGE_LANGUAGE and bridge_count < max_bridge_rows:
            keep, bridge_count = True, bridge_count + 1
        elif lang not in (cfg.PRIMARY_LANGUAGE, cfg.BRIDGE_LANGUAGE):
            n = other_lang_counts.get(lang, 0)
            if n < max_rows_per_other_lang:
                keep, other_lang_counts[lang] = True, n + 1

        if not keep:
            continue

        audio = row["audio"]
        array = normalize_audio_waveform(audio["array"], audio["sampling_rate"])
        duration_s = len(array) / cfg.SAMPLE_RATE

        out_path = out_audio_dir / f"{row_id}.wav"
        sf.write(out_path, array, cfg.SAMPLE_RATE)

        records.append(
            {
                "id": row_id,
                "language": lang,
                "endpoint_bool": bool(endpoint),
                # Some human rows have null filler flags in published samples.
                # Keep runtime booleans for downstream masks, but retain whether
                # each value was actually annotated so false never means
                # "verified absent" when the source value was unknown.
                "midfiller": bool(row["midfiller"]) if row["midfiller"] is not None else False,
                "endfiller": bool(row["endfiller"]) if row["endfiller"] is not None else False,
                "midfiller_annotation_known": row["midfiller"] is not None,
                "endfiller_annotation_known": row["endfiller"] is not None,
                "synthetic": bool(row["synthetic"]),
                "source_dataset": row["dataset"],
                "duration_s": duration_s,
                "orig_sample_rate": audio["sampling_rate"],
                "path": str(out_path.relative_to(cfg.ROOT)),
                "is_augmented": False,
                "augment_type": None,
            }
        )

        # Early exit once we can't possibly take any more primary/bridge rows
        # and every other-language bucket we've seen is already full -- no
        # point paying for more network traffic than necessary.
        if (
            primary_count >= max_primary_rows
            and bridge_count >= max_bridge_rows
            and all(v >= max_rows_per_other_lang for v in other_lang_counts.values())
            and len(other_lang_counts) >= 15  # seen most of the ~21 "other" languages at least once
        ):
            break
    pbar.close()

    df = pl.DataFrame(records)
    df.write_parquet(out_meta_path)
    print(
        f"[dataset] kept {len(df)} rows "
        f"(hin={primary_count}, eng={bridge_count}, other_langs={len(other_lang_counts)}) "
        f"out of {i + 1} scanned"
    )
    return df


# ---------------------------------------------------------------------------
# 2. Stratified train/val/test split
# ---------------------------------------------------------------------------


def stratified_split(
    df: pl.DataFrame, val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split into train/val/test, stratified by (language, endpoint_bool,
    source_dataset).

    WHY THIS PROXY FOR "SPEAKER/ACCENT-AWARE" SPLITTING
    ------------------------------------------------------
    The schema has no `speaker_id` or `accent` column (only `language`,
    `synthetic`, and `dataset` -- a source-batch tag like "human_5_all",
    "liva_1", "chirp3_1"). There is therefore no way to do a *true*
    speaker-disjoint split (same speaker never appearing in both train and
    test) with the fields available. `source_dataset` is the closest usable
    proxy: each tag corresponds to one recording batch / TTS voice pool, so
    stratifying (not splitting *by*, since we still want every stratum
    represented in every split -- just in the same proportions) on it at least
    keeps the class balance and voice-pool mix consistent across splits,
    which is what a speaker-aware split is trying to protect against
    (accidentally evaluating on a distribution that doesn't match training).
    We stratify rather than group-split because group-splitting by
    `source_dataset` would remove entire languages/voice-pools from either
    train or test (some tags are language-specific), which is worse for a
    small dataset than a small amount of within-batch train/test leakage.
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be non-negative and sum to less than 1")

    rng = random.Random(seed)
    df = df.with_columns(pl.arange(0, df.height).alias("_idx"))
    strata = (
        df.group_by(["language", "endpoint_bool", "source_dataset"])
        .agg(pl.col("_idx"))
        .sort(["language", "endpoint_bool", "source_dataset"])
    )

    train_idx, val_idx, test_idx = [], [], []
    for row in strata.iter_rows(named=True):
        idxs = list(row["_idx"])
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, round(n * val_frac)) if n >= 5 and val_frac > 0 else 0
        n_test = max(1, round(n * test_frac)) if n >= 5 and test_frac > 0 else 0
        test_idx += idxs[:n_test]
        val_idx += idxs[n_test : n_test + n_val]
        train_idx += idxs[n_test + n_val :]

    return (
        df.filter(pl.col("_idx").is_in(train_idx)).drop("_idx"),
        df.filter(pl.col("_idx").is_in(val_idx)).drop("_idx"),
        df.filter(pl.col("_idx").is_in(test_idx)).drop("_idx"),
    )


# ---------------------------------------------------------------------------
# 3. Augmentation primitives
#    Each function: (waveform: np.float32[N], sample_rate: int, **kwargs) -> waveform
# ---------------------------------------------------------------------------


def insert_silence(wav: np.ndarray, sr: int, min_ms: int = 100, max_ms: int = 800, position: str = "end") -> np.ndarray:
    """Insert a pause of random length in [min_ms, max_ms].

    Rationale: real hesitation pauses in speech are exactly this -- a gap with
    no speech energy. The brief specifically calls out "natural pauses" as a
    signal the model needs to learn to distinguish from a *final* silence
    (i.e. the turn actually being over). We bias this augmentation towards
    INCOMPLETE examples in the training pipeline (see AUGMENT_FRACTION usage
    in `TurnDetectionDataset`) so the model sees plenty of "there's a pause,
    but the speaker isn't done" cases and doesn't learn the shortcut
    "silence at the end == complete".
    """
    dur_ms = random.uniform(min_ms, max_ms)
    n_silence = int(sr * dur_ms / 1000)
    silence = np.zeros(n_silence, dtype=wav.dtype)
    if position == "end":
        return np.concatenate([wav, silence])
    if position == "mid":
        cut = random.randint(int(0.3 * len(wav)), int(0.8 * len(wav)))
        return np.concatenate([wav[:cut], silence, wav[cut:]])
    raise ValueError(f"unknown position {position!r}")


def speed_perturb(wav: np.ndarray, sr: int, rate_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """Randomly stretch/compress speaking rate by up to +/-10%.

    Rationale: speaking pace varies a lot between speakers and is one of the
    prosodic cues the model should be robust to rather than overfit on --
    e.g. we don't want the model to learn "fast talkers are always mid-turn".
    """
    import librosa

    rate = random.uniform(*rate_range)
    return librosa.effects.time_stretch(wav, rate=rate)


def pitch_shift(wav: np.ndarray, sr: int, semitone_range: tuple[float, float] = (-2.0, 2.0)) -> np.ndarray:
    """Randomly shift pitch by up to +/-2 semitones.

    Rationale: trains invariance to speaker pitch (voice register), while
    +/-2 semitones is small enough to preserve the rising/falling intonation
    contour that's actually diagnostic of turn completion (a full pitch-shift
    could otherwise mask that signal).
    """
    import librosa

    n_steps = random.uniform(*semitone_range)
    return librosa.effects.pitch_shift(wav, sr=sr, n_steps=n_steps)


def volume_perturb(wav: np.ndarray, gain_db_range: tuple[float, float] = (-6.0, 6.0)) -> np.ndarray:
    """Randomly scale amplitude by +/-6dB.

    Rationale: mic distance/gain varies a lot across the dataset's different
    recording sources; the model shouldn't use raw loudness as a shortcut
    feature for anything.
    """
    gain_db = random.uniform(*gain_db_range)
    gain = 10 ** (gain_db / 20)
    return np.clip(wav * gain, -1.0, 1.0)


def add_background_noise(wav: np.ndarray, sr: int, noise_bank: list[np.ndarray], snr_db_range: tuple[float, float] = (5.0, 20.0)) -> np.ndarray:
    """Mix in background noise at a random SNR in `snr_db_range`.

    NOTE ON THE NOISE BANK: the brief asks for "Indian street/office style"
    noise specifically. We don't bundle recordings with unknown licensing.
    `load_noise_bank()` automatically uses licensed recordings placed in
    `data/noise/`, or generates a documented room/HVAC-like fallback when
    that directory is empty. We call this out explicitly rather than silently
    pretending the fallback is authentic.
    """
    noise = random.choice(noise_bank)
    if len(noise) < len(wav):
        reps = int(np.ceil(len(wav) / len(noise)))
        noise = np.tile(noise, reps)
    start = random.randint(0, len(noise) - len(wav))
    noise = noise[start : start + len(wav)]

    sig_power = np.mean(wav**2) + 1e-10
    noise_power = np.mean(noise**2) + 1e-10
    snr_db = random.uniform(*snr_db_range)
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return np.clip(wav + noise, -1.0, 1.0)


def _synthetic_noise_bank(sr: int = cfg.SAMPLE_RATE, n_clips: int = 3, duration_s: float = 6.0) -> list[np.ndarray]:
    """Generate a small bank of low-pass-filtered white noise clips as a
    stand-in "ambient room/office noise" bank. See `add_background_noise`
    docstring for why this is a placeholder rather than real recordings."""
    rng = np.random.default_rng(0)
    clips = []
    for _ in range(n_clips):
        white = rng.normal(0, 1, int(sr * duration_s)).astype(np.float32)
        # A gentle low-pass (via a short moving average) turns white noise
        # into something closer to a room/HVAC hum than harsh hiss.
        kernel = np.ones(9) / 9
        colored = np.convolve(white, kernel, mode="same")
        clips.append(colored / (np.abs(colored).max() + 1e-6) * 0.3)
    return clips


def load_noise_bank(noise_dir: Path, sr: int = cfg.SAMPLE_RATE) -> list[np.ndarray]:
    """Load user-supplied ambient recordings, falling back to synthetic room noise.

    Put licensed Indian street/office recordings in ``data/noise``. WAV, FLAC,
    OGG, and MP3 files are decoded to mono 16 kHz. Keeping acquisition outside
    this function avoids silently downloading or redistributing audio with
    unclear licensing while making authentic noise a no-code drop-in.
    """
    supported = {".wav", ".flac", ".ogg", ".mp3"}
    paths = sorted(path for path in noise_dir.glob("**/*") if path.is_file() and path.suffix.lower() in supported)
    if not paths:
        return _synthetic_noise_bank(sr=sr)

    import librosa

    clips = []
    for path in paths:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.size and np.isfinite(wav).all():
            clips.append(wav)
    if not clips:
        raise ValueError(f"no valid, non-empty ambient recordings found in {noise_dir}")
    return clips


# ---------------------------------------------------------------------------
# 4. Hinglish filler-word bank (TTS-synthesized) + splicing
# ---------------------------------------------------------------------------


class FillerBank:
    """Synthesizes (and caches on disk) short audio clips of common Hinglish
    filler/discourse words, in a mix of Hindi and Indian-English voices, via
    Microsoft Edge's free neural TTS (`edge-tts`). These get spliced into
    real clips by `inject_filler` to teach robustness to code-switched
    discourse markers. Injection preserves the source endpoint label; filler
    presence alone is not a reliable completion label.
    """

    def __init__(self, cache_dir: Path, words: list[str] | None = None, voices: list[str] | None = None) -> None:
        """Create filler cache backed by requested words and TTS voices."""
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.words = words or cfg.HINGLISH_FILLERS
        self.voices = voices or cfg.TTS_VOICES
        self._clips: dict[tuple[str, str], np.ndarray] = {}

    def _clip_path(self, word: str, voice: str) -> Path:
        safe_word = word.replace(" ", "_")
        return self.cache_dir / f"{voice}__{safe_word}.wav"

    async def _synthesize_one(self, word: str, voice: str) -> None:
        import edge_tts

        path = self._clip_path(word, voice)
        if path.exists():
            return
        communicate = edge_tts.Communicate(word, voice)
        # edge-tts streams mp3 chunks; write raw then let soundfile/librosa
        # transcode on load (librosa.load handles mp3 via audioread/ffmpeg).
        mp3_path = path.with_suffix(".mp3")
        # Clips are tiny and edge-tts already controls network concurrency;
        # another async-file dependency would add complexity without benefit.
        with open(mp3_path, "wb") as f:  # noqa: ASYNC230
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])

        import librosa

        wav, _ = librosa.load(mp3_path, sr=cfg.SAMPLE_RATE, mono=True)
        # Trim leading/trailing near-silence so splicing doesn't leave an
        # awkward gap before/after the actual filler word.
        wav, _ = librosa.effects.trim(wav, top_db=30)
        sf.write(path, wav, cfg.SAMPLE_RATE)
        mp3_path.unlink(missing_ok=True)

    def build(self) -> None:
        """Synthesize every (word, voice) pair not already cached. Run once;
        subsequent calls are near-instant since everything is cached on disk."""
        pending = [(w, v) for w in self.words for v in self.voices if not self._clip_path(w, v).exists()]
        if not pending:
            return

        async def _run() -> None:
            # A small concurrency cap is polite to the free TTS endpoint and
            # avoids opening dozens of sockets at once.
            sem = asyncio.Semaphore(4)

            async def _bounded(w: str, v: str) -> None:
                async with sem:
                    await self._synthesize_one(w, v)

            await asyncio.gather(*[_bounded(w, v) for w, v in pending])

        asyncio.run(_run())

    def sample(self) -> np.ndarray:
        """Return one random cached filler clip's waveform."""
        word, voice = random.choice(self.words), random.choice(self.voices)
        key = (word, voice)
        if key not in self._clips:
            path = self._clip_path(word, voice)
            wav, _ = sf.read(path, dtype="float32")
            self._clips[key] = wav
        return self._clips[key]


def inject_filler(wav: np.ndarray, sr: int, filler_bank: FillerBank, position: str = "end") -> np.ndarray:
    """Splice a filler-word clip (+ a short pause) into `wav`.

    position="end":  truncate the clip a bit early, then append
                      [pause] + [filler word] + [short trailing pause]. This is
                      a weak-label synthesis primitive; callers must decide
                      the target label and account for possible ambiguity.

    position="mid":   splice [pause] + [filler word] + [pause] into the
                      *middle* of the clip, keeping the original ending
                      intact. This changes acoustics but does NOT change the
                      original turn-completion label.

    A short ~15ms linear crossfade at each splice point avoids audible clicks
    from discontinuities where two independently-recorded/synthesized clips
    meet.
    """
    filler = filler_bank.sample()
    pre_pause = insert_silence(np.zeros(0, dtype=np.float32), sr, 80, 250, "end")
    post_pause = insert_silence(np.zeros(0, dtype=np.float32), sr, 150, 500, "end")
    snippet = np.concatenate([pre_pause, filler, post_pause])

    def _crossfade_concat(a: np.ndarray, b: np.ndarray, fade_ms: float = 15) -> np.ndarray:
        n = min(int(sr * fade_ms / 1000), len(a), len(b))
        if n <= 0:
            return np.concatenate([a, b])
        fade_out = np.linspace(1, 0, n, dtype=np.float32)
        fade_in = np.linspace(0, 1, n, dtype=np.float32)
        mixed = a[-n:] * fade_out + b[:n] * fade_in
        return np.concatenate([a[:-n], mixed, b[n:]])

    if position == "end":
        cut = random.randint(int(0.5 * len(wav)), max(int(0.5 * len(wav)) + 1, int(0.85 * len(wav))))
        return _crossfade_concat(wav[:cut], snippet)
    if position == "mid":
        cut = random.randint(int(0.3 * len(wav)), int(0.7 * len(wav)))
        head = _crossfade_concat(wav[:cut], snippet)
        return _crossfade_concat(head, wav[cut:])
    raise ValueError(f"unknown position {position!r}")


# ---------------------------------------------------------------------------
# 5. torch Dataset
# ---------------------------------------------------------------------------


@dataclass
class AugmentConfig:
    """Toggles + probabilities for each augmentation, so experiments can
    switch individual augmentations on/off (see docs/02_experiment_plan.md,
    e.g. the "silence length" and "audio-only vs multimodal" ablations)."""

    enabled: bool = True
    p_silence: float = 0.3
    p_speed: float = 0.3
    p_pitch: float = 0.2
    p_noise: float = 0.2
    p_volume: float = 0.3
    p_filler: float = 0.35  # matches cfg.AUGMENT_FRACTION by default
    silence_ms_range: tuple[int, int] = (100, 800)

    def __post_init__(self) -> None:
        for name in ("p_silence", "p_speed", "p_pitch", "p_noise", "p_volume", "p_filler"):
            probability = getattr(self, name)
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {probability}")
        min_ms, max_ms = self.silence_ms_range
        if min_ms < 0 or max_ms < min_ms:
            raise ValueError("silence_ms_range must be non-negative and ordered [min, max]")


class TurnDetectionDataset:
    """Returns raw (waveform, label) pairs -- see module docstring for why
    feature extraction is deliberately kept out of this class.

    Each item is a dict:
        {
            "waveform": np.float32[N] mono 16kHz audio, N <= 8 * 16000,
            "label": int, 1 = turn complete, 0 = incomplete,
            "id": str, "language": str, "midfiller": bool, "endfiller": bool,
            "synthetic": bool, "is_augmented": bool,
        }
    """

    def __init__(
        self,
        meta_df: pl.DataFrame,
        root: Path = cfg.ROOT,
        augment: AugmentConfig | None = None,
        filler_bank: FillerBank | None = None,
        noise_bank: list[np.ndarray] | None = None,
        include_pause_features: bool = False,
    ) -> None:
        """
        Args:
            meta_df: metadata table from `stream_filtered_subset` (or a
                split of it from `stratified_split`). Its `path` column
                holds paths relative to the project root.
            root: the project root those paths are relative to. Defaults to
                `configs.config.ROOT` -- only override this in tests.
        """
        self.meta = meta_df
        self.root = root
        self.augment = augment or AugmentConfig(enabled=False)
        self.filler_bank = filler_bank
        self.include_pause_features = include_pause_features
        if noise_bank is not None:
            if self.augment.enabled and self.augment.p_noise > 0 and not noise_bank:
                raise ValueError("noise_bank cannot be empty when noise augmentation is enabled")
            self.noise_bank = noise_bank
        elif self.augment.enabled and self.augment.p_noise > 0:
            self.noise_bank = load_noise_bank(cfg.DATA_DIR / "noise")
        else:
            self.noise_bank = []

    def __len__(self) -> int:
        return self.meta.height

    def _load_wav(self, rel_path: str) -> np.ndarray:
        wav, sr = sf.read(self.root / rel_path, dtype="float32")
        if sr != cfg.SAMPLE_RATE:
            raise ValueError(f"expected {cfg.SAMPLE_RATE}Hz, got {sr}Hz for {rel_path}")
        if wav.ndim > 1:  # collapse to mono just in case
            wav = wav.mean(axis=1)
        if wav.size == 0:
            raise ValueError(f"empty waveform in {rel_path}")
        if not np.isfinite(wav).all():
            raise ValueError(f"non-finite waveform in {rel_path}")
        return np.asarray(wav, dtype=np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.meta.row(idx, named=True)
        wav = self._load_wav(row["path"])
        original_duration_s = row.get("duration_s", len(wav) / cfg.SAMPLE_RATE)
        label = int(row["endpoint_bool"])
        midfiller = bool(row["midfiller"])
        endfiller = bool(row["endfiller"])
        augment_types: list[str] = []

        # Crop before augmentation as well as after it. Otherwise a pause
        # inserted in the middle of a long clip can fall outside the final
        # 8-second window and have no training effect despite being marked as
        # augmented. Turn-completion cues live at the end, so keep that end.
        if len(wav) > cfg.MAX_REAL_SAMPLES:
            wav = wav[-cfg.MAX_REAL_SAMPLES :]

        if self.augment.enabled:
            a = self.augment
            # Insert fillers into both classes and preserve the endpoint label.
            # This prevents a synthetic TTS voice from becoming a shortcut for
            # "incomplete" and avoids false label flips for completion-capable
            # markers such as "bas", "haan", and "wait".
            if self.filler_bank is not None and random.random() < a.p_filler:
                wav = inject_filler(wav, cfg.SAMPLE_RATE, self.filler_bank, position="mid")
                midfiller = True
                augment_types.append("mid_filler")

            # Incomplete turns get 1.5x more pause augmentation because
            # hesitation is the expensive false-complete failure mode named
            # in the challenge. Complete turns still receive pauses so the
            # model cannot learn "any silence means incomplete."
            silence_probability = min(1.0, a.p_silence * (1.5 if label == 0 else 1.0))
            if random.random() < silence_probability:
                # Silence insertion is applied regardless of label (a pause
                # can happen whether or not the speaker is actually done);
                # position="mid" for incomplete rows (a pause *within* an
                # unfinished thought), "end" for complete rows (trailing
                # silence after finishing, which should NOT flip the label --
                # a complete utterance followed by silence is still complete).
                position = "end" if label == 1 else random.choice(["mid", "end"])
                wav = insert_silence(wav, cfg.SAMPLE_RATE, *a.silence_ms_range, position=position)
                augment_types.append(f"{position}_silence")

            if random.random() < a.p_speed:
                wav = speed_perturb(wav, cfg.SAMPLE_RATE)
                augment_types.append("speed")
            if random.random() < a.p_pitch:
                wav = pitch_shift(wav, cfg.SAMPLE_RATE)
                augment_types.append("pitch")
            if random.random() < a.p_noise:
                wav = add_background_noise(wav, cfg.SAMPLE_RATE, self.noise_bank)
                augment_types.append("noise")
            if random.random() < a.p_volume:
                wav = volume_perturb(wav)
                augment_types.append("volume")

        # Enforce the 8s cap on REAL audio content (from the *end* of the
        # clip -- turn-completion cues live at the end, so if we have to
        # drop audio we drop from the front). collate_fn later left-pads
        # this to the same 8-second window used by Pipecat Smart Turn v3.2.
        if len(wav) > cfg.MAX_REAL_SAMPLES:
            wav = wav[-cfg.MAX_REAL_SAMPLES :]

        item = {
            "waveform": wav.astype(np.float32),
            "label": label,
            "id": row["id"],
            "language": row["language"],
            "midfiller": midfiller,
            "endfiller": endfiller,
            "midfiller_annotation_known": row.get("midfiller_annotation_known"),
            "endfiller_annotation_known": row.get("endfiller_annotation_known"),
            "synthetic": row["synthetic"],
            "original_duration_s": original_duration_s,
            "duration_s": len(wav) / cfg.SAMPLE_RATE,
            "is_augmented": bool(augment_types),
            "augment_types": tuple(augment_types),
            "transcript": row.get("transcript"),
        }
        if self.include_pause_features:
            item.update(measure_pause_features(wav, cfg.SAMPLE_RATE))
        return item


def build_hard_negative_indices(meta_df: pl.DataFrame, short_s: float = 1.5, long_s: float = 4.0) -> np.ndarray:
    """Return row indices for a "hard negative" oversampling set: SHORT
    complete utterances paired against LONG incomplete-with-filler utterances.

    WHY: without this, a lazy model can shortcut the whole task by learning
    "short clip -> probably complete, long clip -> probably incomplete"
    (plausible on average, since fillers/hesitations tend to run long) instead
    of actually listening to prosody. These rows are exactly the
    counterexamples to that shortcut -- short-but-done and long-but-not-done --
    so oversampling them during training forces the model to rely on real
    acoustic cues. Returned indices are meant to be fed into a
    `WeightedRandomSampler` (see train.py) with elevated weight, not used to
    physically duplicate audio files on disk.
    """
    hard_mask = (
        (pl.col("endpoint_bool") & (pl.col("duration_s") <= short_s))
        | (
            (~pl.col("endpoint_bool"))
            & (pl.col("duration_s") >= long_s)
            & (pl.col("midfiller") | pl.col("endfiller"))
        )
    )
    # Derive positions directly instead of mapping through IDs. This remains
    # correct even if an upstream dataset contains duplicate IDs.
    return np.flatnonzero(meta_df.select(hard_mask.alias("hard"))["hard"].to_numpy())


# ---------------------------------------------------------------------------
# 6. Collation: raw waveforms -> padded log-mel features for the model
# ---------------------------------------------------------------------------


def collate_fn(
    batch: list[dict[str, Any]],
    feature_extractor: Callable[..., Any],
    tokenizer: Callable[..., Any] | None = None,
    max_text_length: int = 64,
) -> dict[str, Any]:
    """Turn a list of `TurnDetectionDataset` items into a model-ready batch.

    Padding convention (LEFT-padding, matching Pipecat's own Smart Turn
    convention): every waveform gets zero-padded to the 8-second model
    window, with padding *prepended* so real audio always ends at the same
    position regardless of clip length. Left-padding matters because our pooling heads and
    the task itself both care most about *the end* of the clip -- that's
    where completion/pause cues live -- so keeping "the end of the input"
    and "the end of the speech" aligned across a batch means the model
    doesn't have to separately learn where the real ending sits for each
    example.
    """
    import torch

    waveforms, masks = [], []
    for item in batch:
        wav = item["waveform"]
        if len(wav) == 0:
            raise ValueError("cannot collate an empty waveform")
        pad = cfg.WHISPER_WINDOW_SAMPLES - len(wav)
        if pad > 0:
            wav = np.concatenate([np.zeros(pad, dtype=np.float32), wav])
        else:
            wav = wav[-cfg.WHISPER_WINDOW_SAMPLES :]  # defensive; should never trigger given the 8s cap upstream
        waveforms.append(wav)

        # One encoder position represents 320 waveform samples (two 160-sample
        # mel hops). Build this directly at encoder resolution. Sampling a
        # waveform-level mask at fixed offsets can mark every position as
        # padding for valid clips shorter than one hop, producing NaNs in
        # masked attention pooling.
        real_hidden_positions = max(1, (min(len(item["waveform"]), cfg.WHISPER_WINDOW_SAMPLES) + 319) // 320)
        mask = np.zeros(cfg.WHISPER_ENCODER_HIDDEN_LEN, dtype=np.float32)
        mask[-real_hidden_positions:] = 1.0
        masks.append(mask)

    features = feature_extractor(
        waveforms,
        sampling_rate=cfg.SAMPLE_RATE,
        return_tensors="pt",
        return_attention_mask=False,
        padding="max_length",
        max_length=cfg.WHISPER_WINDOW_SAMPLES,
        truncation=True,
        do_normalize=True,
    )
    input_features = features["input_features"]  # (batch, 80, 800)

    frame_masks = torch.from_numpy(np.stack(masks))

    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)
    result = {
        "input_features": input_features,
        "attention_mask": frame_masks,
        "labels": labels,
        "meta": [{k: v for k, v in item.items() if k != "waveform"} for item in batch],
    }
    if tokenizer is not None:
        texts = [str(item.get("transcript") or "") for item in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_text_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        result["input_ids"] = encoded["input_ids"]
        result["text_attention_mask"] = encoded["attention_mask"]
    return result
