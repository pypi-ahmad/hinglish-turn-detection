"""Cache frozen Whisper transcripts beside existing split metadata.

No turn labels enter ASR generation. Outputs preserve row order and source
columns, add ``transcript`` and ``transcript_model``, and support resume.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import soundfile as sf
import torch
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg


def _load_audio(path: Path) -> np.ndarray:
    waveform, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != cfg.SAMPLE_RATE:
        raise ValueError(f"expected {cfg.SAMPLE_RATE}Hz audio, got {sample_rate}Hz: {path}")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return np.asarray(waveform[-cfg.MAX_REAL_SAMPLES :], dtype=np.float32)


def _write_progress(
    metadata: pl.DataFrame,
    transcripts: list[str | None],
    output_path: Path,
    model_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = metadata.with_columns(
        pl.Series("transcript", transcripts, dtype=pl.String),
        pl.lit(model_name).alias("transcript_model"),
    )
    temporary = output_path.with_suffix(".tmp.parquet")
    frame.write_parquet(temporary)
    temporary.replace(output_path)


def _resume_transcripts(
    metadata: pl.DataFrame, output_path: Path, model_name: str
) -> list[str | None]:
    if not output_path.exists():
        return [None] * metadata.height
    cached = pl.read_parquet(output_path)
    if cached.height != metadata.height or cached["id"].to_list() != metadata["id"].to_list():
        raise ValueError(f"resume output rows do not match input metadata: {output_path}")
    if "transcript" not in cached.columns:
        raise ValueError(f"resume output has no transcript column: {output_path}")
    cached_models = cached["transcript_model"].unique().to_list() if "transcript_model" in cached.columns else []
    if cached_models != [model_name]:
        raise ValueError(
            f"resume output model {cached_models!r} does not match requested model {model_name!r}"
        )
    return cached["transcript"].to_list()


@torch.inference_mode()
def transcribe_metadata(
    input_path: Path,
    output_path: Path,
    processor: WhisperProcessor,
    model: WhisperForConditionalGeneration,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    checkpoint_every: int = 25,
) -> dict[str, float | int | str]:
    """Transcribe one metadata split, resuming completed rows when available."""
    if input_path.resolve() == output_path.resolve():
        raise ValueError("transcript output must not overwrite source metadata")
    metadata = pl.read_parquet(input_path)
    transcripts = _resume_transcripts(metadata, output_path, model.name_or_path)
    pending = [index for index, text in enumerate(transcripts) if text is None]
    generation_seconds = 0.0
    started = time.perf_counter()

    for batch_number, start in enumerate(
        tqdm(range(0, len(pending), batch_size), desc=input_path.stem, unit="batch"), start=1
    ):
        indices = pending[start : start + batch_size]
        waveforms = [_load_audio(cfg.ROOT / metadata[index, "path"]) for index in indices]
        inputs = processor(
            waveforms,
            sampling_rate=cfg.SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_features = inputs["input_features"].to(
            device, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        attention_mask = inputs["attention_mask"].to(device)
        generation_started = time.perf_counter()
        token_ids = model.generate(
            input_features,
            attention_mask=attention_mask,
            task="transcribe",
            max_new_tokens=max_new_tokens,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        generation_seconds += time.perf_counter() - generation_started
        texts = processor.batch_decode(
            token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for index, text in zip(indices, texts, strict=True):
            transcripts[index] = text.strip()
        if batch_number % checkpoint_every == 0:
            _write_progress(metadata, transcripts, output_path, model.name_or_path)

    _write_progress(metadata, transcripts, output_path, model.name_or_path)
    completed = sum(text is not None for text in transcripts)
    empty = sum(text == "" for text in transcripts)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "rows": metadata.height,
        "newly_transcribed": len(pending),
        "completed": completed,
        "empty_transcripts": empty,
        "generation_seconds": generation_seconds,
        "wall_seconds": time.perf_counter() - started,
        "amortized_generation_ms_per_clip": 1000 * generation_seconds / max(1, len(pending)),
    }


def main() -> None:
    """Generate or resume Whisper transcripts for prepared dataset splits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--splits", nargs="+", default=("train_split", "val_split", "test_split")
    )
    parser.add_argument("--output-dir", type=Path, default=cfg.SUBSET_DIR / "transcribed")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch size and max new tokens must be positive")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(device).eval()
    if device == "cuda":
        model.half()

    summary_path = args.output_dir / "transcription_summary.json"
    previous_reports = {}
    if summary_path.exists():
        previous_summary = json.loads(summary_path.read_text())
        previous_reports = {item["output"]: item for item in previous_summary.get("splits", [])}

    reports = []
    for split in args.splits:
        report = transcribe_metadata(
            cfg.SUBSET_DIR / f"{split}.parquet",
            args.output_dir / f"{split}.parquet",
            processor,
            model,
            device,
            args.batch_size,
            args.max_new_tokens,
        )
        if report["newly_transcribed"] == 0 and report["output"] in previous_reports:
            report = {**previous_reports[report["output"]], "resumed_without_work": True}
        reports.append(report)
    summary_path.write_text(json.dumps({"model": args.model, "device": device, "splits": reports}, indent=2))
    print(json.dumps({"summary": str(summary_path), "splits": reports}, indent=2))


if __name__ == "__main__":
    main()
