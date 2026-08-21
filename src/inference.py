"""
Production-facing inference wrapper around a trained TurnDetectionModel
checkpoint. This is the module both `app.py` (the Gradio demo) and anyone
integrating the model into a real voice pipeline should import -- it's the
one place that knows how to go from "raw audio, in whatever form it arrives"
to "P(turn complete) + a yes/no decision", so that logic never has to be
duplicated or subtly reimplemented differently in the demo vs. a batch eval
script vs. someone else's integration code.
"""

from __future__ import annotations

import sys
import threading
import time
from numbers import Real
from pathlib import Path

import numpy as np
import torch
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.dataset import collate_fn
from src.models import build_model

AudioInput = str | Path | np.ndarray | tuple[np.ndarray, int]


def _validate_threshold(threshold: float) -> float:
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("threshold must be a real number between 0 and 1")
    value = float(threshold)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("threshold must be between 0 and 1")
    return value


def _normalize_waveform(wav: np.ndarray) -> np.ndarray:
    """Convert PCM/float audio to finite mono float32 without hiding bad input."""
    array = np.asarray(wav)
    if array.ndim == 2:
        array = array.mean(axis=1)
    elif array.ndim != 1:
        raise ValueError(f"audio must be mono or stereo, got shape {array.shape}")
    if array.size == 0:
        raise ValueError("audio waveform is empty")

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        array = array.astype(np.float32) / scale
    else:
        array = array.astype(np.float32)

    if not np.isfinite(array).all():
        raise ValueError("audio waveform contains NaN or infinite values")
    return array


def _load_and_resample(audio: AudioInput) -> np.ndarray:
    """Normalize any of the accepted input forms down to a mono float32
    array at our target 16kHz sample rate.

    Accepted forms (per the brief's "accept raw audio (wav/mp3) or numpy
    array" requirement):
      - a path (str/Path) to a .wav/.mp3/etc file -- decoded via librosa,
        which handles both directly.
      - a bare numpy array -- ASSUMED to already be 16kHz mono (this is the
        common case for a live pipeline that already resampled once
        upstream); passing a (array, sample_rate) tuple instead is safer if
        you're not sure, since we'll resample explicitly in that case.
      - a (array, sample_rate) tuple, e.g. straight from `gr.Audio`'s
        numpy-mode output in app.py -- resampled to 16kHz if needed.
    """
    import librosa

    if isinstance(audio, (str, Path)):
        audio_path = Path(audio).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        wav, _ = librosa.load(str(audio_path), sr=cfg.SAMPLE_RATE, mono=True)
        return _normalize_waveform(wav)

    if isinstance(audio, tuple):
        wav, sr = audio
        if not isinstance(sr, (int, np.integer)) or sr <= 0:
            raise ValueError(f"sample rate must be a positive integer, got {sr!r}")
        wav = _normalize_waveform(wav)
        if sr != cfg.SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=cfg.SAMPLE_RATE)
        return np.asarray(wav, dtype=np.float32)

    return _normalize_waveform(audio)


class TurnDetector:
    """Loads a trained checkpoint once, then serves predictions.

    Usage:
        detector = TurnDetector("checkpoints/baseline_attention_augmented/best.pt")
        result = detector.predict("some_clip.wav")
        # {"prob_complete": 0.83, "decision": "complete", "latency_ms": 14.2}
    """

    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        """Load validated checkpoint onto requested device and prepare feature extraction."""
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested, but CUDA is not available")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        required_keys = {"config", "model_state_dict"}
        missing_keys = required_keys - ckpt.keys()
        if missing_keys:
            raise ValueError(f"checkpoint is missing required keys: {sorted(missing_keys)}")
        self.checkpoint_path = checkpoint_path
        self.config = ckpt["config"]
        self.decision_threshold = _validate_threshold(ckpt.get("decision_threshold", 0.5))
        self.multimodal = bool(self.config["model"].get("multimodal", False))

        self.model = build_model(
            base_model_name=self.config["model"]["base_model_name"],
            freeze_encoder_layers=self.config["model"]["freeze_encoder_layers"],
            pooling=self.config["model"]["pooling"],
            multimodal=self.multimodal,
            text_vocab_size=self.config["model"].get("text_vocab_size", 51_865),
            text_embedding_dim=self.config["model"].get("text_embedding_dim", 64),
            text_pad_token_id=self.config["model"].get("text_pad_token_id", 50_257),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            self.config["model"]["base_model_name"], chunk_length=cfg.MAX_DURATION_S
        )
        self.asr_processor = None
        self.asr_model = None
        if self.multimodal:
            self.asr_processor = WhisperProcessor.from_pretrained(
                self.config["model"]["base_model_name"]
            )
            self.asr_model = WhisperForConditionalGeneration.from_pretrained(
                self.config["model"]["base_model_name"]
            ).to(self.device).eval()
            if str(self.device).startswith("cuda"):
                self.asr_model.half()
        self._inference_lock = threading.Lock()

    @staticmethod
    def prepare_audio(audio: AudioInput) -> np.ndarray:
        """Return exact mono 16 kHz waveform consumed by model."""
        wav = _load_and_resample(audio)
        return wav[-cfg.MAX_REAL_SAMPLES :] if len(wav) > cfg.MAX_REAL_SAMPLES else wav

    def _to_batch(self, audios: list[AudioInput]) -> dict:
        waveforms = [self.prepare_audio(audio) for audio in audios]
        transcripts = self._transcribe(waveforms) if self.multimodal else [None] * len(waveforms)
        items = [
            {"waveform": waveform, "label": 0, "transcript": transcript}
            for waveform, transcript in zip(waveforms, transcripts, strict=True)
        ]
        tokenizer = self.asr_processor.tokenizer if self.asr_processor is not None else None
        return collate_fn(items, self.feature_extractor, tokenizer=tokenizer)

    def _transcribe(self, waveforms: list[np.ndarray]) -> list[str]:
        """Generate label-independent transcripts for multimodal inference."""
        if self.asr_processor is None or self.asr_model is None:
            raise RuntimeError("multimodal ASR components are not loaded")
        inputs = self.asr_processor(
            waveforms,
            sampling_rate=cfg.SAMPLE_RATE,
            return_attention_mask=True,
            return_tensors="pt",
        )
        dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32
        token_ids = self.asr_model.generate(
            inputs["input_features"].to(self.device, dtype=dtype),
            attention_mask=inputs["attention_mask"].to(self.device),
            task="transcribe",
            max_new_tokens=self.config["model"].get("asr_max_new_tokens", 64),
        )
        return [
            text.strip()
            for text in self.asr_processor.batch_decode(
                token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        ]

    def _forward(self, batch: dict) -> torch.Tensor:
        model_kwargs = {}
        if self.multimodal:
            model_kwargs = {
                "input_ids": batch["input_ids"].to(self.device),
                "text_attention_mask": batch["text_attention_mask"].to(self.device),
            }
        return self.model(
            batch["input_features"].to(self.device),
            batch["attention_mask"].to(self.device),
            **model_kwargs,
        )

    def _synchronize(self) -> None:
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def predict(self, audio: AudioInput, threshold: float | None = None) -> dict:
        """Predict for a single audio clip. Returns probability of
        "complete", the thresholded decision, and measured latency."""
        threshold = self.decision_threshold if threshold is None else _validate_threshold(threshold)
        with self._inference_lock:
            self._synchronize()
            t0 = time.perf_counter()
            batch = self._to_batch([audio])
            logits = self._forward(batch)
            self._synchronize()
            prob_complete = torch.sigmoid(logits).item()
            latency_ms = (time.perf_counter() - t0) * 1000

        result = {
            "prob_complete": prob_complete,
            "decision": "complete" if prob_complete >= threshold else "incomplete",
            "latency_ms": latency_ms,
            "threshold": threshold,
        }
        if self.multimodal:
            result["transcript"] = batch["meta"][0]["transcript"]
        return result

    @torch.inference_mode()
    def predict_batch(
        self, audios: list[AudioInput], threshold: float | None = None
    ) -> list[dict]:
        """Predict for a list of clips in one forward pass -- for offline/
        bulk evaluation, not the live single-utterance path (see
        evaluate.measure_latency's docstring for why batch-1 is what
        actually matters in production)."""
        if not audios:
            raise ValueError("audios must contain at least one clip")
        threshold = self.decision_threshold if threshold is None else _validate_threshold(threshold)
        with self._inference_lock:
            self._synchronize()
            t0 = time.perf_counter()
            batch = self._to_batch(audios)
            logits = self._forward(batch)
            self._synchronize()
            probs = torch.sigmoid(logits).float().cpu().numpy().reshape(-1)
            total_latency_ms = (time.perf_counter() - t0) * 1000

        results = [
            {
                "prob_complete": float(p),
                "decision": "complete" if p >= threshold else "incomplete",
                "latency_ms": total_latency_ms / len(audios),  # amortized per-item, not truly independent
                "threshold": threshold,
            }
            for p in probs
        ]
        if self.multimodal:
            for result, metadata in zip(results, batch["meta"], strict=True):
                result["transcript"] = metadata["transcript"]
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("audio_path")
    parser.add_argument("--threshold", type=float, help="override checkpoint decision threshold")
    args = parser.parse_args()

    detector = TurnDetector(args.checkpoint)
    result = detector.predict(args.audio_path, threshold=args.threshold)
    print(result)
