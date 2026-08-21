"""Gradio demo for Hinglish turn-completion detection.

Run locally with ``python app.py``. Override checkpoint with
``--checkpoint path/to/best.pt`` or ``TURN_DETECTOR_CHECKPOINT``.
"""

from __future__ import annotations

import argparse
import os
import threading
from pathlib import Path

if os.environ.get("SPACES_ZERO_GPU") == "1":
    import spaces
else:
    class _LocalSpaces:
        """No-op compatibility shim for local runs outside Hugging Face Spaces."""

        @staticmethod
        def GPU(duration: int):
            """Return an identity decorator matching ``spaces.GPU``."""
            del duration
            return lambda function: function

    spaces = _LocalSpaces()

import gradio as gr
import numpy as np
from matplotlib.figure import Figure

from configs import config as cfg
from src.inference import TurnDetector

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "checkpoints" / "baseline_attention_augmented" / "best.pt",
    PROJECT_ROOT / "experiments" / "E1_no_augmentation" / "checkpoints" / "best.pt",
)

_detector: TurnDetector | None = None
_detector_path: Path | None = None
_detector_lock = threading.Lock()


def resolve_checkpoint(explicit_path: str | Path | None = None) -> Path | None:
    """Resolve explicit/env checkpoint, then stable project defaults."""
    configured = explicit_path or os.environ.get("TURN_DETECTOR_CHECKPOINT")
    if configured:
        path = Path(configured).expanduser()
        return (PROJECT_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    return next((path for path in DEFAULT_CHECKPOINTS if path.is_file()), None)


CHECKPOINT_PATH = resolve_checkpoint()
EXAMPLE_PATHS = sorted(SAMPLE_DIR.glob("*.wav"))[:8] if SAMPLE_DIR.exists() else []

if os.environ.get("SPACES_ZERO_GPU") == "1" and CHECKPOINT_PATH is not None:
    _detector = TurnDetector(CHECKPOINT_PATH, device="cuda")
    _detector_path = CHECKPOINT_PATH


def checkpoint_label(checkpoint_path: Path | None) -> str:
    """Return UI-safe checkpoint label without exposing external cache paths."""
    if checkpoint_path is None:
        return "**No checkpoint found. Train model or pass `--checkpoint`.**"
    if checkpoint_path.is_relative_to(PROJECT_ROOT):
        return f"Checkpoint: `{checkpoint_path.relative_to(PROJECT_ROOT)}`"
    return f"Checkpoint: `{checkpoint_path.name}`"


def get_detector(checkpoint_path: str | Path | None = None) -> TurnDetector:
    """Load model once; safe when multiple UI requests arrive together."""
    global _detector, _detector_path
    resolved = resolve_checkpoint(checkpoint_path)
    if resolved is None:
        raise FileNotFoundError(
            "No trained checkpoint found. Run training or pass --checkpoint path/to/best.pt."
        )
    with _detector_lock:
        if _detector is None or _detector_path != resolved:
            _detector = TurnDetector(resolved)
            _detector_path = resolved
    return _detector


def _plot_waveform(wav: np.ndarray, sample_rate: int) -> Figure:
    figure = Figure(figsize=(7, 2.2), layout="tight")
    axis = figure.subplots()
    duration = len(wav) / sample_rate
    times = np.arange(len(wav), dtype=np.float32) / sample_rate
    axis.plot(times, wav, linewidth=0.6, color="#2563EB")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(0, max(duration, 0.1))
    axis.grid(alpha=0.2)
    return figure


def _format_prediction(result: dict, device: str) -> tuple[str, str]:
    probability = float(result["prob_complete"])
    complete = result["decision"] == "complete"
    headline = "## User finished speaking" if complete else "## User is still speaking / pausing"
    decision_confidence = probability if complete else 1 - probability
    details = (
        f"Complete probability: {probability:.1%}\n\n"
        f"Decision confidence: {decision_confidence:.1%}\n\n"
        f"End-to-end latency: {result['latency_ms']:.1f} ms on {device}"
    )
    return headline, details


@spaces.GPU(duration=30)
def predict(audio: tuple[int, np.ndarray] | None) -> tuple[Figure, str, str]:
    """Gradio callback. Audio tuple order is (sample_rate, waveform)."""
    if audio is None:
        raise gr.Error("Upload audio or record from microphone first.")
    try:
        sample_rate, waveform = audio
        detector = get_detector()
        prepared = detector.prepare_audio((np.asarray(waveform), sample_rate))
        result = detector.predict(prepared)
    except (FileNotFoundError, TypeError, ValueError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    headline, details = _format_prediction(result, detector.device)
    return _plot_waveform(prepared, cfg.SAMPLE_RATE), headline, details


def build_demo() -> gr.Blocks:
    """Build queued Gradio interface without loading model at import time."""
    checkpoint_status = checkpoint_label(CHECKPOINT_PATH)

    with gr.Blocks(title="Hinglish Turn Detection") as demo:
        gr.Markdown(
            "# Hinglish Turn Detection\n"
            "Upload or record end of utterance. Model decides whether speaker finished or is pausing.\n\n"
            f"Whisper-tiny encoder + attention pooling. {checkpoint_status}"
        )
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(
                    sources=["upload", "microphone"],
                    type="numpy",
                    label="Upload or record audio",
                )
                submit = gr.Button("Detect turn completion", variant="primary")
            with gr.Column():
                waveform = gr.Plot(label="Model input waveform (last 8 seconds, 16 kHz)")
                decision = gr.Markdown()
                details = gr.Textbox(label="Confidence and latency", interactive=False)

        submit.click(
            fn=predict,
            inputs=audio_input,
            outputs=[waveform, decision, details],
            api_name="predict",
        )

        if EXAMPLE_PATHS:
            gr.Examples(
                examples=[[str(path)] for path in EXAMPLE_PATHS],
                inputs=audio_input,
                label="Example Hindi/English clips",
            )
        else:
            gr.Markdown("_No local examples found under `data/samples/`._")

    return demo.queue(max_size=32, default_concurrency_limit=1)


demo = build_demo()


def main() -> None:
    """Load selected checkpoint and launch local Gradio server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Checkpoint path (overrides default and environment)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    global CHECKPOINT_PATH
    CHECKPOINT_PATH = resolve_checkpoint(args.checkpoint)
    get_detector(CHECKPOINT_PATH)  # fail before opening server; avoids first-request cold load
    runtime_demo = build_demo()
    runtime_demo.launch(server_name=args.host, server_port=args.port, show_error=True)


if __name__ == "__main__":
    main()
