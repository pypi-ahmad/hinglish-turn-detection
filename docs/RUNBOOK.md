<!-- markdownlint-disable MD013 -->

# Runbook

Operational notes for running this repository locally. This is a research/
submission codebase, not a deployed service: there is no CI configuration
(`.github/` does not exist in this tree), no Dockerfile or compose file, and
no process manager. Everything below is a foreground command you run
yourself.

## Start

### Demo (Gradio UI)

```powershell
uv sync
uv run python app.py
```

Loads a checkpoint (first existing of `checkpoints/safety_finalist/best.pt`,
`checkpoints/baseline_attention_augmented/best.pt`, or
`experiments/E1_no_augmentation/checkpoints/best.pt`, unless overridden; see
`resolve_checkpoint` in `app.py`) and serves on `0.0.0.0:7860` by default.
Open `http://127.0.0.1:7860` in a browser. Override with `--checkpoint`,
`--host`, `--port`, or the `TURN_DETECTOR_CHECKPOINT` environment variable.

`app.py main()` calls `get_detector()` before starting the Gradio server
specifically so a missing/bad checkpoint fails fast in the terminal instead
of on the first UI request.

### CLI inference on one file

```powershell
uv run python src/inference.py checkpoints/safety_finalist/best.pt path/to/audio.wav
```

### Training / evaluation / experiment sweeps

See the root [README](../README.md#training-and-evaluation) for the verified
command sequence (`scripts/prepare_data.py` → `src/train.py` →
`src/evaluate.py` → `scripts/run_experiments.py` /
`scripts/select_safety_finalist.py`). Those commands were re-checked against
the current `argparse` definitions in each script while writing this
document.

## Stop

Every entry point above is a foreground process (`app.py`'s Gradio server,
or a one-shot training/eval script). `Ctrl+C` in the terminal it's running in
is the only stop mechanism; there is no daemon, systemd unit, or background
service to manage.

## Logs

There is no log file or logging framework configured anywhere in `src/`,
`scripts/`, or `app.py`. All diagnostic output is `print()`/`tqdm` progress
bars to stdout/stderr in the terminal that launched the process. Training
also writes machine-readable run state to `experiments/<name>/history.json`
and `metrics.json`, which is the closest thing to a persistent log for a
training run.

## Common failures (inferable from the code's own error messages)

| Symptom | Likely cause | Where it's raised |
| --- | --- | --- |
| `FileNotFoundError: No trained checkpoint found. Run training or pass --checkpoint path/to/best.pt.` | None of the default checkpoint paths exist and no `--checkpoint`/`TURN_DETECTOR_CHECKPOINT` was given. | `app.py:get_detector` |
| `FileNotFoundError: checkpoint not found: <path>` | Bad `--checkpoint` path passed to `TurnDetector`/`src/inference.py`. | `src/inference.py:TurnDetector.__init__` |
| `ValueError: checkpoint is missing required keys: [...]` | A `.pt` file that isn't one of this project's checkpoints (missing `config`/`model_state_dict`). | `src/inference.py:TurnDetector.__init__` |
| `RuntimeError: CUDA device requested, but CUDA is not available` | Explicit `device="cuda"` passed on a machine/environment without a working CUDA install. | `src/inference.py:TurnDetector.__init__` |
| `ValueError: audio waveform contains NaN or infinite values` / `is empty` | Corrupt or silent-but-malformed input audio reaching `TurnDetector.predict`/`predict_batch`. | `src/inference.py:_normalize_waveform` |
| `ValueError: multimodal training requires unaugmented audio matching cached transcripts` | Ran `src/train.py` with a multimodal config (`model.multimodal: true`) that also has `data.use_augmentation: true`. Fix the config, don't work around it in code. | `src/train.py:train` |
| `ValueError: train/validation metadata requires a non-null transcript column` | Trying to train/evaluate the multimodal (M1) model against a metadata Parquet that hasn't been through `scripts/transcribe_dataset.py`. | `src/train.py:build_dataloaders`, `src/evaluate.py:evaluate_checkpoint` |
| `ValueError: model has N parameters; requirement is under 15,000,000` | A model config change pushed the architecture over the project's stated parameter budget. | `src/train.py:train` |
| Slow or hanging first run of the demo or any training script | First run needs to reach the Hugging Face Hub (to fetch `openai/whisper-tiny` weights/config, and, for training/data prep, the `pipecat-ai/smart-turn-data-v3.2-*` dataset) or the Edge TTS endpoint (first-time filler synthesis in `FillerBank`). Both are network calls with no offline fallback in this repo; a populated local HF cache avoids the Hub calls on subsequent runs. | `src/models.py` (`WhisperModel.from_pretrained`), `src/dataset.py` (`load_dataset`, `FillerBank`) |

## Known documentation gap found while writing this runbook

The root README's "Repository layout" and "Reproducibility checks" sections
reference a `tests/` directory and `uv run python -m unittest discover -s
tests -v`. No `tests/` directory exists in the current working tree (`git
ls-files` shows no tracked test files, though a `tests/` directory did exist
in this repository's git history). Until tests are restored, there is no
automated regression suite to run; the closest available checks are `uvx
ruff check src scripts app.py` (linting only) and manually running the
`if __name__ == "__main__"` smoke checks already present in
`src/models.py`, `src/inference.py`, and other modules.
