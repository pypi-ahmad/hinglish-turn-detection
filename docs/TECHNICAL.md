<!-- markdownlint-disable MD013 -->

# Technical notes

Stack rationale, invariants, and error-handling/persistence details that are
not obvious from file names alone. For the full dependency-by-dependency
breakdown see [`docs/codebase/STACK.md`](codebase/STACK.md); for conventions
(naming, style, config shape) see [`docs/codebase/CONVENTIONS.md`](codebase/CONVENTIONS.md).

## Stack

| Choice | Why (per the code) |
| --- | --- |
| Python ≥3.12, managed with `uv` | `pyproject.toml` pins `requires-python = ">=3.12"`; `.python-version` pins `3.12`; there is a `uv.lock` for the full training/dev environment. |
| `torch` from a `pytorch-cu128` extra index | `pyproject.toml`'s `[tool.uv.sources]`/`[[tool.uv.index]]` pin `torch` to PyTorch's CUDA 12.8 wheel index, not PyPI's default CPU build. |
| `transformers.WhisperModel` (encoder only) | `src/models.py` loads the full seq2seq `WhisperModel` via `from_pretrained` (the only public entry point that restores pretrained encoder weights correctly) and immediately discards the decoder; see that file's module docstring for the full rationale. |
| Hand-written training loop, not `transformers.Trainer` | `src/train.py`'s docstring: `TurnDetectionModel` isn't a `PreTrainedModel`, and the augmentation/collation pipeline is already custom, so `Trainer`'s assumptions would cost more than they save for a ~40-line loop. |
| `polars` for metadata tables | All metadata I/O (`src/dataset.py`, `src/train.py`) uses `polars.DataFrame`/Parquet, not `pandas`. |
| `librosa` for resampling/effects, `soundfile` for WAV I/O | `src/dataset.py`, `src/inference.py`: `librosa.load`/`resample`/`effects.time_stretch`/`effects.pitch_shift` for decoding and augmentation; `soundfile.read`/`write` for the actual WAV files on disk. |
| `edge-tts` for filler-word synthesis | `FillerBank` in `src/dataset.py` uses Microsoft Edge's free neural TTS to generate Hindi/Indian-English filler clips; no bundled/licensed filler audio ships in the repo, so this runs (and caches) on first use. |
| `gradio` for the demo UI | `app.py` builds a `gr.Blocks` app; `spaces.GPU` is used conditionally only when `SPACES_ZERO_GPU=1` is set (a no-op shim otherwise), so the same `app.py` runs locally and on a Hugging Face Space with the ZeroGPU decorator. |
| `onnx`/`onnxruntime` listed as dependencies | Present in `pyproject.toml`; there is an `onnx/` output directory declared in `configs/config.py` (`ONNX_DIR`), but no export script exists under `scripts/` or `src/` in this tree; treat ONNX export as a documented future step (see the README's "Future work"), not a working command today. |
| `requirements.txt` is a separate, smaller pin set | Its header comment says it is the runtime set for "the Gradio demo and Hugging Face Space" specifically (`datasets`, `librosa`, `matplotlib`, `numpy`, `polars`, `soundfile`, `tqdm`, `transformers`; no `torch`, since Spaces provides its own). Use `uv sync` (which reads `pyproject.toml`/`uv.lock`) for training, data prep, notebooks, and any local development. |

## Invariants worth knowing before you change code

- **Left-padding, not right-padding.** `collate_fn` in `src/dataset.py`
  zero-pads every waveform on the *left* so real audio always ends at the
  same position in the fixed 8-second window. `LastFramePool` in
  `src/models.py` depends on this: it blindly takes `hidden_states[:, -1, :]`
  and is only correct because the last frame is guaranteed to be real audio,
  never padding.
- **Units and windows are fixed constants, not knobs per call.** Audio is
  always 16 kHz mono (`cfg.SAMPLE_RATE`), cropped to the trailing 8 seconds
  (`cfg.MAX_DURATION_S` / `cfg.MAX_REAL_SAMPLES`), which the feature extractor
  turns into 800 mel frames and the encoder into 400 hidden positions
  (`cfg.WHISPER_MEL_FRAMES`, `cfg.WHISPER_ENCODER_HIDDEN_LEN`). These numbers
  are load-bearing across `src/dataset.py`, `src/models.py`, and
  `src/inference.py`; changing one without the others will silently misalign
  the attention mask.
- **Validation is never augmented.** `build_dataloaders` in `src/train.py`
  explicitly builds the validation dataset with `AugmentConfig(enabled=False)`.
  Comparing epoch-to-epoch validation numbers assumes this; don't add
  augmentation to the validation path without also invalidating historical
  `history.json` comparisons.
- **Threshold calibration reads validation data only, never the held-out
  test set.** `select_operating_threshold` (`src/evaluate.py`) is called from
  `src/train.py` against validation probabilities. Test-set evaluation is a
  separate, explicit step (`scripts/run_experiments.py --final-test`,
  `src/evaluate.py evaluate_checkpoint`) and is documented in the README as a
  one-time check, not something to iterate against.
- **Multimodal (M1) training requires disabled audio augmentation.**
  `src/train.py:train` raises `ValueError` if `multimodal=True` and
  `data.use_augmentation` is true: cached transcripts describe one specific,
  fixed version of the audio, and on-the-fly perturbation would desync
  transcript from waveform. `configs/multimodal.yaml` sets
  `use_augmentation: false` for this reason.
- **Checkpoints embed their own config and threshold.** `src/inference.py`'s
  `TurnDetector.__init__` rebuilds the model architecture entirely from
  `checkpoint["config"]["model"]` and reads `checkpoint["decision_threshold"]`
  (falling back to `0.5` if absent, for older checkpoints saved before
  calibration was added); you cannot load a checkpoint without knowing what
  architecture produced it, by design, since the pooling mode and multimodal
  flag change the model's shape.

## Error handling

Validation happens at every external-input boundary and fails loudly
(`ValueError`/`TypeError`/`FileNotFoundError`), rather than silently
coercing bad input:

- `src/inference.py`'s `_normalize_waveform`/`_load_and_resample` reject empty
  audio, non-finite (`NaN`/`inf`) samples, ambiguous channel layouts, and
  non-positive sample rates before any model code runs.
- `_validate_threshold` rejects non-numeric, out-of-`[0, 1]`, or boolean
  thresholds (Python's `bool` is a `Real` subclass, so it is explicitly
  excluded: passing `threshold=True` is a bug, not "always complete").
- `TurnDetector.__init__` requires a checkpoint file to exist and to contain
  both `config` and `model_state_dict` keys; loads with `weights_only=True`
  (no arbitrary pickle execution from a checkpoint file).
- `src/dataset.py`'s `stream_filtered_subset` raises on a malformed row
  (missing/invalid `id`, `language`, or non-boolean `endpoint_bool`) rather
  than skipping and silently shrinking the dataset.
- Nothing in `src/` or `scripts/` catches and swallows exceptions to continue
  past bad data; the one place that does something similar is
  `app.py`'s Gradio callback, which re-raises expected exception types as
  `gr.Error` so they render as a UI message instead of a stack trace, without
  changing what triggered them.

## Persistence paths

See [`docs/ARCHITECTURE.md`](ARCHITECTURE.md#where-state-lives) for the full
table of what gets written where. Everything is flat files under the repo
tree (WAV, Parquet, JSON, PyTorch `.pt`); there is no database, cache server,
or object store involved.
