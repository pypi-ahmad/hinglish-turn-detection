<!-- markdownlint-disable MD013 -->

# Technology stack

This is a Python 3.12 project managed with `uv`. `pyproject.toml` describes the
full training environment, while `uv.lock` records exact transitive versions.
`requirements.txt` is the smaller Hugging Face Spaces runtime list.

| Area | Libraries | Use |
| --- | --- | --- |
| Model | PyTorch, Transformers | Whisper encoder training and inference |
| Data | Datasets, Polars, PyArrow | Streaming input and Parquet metadata |
| Audio | librosa, SoundFile, pydub | Decode, resample, augment, save |
| Metrics | scikit-learn | Accuracy, precision, recall, F1, AUC |
| Demo | Gradio, Matplotlib | Browser UI and waveform plot |
| Hosting | Hugging Face Hub and Spaces | Model download and public demo |

The audio model uses the encoder from `openai/whisper-tiny`. It receives mono
16 kHz audio, keeps the last eight seconds, and emits one completion logit.
Attention, mean, and last-frame pooling are available. The optional multimodal
model adds embeddings from a Whisper transcript.

There is no database, API server, container, task queue, or CI workflow.
Tests use `unittest`; Ruff handles linting. No type checker or coverage gate is
configured.

## Evidence

- `pyproject.toml`, `uv.lock`, and `requirements.txt`
- `src/models.py:184`
- `src/dataset.py:80`
- `app.py:126`
