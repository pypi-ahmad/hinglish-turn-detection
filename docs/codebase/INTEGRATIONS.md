<!-- markdownlint-disable MD013 -->

# External integrations

## Hugging Face

Data preparation streams `pipecat-ai/smart-turn-data-v3.2-train` and the
separate test repository through `datasets.load_dataset`. Model construction
uses `openai/whisper-tiny` through Transformers. These remote references are not
pinned to immutable revisions.

The local checkpoint does not make cold startup fully offline. Model and feature
extractor construction still call `from_pretrained`, so the host needs network
access or a populated Hugging Face cache. `scripts/download_and_demo.py` can
download a submitted checkpoint. Private assets use standard `HF_TOKEN`
authentication; the repository does not store a token.

## Gradio and Spaces

The root README contains Space metadata and points to `app.py`. The module-level
`demo` object is available to Spaces. `SPACES_ZERO_GPU=1` enables the GPU
decorator; local execution uses a no-op replacement. The queue accepts 32 jobs
and handles one request at a time.

## Edge TTS

`FillerBank` uses `edge-tts` for Hindi and Indian-English filler clips. The first
build needs network access, then reuses its local cache.

## Files

WAV and Parquet files connect preparation to training. Training writes PyTorch
checkpoints and JSON histories. Experiment scripts add CSV comparisons and
Markdown reports. Transcript caching writes a temporary Parquet file before it
replaces the prior output.

Network calls have no repository-level retry policy. The demo has no health
endpoint or structured production telemetry.
