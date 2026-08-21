<!-- markdownlint-disable MD013 -->

# Review findings

No critical security flaw or obvious label leak appeared in the traced paths.
Checkpoint loading uses `weights_only=True`, YAML uses `safe_load`, and held-out
test data is separate from checkpoint selection.

## Warnings

1. Public inference has no explicit upload limit. File input is fully decoded
   before the final eight-second crop, so a large upload can waste memory and CPU.
2. Cold inference still depends on Hugging Face assets even with a local
   checkpoint.
3. Dataset and model Hub references do not use immutable revisions.
4. Fast tests miss full training, checkpoint evaluation, canonical inference,
   and the Gradio callback.
5. The repository has no automated CI, coverage, type, or dependency-security
   gate.

## Design trade-offs

Importing `configs/config.py` creates directories. Direct script execution adds
the repository root to `sys.path`. A single lock serializes inference, which is
safe for the current queue but limits one-process throughput. Gradio displays
selected exception text directly. `requirements.txt` also depends on packages
provided by the Space base image.

## Evidence limits

- No speaker or accent IDs, so evaluation cannot prove speaker separation.
- No verified code-switch transcripts, so Hinglish slices are proxies.
- Training used a bounded subset and mostly synthetic audio.
- Broad ablation matrix uses one seed; safety finalists use three.
- Threshold is validation-calibrated, but held-out recall fell below 85% target.

The phrase "production inference" should be read as a reusable, validated
single-process API. It does not imply request limits, health checks, telemetry,
or offline packaging.

## Evidence

- `src/inference.py:66`, `:111`, `:159`
- `app.py:109-120`, `:165`
- `configs/config.py`
- `src/dataset.py:230`, `src/models.py:228`
- `tests/`
