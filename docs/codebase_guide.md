<!-- markdownlint-disable MD013 -->

# Codebase Guide

This guide maps project reasoning to implementation. Read [Full Technical Approach](04_full_approach.md) first for why decisions were made; use this page to find where each decision lives.

## End-to-end flow

```text
Hugging Face audio rows
        │
        ▼
scripts/prepare_data.py ──► normalized WAV + split Parquet manifests
        │
        ├──► scripts/explore_data.py ──► docs/data_exploration.md
        │
        ▼
src/dataset.py ──► waveform, label, metadata, optional online augmentation
        │
        ▼
collate_fn ──► Whisper log-mel tensors + masks
        │
        ▼
src/models.py ──► Whisper encoder + pooling + binary logit
        │
        ├──► src/train.py ──► best-validation checkpoint + history
        ├──► src/evaluate.py ──► aggregate and hard-slice metrics
        └──► src/inference.py / app.py ──► probability, decision, latency
```

`scripts/run_experiments.py` orchestrates controlled audio ablations around those modules. `scripts/run_multimodal_experiment.py` owns the matched audio-versus-audio+text workflow. Machine evidence stays under `experiments/`; human-readable generated reports go to `docs/generated/`.

## Responsibility boundaries

| Path | Owns | Does not own |
| --- | --- | --- |
| `configs/config.py` | Stable paths, constants, filler vocabulary, acquisition defaults | Per-experiment changes |
| `configs/*.yaml` | Reproducible model/data/training settings | Training implementation |
| `src/dataset.py` | Audio validation, normalization, augmentation, hard-example indices, batching | Model architecture or metric interpretation |
| `src/models.py` | Encoder, pooling variants, classifier, optional text fusion | Audio decoding or checkpoint selection |
| `src/train.py` | Optimization, scheduling, mixed precision, checkpoint selection | Final test-set model selection |
| `src/evaluate.py` | Metrics, FCR, pause/filler/language slices, latency | Training or threshold tuning |
| `src/inference.py` | File/array input contract, resampling, batching, thresholded decisions | UI layout |
| `app.py` | Gradio interaction and visualization | Model internals |
| `scripts/prepare_data.py` | Acquisition, local manifests, split generation, previews | Online epoch augmentation |
| `scripts/explore_data.py` | Reproducible local-data report | Dataset downloading |
| `scripts/run_experiments.py` | One-factor ablation plan, manifests, result aggregation | Core training/evaluation logic |
| `tests/` | Regression proof for data, models, training, evaluation, inference, and app behavior | Large-model quality claims |

## Stable contracts

### Labels

- `0`: incomplete; user is still speaking or pausing.
- `1`: complete; system may respond.
- Model output: raw logit during training, sigmoid probability at inference.

### Audio

- Internal sample rate: 16 kHz mono `float32`.
- Maximum real content: final eight seconds.
- Short clips: left-padded so endpoint remains aligned.
- Dataset returns waveform; `collate_fn` owns feature extraction.

### Experiment discipline

- Best checkpoint is selected by validation F1.
- Test evaluation requires explicit `--final-test`.
- Threshold is fixed at 0.5 for controlled comparisons.
- FCR is always interpreted with a recall guard.
- Configs and metadata files are fingerprinted before expensive runs.
- Tagged runs write isolated machine artifacts under `experiments/<run-tag>/` and readable reports under `docs/generated/`.

## Main commands

```powershell
# Install locked environment
uv sync

# Prepare and inspect bounded data
uv run python scripts/prepare_data.py
uv run python scripts/explore_data.py

# Train and evaluate bundled architecture
uv run python src/train.py --config configs/baseline.yaml
uv run python src/evaluate.py --checkpoint checkpoints/baseline_attention_augmented/best.pt --metadata data/subset/test_split.parquet

# Resolve experiment plan without GPU training
uv run python scripts/run_experiments.py --dry-run --suite full

# Run demo
uv run python app.py

# Verify repository
uv run python -m unittest discover -s tests -v
uvx ruff check src scripts tests app.py
uv lock --check
uv pip check
```

## Artifact map

| Artifact | Location | Commit policy |
| --- | --- | --- |
| Small demo WAVs and labels | `data/samples/` | Included |
| Prepared training audio/manifests | `data/` | Regenerable; ignored |
| Bundled demo checkpoint | `checkpoints/baseline_attention_augmented/best.pt` | Included submission artifact |
| Other checkpoints | `experiments/**/checkpoints/` | Regenerable; ignored |
| Configs, histories, metrics, fingerprints | `experiments/` | Included as evidence |
| Curated explanation | `docs/` | Included |
| Generated readable comparisons | `docs/generated/` | Included and reproducible |

## Extending safely

When adding a new augmentation:

1. add one probability/config field to `AugmentConfig`;
2. apply it only in `TurnDetectionDataset` training path;
3. add a regression test proving label and metadata behavior;
4. register an isolated experiment in `EXPERIMENTS`;
5. document hypothesis and success criterion before running it.

When adding a new model variant:

1. keep `build_model` as construction entry point;
2. preserve the `forward` logit contract;
3. keep evaluation through `evaluate_checkpoint`;
4. record parameter count and batch-one latency;
5. compare against the nearest control with the same data and budget.

When changing the decision threshold, choose it on validation data for a declared FCR/recall target. Never tune it on the final test set.
