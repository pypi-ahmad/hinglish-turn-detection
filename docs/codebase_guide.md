<!-- markdownlint-disable MD013 -->

# Codebase guide

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
        ├──► src/train.py ──► constrained-validation checkpoint + history
        ├──► src/evaluate.py ──► metrics, slices, threshold calibration
        └──► src/inference.py / app.py ──► probability, decision, threshold, latency
```

`scripts/run_experiments.py` orchestrates controlled audio ablations around those modules. `scripts/select_safety_finalist.py` aggregates matched seed runs and copies the median-seed winner. `scripts/run_multimodal_experiment.py` owns the matched audio-versus-audio+text workflow. Machine evidence stays under `experiments/`; human-readable generated reports go to `docs/generated/`.

## Responsibility boundaries

| Path | Owns | Does not own |
| --- | --- | --- |
| `configs/config.py` | Stable paths, constants, filler vocabulary, acquisition defaults | Per-experiment changes |
| `configs/*.yaml` | Reproducible model/data/training settings | Training implementation |
| `src/dataset.py` | Audio validation, normalization, augmentation, hard-example indices, batching | Model architecture or metric interpretation |
| `src/models.py` | Encoder, pooling variants, classifier, optional text fusion | Audio decoding or checkpoint selection |
| `src/train.py` | Optimization, scheduling, mixed precision, checkpoint selection | Final test-set model selection |
| `src/evaluate.py` | Metrics, FCR, slices, latency, validation threshold selection | Training |
| `src/inference.py` | File/array input contract, resampling, batching, thresholded decisions | UI layout |
| `app.py` | Gradio interaction and visualization | Model internals |
| `scripts/prepare_data.py` | Acquisition, local manifests, split generation, previews | Online epoch augmentation |
| `scripts/explore_data.py` | Reproducible local-data report | Dataset downloading |
| `scripts/run_experiments.py` | One-factor ablation plan, manifests, result aggregation | Core training/evaluation logic |
| `scripts/select_safety_finalist.py` | Cross-seed architecture/seed selection and final test handoff | Training implementation |
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

- Best checkpoint is selected by constrained validation F1 when calibration is enabled.
- Experiment-runner test evaluation requires explicit `--final-test`; finalist selector evaluates its frozen winner once.
- Production threshold comes from checkpoint; legacy checkpoints fall back to 0.5.
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
uv run python src/evaluate.py --checkpoint checkpoints/safety_finalist/best.pt --metadata data/subset/test_split.parquet

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

## CLI and environment reference

| Entry point | Main options |
| --- | --- |
| `app.py` | checkpoint, host, port |
| `src/train.py` | config, train metadata, validation metadata |
| `src/evaluate.py` | checkpoint, metadata, output, optional threshold override |
| `src/inference.py` | checkpoint, audio path, optional threshold override |
| `scripts/prepare_data.py` | scan budgets, preview count, validation fraction |
| `scripts/run_experiments.py` | suite, experiment IDs, base config, epochs, seed, run tag, reuse, dry run, final test, metadata paths |
| `scripts/select_safety_finalist.py` | seed run roots, output directory, held-out test metadata |
| `scripts/transcribe_dataset.py` | model, batch size, token limit, device, splits, output directory |
| `scripts/run_multimodal_experiment.py` | config, result paths, metadata paths, reuse |
| `scripts/download_and_demo.py` | repository ID, filename, revision, host, port |

| Variable | Use |
| --- | --- |
| `TURN_DETECTOR_CHECKPOINT` | App checkpoint when no CLI path is supplied |
| `HF_MODEL_REPO_ID` | Default repository for `download_and_demo.py` |
| `HF_TOKEN` | Standard Hugging Face authentication |
| `SPACES_ZERO_GPU` | Managed Spaces GPU switch |

Run a command with `--help` for exact defaults. Training has no `--set` option;
copy a YAML config to change an experiment. Cold model loading, dataset
streaming, and first-time Edge TTS generation need network access unless their
assets are already cached.

Important YAML controls:

| Key | Meaning |
| --- | --- |
| `model.pooling` | `attention`, `mean`, or `last` |
| `model.freeze_encoder_layers` | Number of earliest Whisper encoder layers frozen, 0-4 |
| `data.use_augmentation` | Enables online training transforms only |
| `data.use_hard_negatives` | Enables hard-case sampler boost |
| `evaluation.threshold_calibration.enabled` | Enables validation-only operating-point search |
| `evaluation.threshold_calibration.max_false_complete_rate` | FCR ceiling; baseline uses 0.10 |
| `evaluation.threshold_calibration.min_recall` | Recall floor; baseline uses 0.85 |
| `checkpoint.dir` | Destination containing `best.pt` |

When calibration is absent or disabled, training and legacy inference use 0.5.
When enabled, checkpoint stores threshold, feasibility, constraints, selected
metrics, and confusion counts.

## Artifact map

| Artifact | Location | Commit policy |
| --- | --- | --- |
| Small demo WAVs and labels | `data/samples/` | Included |
| Prepared training audio/manifests | `data/` | Regenerable; ignored |
| Bundled demo checkpoint | `checkpoints/safety_finalist/best.pt` | Included submission artifact |
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
