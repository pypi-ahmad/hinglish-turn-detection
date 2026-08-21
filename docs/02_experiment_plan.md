<!-- markdownlint-disable MD013 MD060 -->

# Experiment plan for Hinglish turn detection

## Plan status: safety promotion completed

Original matrix below was pre-registered for fixed-0.5 architecture analysis.
Follow-up promotion protocol ran E1, E4, and E8 for six epochs across seeds
42/43/44. Each run selected threshold on validation only: maximize F1 subject
to FCR ≤10% and recall ≥85%. Architecture selection used median constrained F1;
deployment used median-performing seed. E4 seed 44 won at threshold `0.5777403`.
See [safety summary](generated/safety_v1_summary.md). Historical 0.5 rules below
still describe original protocol-v2 matrix, not production checkpoint.

## Purpose

This plan separates data effects, architecture effects, semantic information,
and hard-case behavior. Goal is not highest single accuracy number. Goal is to
learn which choices reduce premature interruption, why they help, and what
latency or model-size cost they impose.

Plan is pre-registered in `scripts/run_experiments.py`. Results belong in
`experiments/`; they must not be written back into hypotheses after seeing the
test set.

The [executable experiment-design notebook](../notebooks/02_experiment_design.ipynb)
loads the same registry, audits comparator configuration differences, checks
multimodal feasibility, and builds an in-memory fingerprinted dry-run manifest.

## Questions this plan answers

1. How much does data quality and augmentation matter relative to architecture?
2. What do pause length and filler injection contribute independently?
3. Does attention pooling improve pause-heavy cases over mean or last-frame pooling?
4. Does transcript-derived semantic information justify its ASR latency for Hinglish?
5. Does model remain reliable on short-complete, long-incomplete, Hindi-filler,
   and Hindi-pause cases?

## Fixed protocol

Unless a card explicitly says otherwise, every comparison keeps these constant:

| Control | Fixed value |
|---|---|
| Source data | Same train/validation manifests and optional final test manifest |
| Audio | Mono 16 kHz, last 8 seconds, left-padded |
| Encoder | `openai/whisper-tiny` encoder |
| Classifier | Same hidden size, normalization, activation, and dropout |
| Optimizer | AdamW, learning rate `5e-5`, weight decay `0.01` |
| Schedule | Cosine schedule, 20% warmup |
| Batch | 24 × four-step accumulation; effective batch 96 |
| Training budget | Same epoch count and seed within each comparison block |
| Sampling | Class-balanced sampler and hard-example weight unless ablated |
| Checkpoint selection | Highest validation F1 |
| Decision threshold | 0.5 for controlled model comparisons |
| Evaluation implementation | `src/evaluate.py` for every checkpoint |

One-factor comparisons are essential. For example, E9 changes filler
probability only; it does not also change pooling or fine-tuning depth.

## Metrics and guardrails

Every experiment tracks:

- Accuracy, precision, recall, F1, and ROC-AUC;
- false complete rate (FCR): probability of predicting complete when truth is
  incomplete;
- CPU and GPU batch-one latency;
- parameter count and FP32 model size.

Every experiment also reports these slices when populated:

- Hindi;
- known mid-filler and end-filler;
- internal pause and trailing pause;
- Hindi internal pause;
- Hindi filler plus internal pause;
- short-complete / long-incomplete hard examples;
- hard-Hinglish proxy: Hindi hard examples or Hindi filler-plus-pause examples;
- synthetic and recorded audio;
- optional manually curated Hinglish challenge set.

FCR is primary safety metric, but cannot be optimized alone. A model that calls
everything incomplete has zero FCR and is useless. Every FCR success rule has a
recall guard. Slice claims require at least 50 examples and both labels;
smaller or single-class slices are descriptive only.

## Decision hierarchy

Each claim belongs to one registered experiment/comparator pair. Analysis uses
this order:

1. Check run integrity: matching data fingerprint, seed, budget, and protocol.
2. Check overall F1 and FCR against pre-registered practical-effect threshold.
3. Apply completion-recall guardrail; a safer-looking model that delays every
   turn is rejected.
4. Inspect pause, filler, Hindi, hard-case, and recorded-audio slices with their
   support counts.
5. Compare latency, parameter count, and model size. When quality is practically
   tied, prefer simpler and faster system.
6. Label conclusion **supported**, **not supported**, or **inconclusive**. Do
   not force a winner from small or inconsistent differences.

Validation F1 selects checkpoints. Model selection uses validation comparisons.
Product threshold tuning is separate decision based on acceptable interruption
versus response-delay cost; threshold stays at 0.5 during architecture ablations.

## Experiment matrix

| ID | Variable isolated | Comparator | Main question |
|---|---|---|---|
| E1 | No online augmentation | Reference | Raw-data baseline |
| E2 | Full targeted augmentation | E1 | Total data-preparation effect |
| E3 | Mean pooling | E2 | Attention versus distributed averaging |
| E4 | Last-frame pooling | E2 | Sensitivity to trailing silence |
| E5 | 50-250 ms inserted pauses | E2 | Short-pause generalization |
| E6 | 600-1500 ms inserted pauses | E2 | Long-pause generalization |
| E7 | Fully frozen encoder | E2 | Need for task-specific representation learning |
| E8 | Freeze first two encoder layers | E2 | Quality versus trainable capacity |
| E9 | Remove filler injection only | E2 | Incremental filler value |
| E10 | Filler injection only | E1 | Standalone filler value |
| E11 | Remove silence insertion only | E2 | Incremental pause value |
| E12 | Remove hard-example oversampling | E2 | Hard-case curriculum value |
| M1 | Audio plus cached transcript features | E1 | Semantic value at matched data conditions |
| H1 | Evaluation-only challenge set | Selected finalists | Real hard-Hinglish behavior |

## Detailed experiment cards

### E1 : Unaugmented audio reference

- **Hypothesis:** Whisper audio representations learn a meaningful endpoint
  baseline from original clips without synthetic transformations.
- **What is changed:** `data.use_augmentation=false`.
- **What is kept constant:** Attention pooling, full encoder fine-tuning, hard
  sampling, optimizer, split, seed, and training budget.
- **Metrics:** Common metrics plus all clean-data pause, filler, language, and
  hard-case slices.
- **Success criteria:** Reference only. It establishes effect sizes for E2 and
  E10; no standalone pass/fail claim.

### E2 : Full targeted augmentation

- **Hypothesis:** Label-preserving filler injection and bounded acoustic
  transforms reduce false completes on filler/pause slices.
- **What is changed:** Enable silence, filler, speed, pitch, noise, and volume
  transformations using baseline probabilities.
- **What is kept constant:** Architecture and every optimization control from E1.
- **Metrics:** Common metrics; primary slices are internal pause, Hindi internal
  pause, Hindi filler-plus-pause, and hard-Hinglish proxy.
- **Success criteria:** Versus E1, lower hard-slice FCR, recall loss no larger
  than 2 percentage points, and overall F1 no worse by more than 1 point.

### E3 : Mean pooling

- **Hypothesis:** Mean pooling dilutes short endpoint cues inside an eight-second window.
- **What is changed:** `model.pooling=mean`.
- **What is kept constant:** Full E2 data and training protocol.
- **Metrics:** Common metrics; internal/trailing-pause and hard-Hinglish F1/FCR.
- **Success criteria:** Attention is justified if it gains at least 1 F1 point
  on internal-pause or hard-Hinglish slice without material latency cost.

### E4 : Last-frame pooling

- **Hypothesis:** Last-frame pooling fails when trailing silence separates final
  speech evidence from end of tensor.
- **What is changed:** `model.pooling=last`.
- **What is kept constant:** Full E2 data and training protocol.
- **Metrics:** Common metrics; trailing-pause, internal-pause, and Hindi-pause slices.
- **Success criteria:** Attention lowers pause-slice FCR while retaining recall
  within 2 points.

### E5 : Short silence augmentation

- **Hypothesis:** Training only on 50-250 ms gaps will not cover long hesitation.
- **What is changed:** Silence range becomes 50-250 ms.
- **What is kept constant:** All E2 augmentation probabilities and model choices.
- **Metrics:** Common metrics; internal-pause, Hindi-pause, and hard-Hinglish slices.
- **Success criteria:** Standard 100-800 ms range lowers pause-slice FCR without
  losing more than 1 point overall F1.

### E6 : Long silence augmentation

- **Hypothesis:** Training only on 600-1500 ms gaps makes brief breath pauses look incomplete.
- **What is changed:** Silence range becomes 600-1500 ms.
- **What is kept constant:** All E2 augmentation probabilities and model choices.
- **Metrics:** Common metrics; complete-class recall, trailing-pause accuracy,
  internal-pause FCR, and hard-Hinglish slices.
- **Success criteria:** Standard range improves complete recall and overall F1
  while matching long-pause protection within 2 FCR points.

### E7 : Frozen encoder

- **Hypothesis:** Frozen ASR features do not adapt enough to endpoint prosody.
- **What is changed:** Freeze all four Whisper encoder layers.
- **What is kept constant:** Pooling head, augmented data, optimizer, split,
  seed, and epoch budget. Equal budget makes this a representation ablation,
  not a frozen-head hyperparameter search.
- **Metrics:** Common metrics, trainable parameter count, and all hard slices.
- **Success criteria:** Full tuning improves overall and hard-slice F1 by at
  least 2 points. Low FCR with severely reduced recall does not count as success.

### E8 : Partial encoder fine-tuning

- **Hypothesis:** Updating later encoder layers retains most full-tuning quality
  with lower trainable capacity.
- **What is changed:** Freeze first two of four encoder layers.
- **What is kept constant:** Everything else from E2.
- **Metrics:** Common metrics, trainable parameters, training time, and hard slices.
- **Success criteria:** Within 1 F1 point and 2 FCR points of E2 overall and on
  hard-Hinglish proxy.

### E9 : No filler injection

- **Hypothesis:** Filler injection contributes value beyond pause/noise transforms.
- **What is changed:** `p_filler=0`; all other E2 transforms remain enabled.
- **What is kept constant:** Silence policy, model, sampling, split, seed, and budget.
- **Metrics:** Common metrics; Hindi mid-filler, Hindi filler-plus-pause, and
  hard-Hinglish slices are primary.
- **Success criteria:** E2 improves a filler-focused F1 or FCR metric by at
  least 1 point without losing more than 1 point overall F1.

### E10 : Filler-only augmentation

- **Hypothesis:** Label-preserving filler injection provides standalone benefit
  rather than working only as part of broad augmentation.
- **What is changed:** Filler probability remains 0.35; silence, speed, pitch,
  noise, and volume probabilities become zero.
- **What is kept constant:** E1 architecture and all training controls.
- **Metrics:** Common metrics and filler-focused slices.
- **Success criteria:** Versus E1, improve filler-slice F1 or FCR by at least 1
  point while overall F1 remains within 1 point.

### E11 : No silence insertion

- **Hypothesis:** Silence insertion, rather than generic augmentation, drives
  pause robustness.
- **What is changed:** `p_silence=0`; all other E2 transforms remain enabled.
- **What is kept constant:** Filler, noise, speed, pitch, volume, architecture,
  split, seed, and budget.
- **Metrics:** Common metrics; internal-pause and Hindi-pause FCR are primary.
- **Success criteria:** E2 lowers internal-pause FCR by at least 2 points with
  recall loss no larger than 2 points.

### E12 : No hard-example mining

- **Hypothesis:** Hard-example oversampling improves difficult endpoint cases
  instead of merely shifting overall class behavior.
- **What is changed:** `data.use_hard_negatives=false`.
- **What is kept constant:** Full E2 augmentation and architecture.
- **Metrics:** Common metrics; short-complete, long-incomplete, Hindi filler,
  and hard-Hinglish slices.
- **Success criteria:** E2 improves hard-Hinglish F1 by at least 2 points while
  overall F1 changes by no worse than -1 point.

### M1 : Audio plus semantic text

- **Hypothesis:** Transcript tokens help distinguish discourse markers and
  incomplete syntax that acoustics alone underuse.
- **What is changed:** Add frozen Whisper ASR transcript generation, learned
  64-dimensional token embeddings, and audio/text concatenation.
- **What is kept constant:** Compare with E1: same unaugmented audio, split,
  attention pooling, seed, optimizer, and budget. Dynamic audio augmentation is
  disabled because cached text must remain aligned.
- **Metrics:** Common metrics plus end-to-end latency including ASR, Hindi and
  Hindi-filler slices, and empty/incorrect transcript rate.
- **Success criteria:** Adopt only if Hindi/hard-Hinglish F1 improves by at
  least 2 points or FCR improves by at least 2 points with recall preserved,
  and measured ASR latency fits deployment budget. Classifier-only latency is
  insufficient evidence.

### H1 : Curated Hinglish challenge evaluation

- **Hypothesis:** Aggregate test performance overstates robustness on natural
  code-switching, fillers, rising intonation, and ambiguous pauses.
- **What is changed:** Evaluation data only: use independently curated,
  speaker-disjoint Hinglish clips. No training or threshold tuning on H1.
- **What is kept constant:** Frozen finalist checkpoints and decision threshold.
- **Metrics:** Common classification metrics, FCR, error categories, and
  per-speaker bootstrap intervals. Latency and size come from corresponding model.
- **Success criteria:** FCR below 10% with recall at least 80%, no speaker with
  catastrophic failure, and no more than 5-point F1 drop from clean final test.
  Until real H1 exists, `hard_hinglish_proxy` is explicitly only a proxy.

## Threats to validity and controls

| Threat | How it could mislead | Control or reporting rule |
|---|---|---|
| Random-seed variance | Small apparent wins may reverse | Repeat finalists and direct controls with seeds 42-44; report mean, standard deviation, and direction consistency |
| Repeated comparison | Large ablation matrix increases chance findings | Treat core matrix as exploratory; confirm only decision-relevant finalists against direct controls |
| Test-set feedback | Repeated test inspection turns test into validation | Select model and threshold on validation; run frozen finalist on test once |
| Slice sparsity | Extreme metric from few or single-class examples | Report support; require at least 50 rows and both labels for comparative claims |
| Synthetic-domain bias | Model may exploit TTS cadence rather than turn structure | Report synthetic and recorded slices separately; require real Hinglish H1 before deployment claim |
| Proxy construct validity | Hindi/filler/pause masks are not verified Hinglish | Call them proxies; do not claim code-switch robustness without transcripts/manual labels |
| Speaker leakage | Same voice family may appear across partitions | Fingerprint IDs/audio; report lack of speaker IDs; require speaker-disjoint H1 |
| Unequal convergence | Equal epochs may favor one tuning strategy | Use equal budget for causal ablation, then allow finalist-specific tuning only in separately labeled optimization phase |
| Multimodal alignment | Cached transcript may not match augmented waveform | Disable dynamic augmentation for M1 and compare against unaugmented E1 |
| Latency measurement bias | Classifier-only timing hides ASR cost | Report warmed batch-one end-to-end M1 latency including transcription |

## Run order and statistical discipline

1. Run dry plan validation. It fingerprints every metadata file and resolves
   each config before GPU work.
2. Run core matrix on validation only with seed 42. Treat differences as
   exploratory.
3. Run full matrix if pooling/fine-tuning questions remain decision-relevant.
4. Repeat finalists and their direct controls with seeds 43 and 44. Report
   mean, standard deviation, and direction consistency. A claimed improvement
   should have same direction in at least two of three seeds.
5. Freeze architecture, data policy, checkpoint rule, and decision threshold.
6. Use `--final-test` once for frozen finalists. Do not select a winner from
   repeated test-set inspection.
7. Evaluate optional H1 only after final selection.

For close paired finalists, bootstrap per-example prediction differences. If a
95% interval includes zero, report "inconclusive under current sample," not a win.

## Running framework

Resolve and fingerprint core suite without training:

```powershell
uv run python scripts/run_experiments.py --dry-run
```

Run validation-first core and full suites:

```powershell
uv run python scripts/run_experiments.py --suite core --epochs 3
uv run python scripts/run_experiments.py --suite full --epochs 3
```

Completed confirmatory protocol used isolated directories:

```powershell
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 42 --run-tag safety_v1_seed42
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 43 --run-tag safety_v1_seed43
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 44 --run-tag safety_v1_seed44
```

Select frozen winner and perform one final test:

```powershell
uv run python scripts/select_safety_finalist.py experiments/safety_v1_seed42 experiments/safety_v1_seed43 experiments/safety_v1_seed44
```

Add independent challenge manifest:

```powershell
uv run python src/evaluate.py --checkpoint checkpoints/safety_finalist/best.pt --metadata data/subset/hinglish_challenge.parquet --output experiments/safety_finalist/challenge_metrics.json
```

M1 remains a separate matched workflow because transcript caching is expensive:

```powershell
uv run python scripts/transcribe_dataset.py
uv run python scripts/run_multimodal_experiment.py
```

## Structured outputs

Each suite root writes:

- `experiment_manifest.json`: hypotheses, success criteria, resolved configs,
  config hashes, row counts, and SHA-256 data fingerprints;
- `<experiment>/config.yaml`: exact training config;
- `<experiment>/history.json`: per-epoch loss and validation metrics;
- `<experiment>/metrics.json`: selected-checkpoint validation metrics and latency;
- `<experiment>/validation_metrics.json`: default comparison metrics and slices;
- `<experiment>/test_metrics.json`: written only with `--final-test`;
- `<experiment>/challenge_metrics.json`: optional H1 evaluation;
- `<experiment>/result.json`: compact structured run record;
- `all_results.json` and `comparison.csv`: combined machine-readable outputs;
- `docs/generated/ablation_report.md` or
  `docs/generated/<run-tag>_ablation_report.md`: human-readable comparison.

Protocol version is embedded in configs and result records. Old artifacts with
the earlier trailing-filler weak-label policy cannot be silently reused by the
current label-preserving protocol.
