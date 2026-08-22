---
title: Hinglish Turn Detection
emoji: 🎙️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.25.0
python_version: "3.12"
app_file: app.py
short_description: Tiny audio turn detection for Indian Hinglish speech
startup_duration_timeout: 30m
pinned: false
---

<!-- markdownlint-disable MD013 MD025 -->

# Hinglish turn detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Live Space](https://img.shields.io/badge/demo-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/pypi-ahmad/hinglish-turn-detection)
[![GitHub](https://img.shields.io/badge/source-GitHub-181717?logo=github)](https://github.com/pypi-ahmad/hinglish-turn-detection)

Tiny audio classifier for a deceptively hard voice-agent question: **did user finish speaking, or are they pausing mid-thought?** Model targets Indian Hindi/English speech, fillers, and ambiguous pauses while remaining small enough for low-latency serving.

It uses [Pipecat Smart Turn v3.2](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train), an `openai/whisper-tiny` encoder, last-frame pooling, and binary classification head. Final model has **7.9M parameters** and predicts `P(turn complete)` from last eight seconds of audio.

## Executive summary

- The source has 270,946 rows, but no speaker IDs, transcripts, accent labels,
  or verified Hinglish/code-switch labels. The pipeline records those limits,
  decodes a bounded sample, preserves silence, and keeps unknown filler
  annotations distinct from confirmed absence.
- Augmentation targets specific failure cases. It uses label-preserving fillers,
  class-aware pauses, restrained acoustic changes, noise, and hard-example
  sampling without teaching the shortcut "filler means incomplete."
- Nine audio ablations and one matched audio-text comparison found no universal
  winner. Full augmentation led overall F1, original audio had the lowest FCR,
  last-frame pooling led hard-Hinglish proxy F1, and partial fine-tuning kept
  similar quality with 4.45M trainable parameters.
- Some useful results were negative. Mean pooling lost endpoint evidence, a
  fully frozen encoder failed, and live ASR made multimodal inference 27.6 times
  slower while increasing interruption risk.
- A second safety study repeated E1, E4, and E8 for six epochs across seeds
  42/43/44. Validation-only calibration enforced FCR ≤10% and recall ≥85%; E4
  won median F1 and seed 44 became production checkpoint.
- Final held-out evaluation confirmed lower interruption risk (9.84% FCR), but
  recall fell to 83.26%. Next need is human speaker-disjoint Hinglish data, not
  post-hoc threshold tuning on test results.

Start with the [full technical approach](docs/04_full_approach.md), or follow the complete [documentation reading guide](docs/README.md).

Prefer lessons over a long report? Open the [interactive tutorial](tutorial/index.html). It covers the problem, data, model, experiments, deployment, and interview explanation as a short multipage course.

## Problem statement

Voice activity detection can tell whether sound stopped. It cannot tell why. A speaker may be finished, searching for next word, or saying "matlab...", "haan...", "actually...", or "ek second..." before continuing.

Wrong error direction matters:

- **False complete:** assistant interrupts user mid-thought.
- **False incomplete:** assistant waits slightly longer.

Therefore project reports accuracy and F1, but also treats **false-complete rate** as product-safety metric:

```text
P(predict complete | utterance is actually incomplete)
```

## Approach summary

Depth came from defining failure modes before choosing model:

1. Inspect the data first. The source has 270,946 rows and 23 languages, but
   no usable transcripts, speaker IDs, accent IDs, or explicit Hinglish label.
2. Bound acquisition. Fixed scan budgets avoid an implicit 41.4 GB download and
   record exactly what the local sample contains.
3. Counter duration shortcuts by sampling short complete turns and long
   incomplete turns with fillers more often.
4. Use pause, speed, pitch, volume, noise, and filler transforms only for named
   deployment failures.
5. Match Whisper's context to the task by retaining its first 400 learned
   encoder positions for an eight-second input.
6. Report FCR and Hindi, filler, synthetic/human, pause, and hard-case slices.
7. Hold the seed, split, optimizer, and three-epoch budget fixed during
   ablations.
8. Generate cached M1 transcripts without labels and disable dynamic audio
   augmentation so its audio and text remain aligned.

## Data preparation

### Observed data

| Property | Observation |
| --- | --- |
| Full train repository | 270,946 rows, 41.4 GB, 23 languages |
| Full Hindi slice | 12,006 rows (4.43%), near-balanced labels |
| Local bounded train sample | 7,517 rows; 49.7% complete / 50.3% incomplete |
| Local split | 6,613 train + 904 validation |
| Held-out publisher test sample | 4,890 rows |
| Local Hindi rows | 721 train-sample rows; 254 held-out rows |
| Transcripts | source `spoken_text` null; frozen-Whisper cache derived for M1 |
| Speaker/accent IDs | unavailable |

Data is decoded to mono 16 kHz WAV. Last eight seconds are retained because turn-ending cues concentrate near utterance boundary. Short clips are left-padded so final speech frame stays aligned.

### Hinglish, fillers, and pauses

| Technique | Reason |
| --- | --- |
| Silence insertion (100-800 ms) | Prevent "silence means complete" shortcut; incomplete rows receive higher pause probability |
| Hindi/Indian-English filler TTS | Construct missing lexical phenomena: `um`, `uh`, `matlab`, `actually`, `tho`, `yaar`, `bas`, `wait`, `ek second`, `haan` |
| Speed 0.9-1.1× | Reduce dependence on speaking rate |
| Pitch ±2 semitones | Improve speaker variation without destroying intonation |
| Volume ±6 dB | Remove microphone gain as shortcut |
| Noise at 5-20 dB SNR | Improve non-studio robustness; licensed recordings supported, synthetic room noise used as explicit fallback |
| Hard-example oversampling | Contrast short-complete with long-incomplete-with-filler cases |

Stochastic transforms run on-the-fly. This avoids storing duplicated audio and gives each epoch fresh perturbations. Filler audio is inserted internally into both classes and preserves the publisher label; filler presence alone is not reliable evidence of incompleteness.

Split is stratified by `(language, endpoint_bool, source_dataset)`. True speaker-disjoint evaluation is impossible because source has no speaker identity. Detailed rationale: [data preparation approach](docs/01_data_preparation_approach.md).

## Model architecture

```text
audio → mono 16 kHz → last 8 s → Whisper log-mel features
      → Whisper-tiny encoder (400 positions)
      → last encoder frame
      → Linear(384,256) → LayerNorm → GELU → Dropout
      → Linear(256,64) → GELU → Linear(64,1)
      → sigmoid → P(turn complete)
```

| Item | Value |
| --- | --- |
| Encoder | `openai/whisper-tiny` encoder only |
| Output | binary: incomplete `0`, complete `1` |
| Parameters | 7,901,569 |
| FP32 parameter size | 30.14 MiB |
| Input window | 8 seconds, 800 mel frames, 400 encoder positions |
| Training | AdamW, cosine schedule, FP16, gradient clipping |
| Checkpoint selection | maximum validation F1 under FCR ≤10% and recall ≥85% |

Multimodal M1 keeps same audio branch, generates transcripts offline with
frozen `openai/whisper-tiny`, mean-pools learned 64-dimensional Whisper-token
embeddings, concatenates them with 384-dimensional audio embedding, then uses
same small classifier. Total: 11,336,130 parameters. Cached transcripts keep
training practical; live use must also pay autoregressive ASR latency.

## Experiments and lessons

All completed ablations below use the same split, seed 42, and three-epoch budget. Checkpoints were selected by validation F1, then evaluated once on the held-out test sample. Full precision/recall/latency and hard-slice analysis: [ablation insights](docs/03_ablation_insights.md) and [protocol-v2 comparison CSV](experiments/protocol_v2_seed42/comparison.csv).

| Experiment | Change | Accuracy | F1 | AUC | False-complete |
| --- | --- | ---: | ---: | ---: | ---: |
| E1 | Original audio, attention pooling | 87.61% | 87.87% | 93.64% | **13.96%** |
| E2 | Full augmentation, attention pooling | **87.75%** | **88.14%** | 93.79% | 14.99% |
| E3 | Mean pooling | 86.05% | 86.62% | 92.98% | 17.62% |
| E4 | Last-frame pooling | 87.32% | 87.72% | **93.86%** | 15.36% |
| E5 | Short pauses, 50-250 ms | 87.65% | 88.00% | 93.58% | 14.74% |
| E6 | Long pauses, 600-1500 ms | 87.26% | 87.78% | 93.35% | 16.43% |
| E7 | Fully frozen encoder | 71.53% | 73.54% | 77.51% | 35.61% |
| E8 | First two encoder layers frozen | 87.46% | 87.97% | 93.32% | 16.18% |
| E11 | No silence insertion | 87.12% | 87.38% | 93.53% | 14.45% |
| M1 | Audio + cached Whisper transcript | 87.71% | 88.12% | 93.94% | 15.19% |

Key findings:

- Attention pooling beats mean pooling by 1.52 F1 points and reduces false-complete rate by 2.63 points.
- Full augmentation gains 0.27 F1 points over original audio but worsens false-complete rate by 1.03 points: improved recall is not a universal safety win.
- Last-frame pooling unexpectedly leads AUC and hard-Hinglish proxy F1, contradicting the initial trailing-silence hypothesis.
- Broad and short-pause policies are close; long-only pauses weaken false-complete behavior.
- Partial fine-tuning retains nearly all full-tuning F1 with about 4.45M trainable parameters; the fully frozen encoder fails under the fixed budget.
- M1 adds 0.25 F1 points over matched E1 but raises false-complete rate by 1.24 points and makes live inference 27.6× slower.
- Single-seed differences are directional, not statistical proof.

### Safety finalist study

E1, E4, and E8 were then repeated for six epochs across seeds 42, 43, and 44.
Each checkpoint threshold was calibrated only on validation data. All nine runs
met validation FCR ≤10% and recall ≥85%.

| Architecture | Median validation F1 | Median validation FCR |
| --- | ---: | ---: |
| E1: attention, original data | 87.67% | 9.87% |
| E4: last-frame, augmented data | **89.82%** | **9.21%** |
| E8: attention, partial fine-tuning | 88.34% | 9.87% |

Full results: [safety finalist summary](docs/generated/safety_v1_summary.md).

## Final checkpoint results

Bundled production checkpoint is median-performing E4 seed 44 at `checkpoints/safety_finalist/best.pt`. Its validation-selected threshold is `0.5777403`.

| Split | n | Accuracy | Precision | Recall | F1 | AUC | False-complete |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 904 | 90.04% | 91.06% | 88.62% | 89.82% | 95.53% | 8.55% |
| Held-out test | 4,890 | 86.69% | 89.55% | 83.26% | 86.29% | 94.47% | 9.84% |

Held-out slice results:

| Slice | n | Accuracy | False-complete |
| --- | ---: | ---: | ---: |
| Hindi | 254 | 87.01% | 1.77% |
| Mid-filler | 1,830 | 84.86% | 10.01% |
| End-filler | 1,200 | 94.75% | 5.25% |
| Human audio | 1,071 | 90.01% | 12.31% |
| Hard examples | 1,521 | 92.24% | 7.62% |

Warm model-forward latency for selected run: **24.71 ms CPU**, **4.81 ms GPU**. Feature extraction and cold-start loading are excluded from those benchmark figures; production inference separately returns end-to-end per-request latency.

## Quick start

### Local demo with bundled weights

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run python app.py
```

Open `http://127.0.0.1:7860`. Demo supports upload/microphone input, waveform display, completion probability, decision confidence, and latency.

| Setting | Default | Use |
| --- | --- | --- |
| `--checkpoint PATH` | bundled `best.pt` | Select another checkpoint |
| `TURN_DETECTOR_CHECKPOINT` | unset | Checkpoint when the flag is absent |
| `--host HOST` | `127.0.0.1` | Bind address |
| `--port PORT` | `7860` | Server port |
| `SPACES_ZERO_GPU=1` | unset | Enable the managed Spaces GPU decorator |

Selection order is CLI path, environment variable, then bundled defaults. Cold
startup also loads public Whisper Tiny configuration and feature-extractor
files, so the first run needs Hub access or a populated cache.

### Download weights from Hugging Face and run

After publishing `best.pt` to model repository:

```powershell
uv run python scripts/download_and_demo.py YOUR_USERNAME/YOUR_MODEL_REPO
```

Private repositories work with standard `HF_TOKEN` authentication. Script also accepts `HF_MODEL_REPO_ID`, `--filename`, and `--revision`.

## Training and evaluation

### 1. Prepare bounded data

Start small before default scan budget:

```powershell
uv run python scripts/prepare_data.py `
  --train-scan-budget 2000 `
  --test-scan-budget 500 `
  --n-preview-examples 3
```

Full configured preparation:

```powershell
uv run python scripts/prepare_data.py
uv run python scripts/explore_data.py
```

Exploration writes the canonical report to `docs/data_exploration.md` and the
challenge-requested copy to `notebooks/01_data_exploration.md`.

### 2. Train

```powershell
# One-epoch pipeline check
uv run python src/train.py --config configs/smoke.yaml

# Six-epoch baseline
uv run python src/train.py --config configs/baseline.yaml
```

### 3. Evaluate

```powershell
uv run python src/evaluate.py `
  --checkpoint checkpoints/safety_finalist/best.pt `
  --metadata data/subset/test_split.parquet `
  --output experiments/safety_finalist/test_metrics.json
```

### 4. Reproduce ablations

```powershell
uv run python scripts/run_experiments.py --dry-run --suite full
uv run python scripts/run_experiments.py --suite core --epochs 3
```

Default comparisons use validation data and current baseline safety calibration.
Reproduce finalist study without opening test during candidate runs:

```powershell
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 42 --run-tag safety_v1_seed42
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 43 --run-tag safety_v1_seed43
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 44 --run-tag safety_v1_seed44
uv run python scripts/select_safety_finalist.py experiments/safety_v1_seed42 experiments/safety_v1_seed43 experiments/safety_v1_seed44
```

Final command selects from validation results, copies median-seed winner, then
performs one held-out evaluation. Do not rerun it while tuning.

### 5. Reproduce audio+text ablation

```powershell
uv run python scripts/transcribe_dataset.py --batch-size 16
uv run python scripts/run_multimodal_experiment.py
```

Outputs: [multimodal report](docs/generated/multimodal_report.md) and
`experiments/M1_audio_plus_text/`. Frozen ASR generation is label-independent.
Audio augmentation is disabled for this comparison because cached text must
remain aligned with audio and label.

## Inference API

CLI:

```powershell
uv run python src/inference.py `
  checkpoints/safety_finalist/best.pt `
  path/to/audio.mp3
```

Python API accepts WAV/MP3 paths, NumPy arrays, `(waveform, sample_rate)` tuples, and batches:

```python
from src.inference import TurnDetector

detector = TurnDetector("checkpoints/safety_finalist/best.pt")

single = detector.predict("utterance.wav")  # checkpoint threshold by default
batch = detector.predict_batch(["first.wav", "second.mp3"])

print(single)
# {'prob_complete': 0.91, 'decision': 'complete', 'latency_ms': ...}
```

The API accepts audio paths, NumPy arrays assumed to be 16 kHz, and
`(waveform, sample_rate)` tuples. It rejects empty or non-finite audio and
invalid thresholds. Batch latency is amortized per item. A public service should
limit upload size and duration because files are decoded before the final
eight-second crop. Pass `threshold=0.5` only when an explicit override is wanted;
results include applied threshold.

Passing M1 checkpoint auto-loads frozen Whisper ASR, returns additional
`transcript` field, and includes ASR in end-to-end `latency_ms`.

## Repository layout

```text
.
├── app.py                         # Gradio upload/microphone demo
├── checkpoints/                   # Bundled best submission checkpoint
├── configs/                       # Fixed settings + YAML experiment configs
├── data/samples/                  # Small labeled demo samples
├── docs/                          # Reports, tutorial, codebase notes, diagrams
├── experiments/                   # Machine configs, metrics, histories, comparisons
├── notebooks/
│   ├── 01_data_exploration.md     # Generated dataset findings report
│   ├── 01_data_preparation.ipynb  # Executable data analysis and augmentation study
│   ├── 02_experiment_design.ipynb # Pre-registered comparisons and logging contract
│   └── 03_ablations_and_results.ipynb # Executed results and qualitative errors
├── scripts/
│   ├── download_and_demo.py       # Hub download + one-command demo
│   ├── explore_data.py            # Reproducible exploration report
│   ├── prepare_data.py            # Streaming, splits, previews
│   ├── run_experiments.py         # Controlled audio ablation harness
│   ├── run_multimodal_experiment.py # Matched E1-vs-M1 comparison
│   ├── select_safety_finalist.py   # Multi-seed constrained selection
│   └── transcribe_dataset.py      # Resumable frozen-Whisper transcript cache
├── src/
│   ├── dataset.py                 # Audio dataset and augmentation
│   ├── evaluate.py                # Metrics, slices, latency
│   ├── inference.py               # Production inference API
│   ├── models.py                  # Whisper encoder + pooling/head
│   └── train.py                   # Training/checkpoint loop
├── tests/                         # Fast unittest regression suite
├── requirements.txt               # Minimal pinned demo/Space runtime
└── uv.lock                        # Full transitive environment lock
```

## Reproducibility checks

```powershell
uv run python -m unittest discover -s tests -v
uvx ruff check src scripts tests app.py
uv lock --check
uv pip check
```

Each experiment directory stores resolved config, training history, validation metrics, optional final-test metrics, checkpoint, parameter count, and CPU/GPU latency. Calibrated configs select maximum validation F1 under FCR/recall constraints; test evaluation requires explicit `--final-test`.

## Limitations

- Local training uses bounded subset, not all 270,946 source rows.
- No speaker/accent IDs: true speaker-disjoint evaluation unavailable.
- Dataset contains no verified Hinglish transcripts or code-switch labels. Filler mixing is controlled TTS augmentation, not real human Hinglish benchmark.
- Most local training audio is synthetic; real-world conversational prosody remains underrepresented.
- Noise fallback is synthetic room/HVAC sound unless licensed recordings are added.
- Mid-filler slice remains weakest measured case.
- Broad ablation matrix uses one seed; safety finalists use three seeds, still too few for tight confidence intervals.
- Whisper-tiny transcripts are noisy for Hindi/transliterated speech; text gains may not transfer to real code-switch conversations.
- M1 classifier-only GPU timing is 5.99 ms, but warmed end-to-end batch-1 inference averages 240.15 ms versus 8.71 ms audio-only (20 clips, model load excluded). Cached 32.25 ms/clip ASR is batched throughput, not live latency.
- Validation-calibrated threshold lowers held-out FCR below 10%, but held-out recall is 83.26%, below desired 85% guardrail.

## Future work

1. Collect speaker-consented, human Hinglish conversations with speaker IDs and real code-switch transcripts.
2. Repeat finalist on larger, human speaker-disjoint data with confidence intervals.
3. Recalibrate only on new validation data; do not tune against current held-out result.
4. Add licensed Indian street, office, and phone-channel noise.
5. Distill or quantize model for mobile/edge deployment.
6. Distill text branch or test streaming ASR only if false-complete calibration offsets M1's added interruption risk.

Detailed material: [documentation guide](docs/README.md), [data-preparation notebook](notebooks/01_data_preparation.ipynb), [experiment-design notebook](notebooks/02_experiment_design.ipynb), [ablation notebook](notebooks/03_ablations_and_results.ipynb), [full technical approach](docs/04_full_approach.md), [codebase guide](docs/codebase_guide.md), [data exploration](docs/data_exploration.md), [data preparation](docs/01_data_preparation_approach.md), [experiment plan](docs/02_experiment_plan.md), [ablation insights](docs/03_ablation_insights.md), [bundled checkpoint](docs/submission_checkpoint.md), and [multimodal comparison](docs/generated/multimodal_report.md).
