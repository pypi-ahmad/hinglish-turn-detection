<!-- markdownlint-disable MD013 MD025 -->

# Hinglish turn detection

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Live Space](https://img.shields.io/badge/demo-Hugging%20Face-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/pypi-ahmad/hinglish-turn-detection)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

> [!TIP]
> Try the [live demo](https://pypi-ahmad-hinglish-turn-detection.hf.space) or view the [Hugging Face Space](https://huggingface.co/spaces/pypi-ahmad/hinglish-turn-detection).

This repository answers one voice-agent question: **has the user finished speaking, or are they pausing mid-thought?** It targets Indian Hindi/English speech, where fillers and pauses are easy to misread. The classifier is small enough for low-latency serving.

The model uses [Pipecat Smart Turn v3.2](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train), an `openai/whisper-tiny` encoder, last-frame pooling, and a binary classification head. It has **7.9M parameters** and predicts `P(turn complete)` from the final eight seconds of audio.

## What this repository contains

- The source has 270,946 rows, but no speaker IDs, transcripts, accent labels,
  or verified Hinglish/code-switch labels. The preparation pipeline records
  those limits, decodes a bounded sample, keeps silence, and distinguishes
  unknown filler annotations from confirmed absence.
- Augmentation targets observed failure cases. Fillers retain their original
  labels, pause probability depends on class, and acoustic transforms stay
  restrained. Hard examples appear more often without teaching the model that
  every filler means incomplete.
- The study includes nine audio ablations and one matched audio-text comparison.
  No setup won every metric. Full augmentation had the best overall F1, original
  audio had the lowest FCR, and last-frame pooling had the best hard-Hinglish
  proxy F1. Partial fine-tuning remained close with 4.45M trainable parameters.
- Mean pooling lost endpoint information and a fully frozen encoder performed
  poorly. Live ASR also made multimodal inference 27.6 times slower and raised
  interruption risk.
- A separate safety study trained E1, E4, and E8 for six epochs with seeds 42,
  43, and 44. Thresholds came from validation data only, with FCR ≤10% and
  recall ≥85%. E4 had the best median F1, and seed 44 became the production
  checkpoint.
- On held-out data, the selected checkpoint reached 9.84% FCR and 83.26%
  recall. The next step is human, speaker-disjoint Hinglish data. The held-out
  result must not be reused to tune the threshold.

For the full rationale, read the [technical approach](docs/04_full_approach.md). The [documentation reading guide](docs/README.md) lists the remaining reports in order.

The [interactive tutorial](tutorial/index.html) covers the problem, data, model, experiments, deployment, and interview explanation in less detail.

## Problem statement

Voice activity detection can tell whether sound stopped. It cannot tell why. A speaker may be finished, searching for next word, or saying "matlab...", "haan...", "actually...", or "ek second..." before continuing.

The two errors have different costs:

- **False complete:** assistant interrupts user mid-thought.
- **False incomplete:** assistant waits slightly longer.

The reports include accuracy and F1, but the main safety metric is the **false-complete rate**:

```text
P(predict complete | utterance is actually incomplete)
```

## How the solution was built

The work started with failure cases rather than a preferred model:

1. Inspect the data. The source has 270,946 rows and 23 languages, but
   no usable transcripts, speaker IDs, accent IDs, or explicit Hinglish label.
2. Bound acquisition. Fixed scan budgets avoid an unplanned 41.4 GB download and
   record exactly what the local sample contains.
3. Reduce duration shortcuts by sampling short complete turns and long
   incomplete turns with fillers more often.
4. Use pause, speed, pitch, volume, noise, and filler transforms only for known
   deployment failures.
5. Match Whisper's context to the task by retaining its first 400 learned
   encoder positions for an eight-second input.
6. Report FCR alongside Hindi, filler, synthetic/human, pause, and hard-case slices.
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

The pipeline decodes audio to mono 16 kHz WAV and keeps the final eight seconds, where turn-ending cues are most likely. It left-pads short clips to keep the last speech frame aligned.

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

Stochastic transforms run during training, so the pipeline stores no duplicate audio and each epoch receives different perturbations. Filler audio is inserted into clips from both classes without changing the publisher label. A filler alone is not reliable evidence that the speaker will continue.

The split is stratified by `(language, endpoint_bool, source_dataset)`. Speaker-disjoint evaluation is impossible because the source has no speaker identity. The [data preparation approach](docs/01_data_preparation_approach.md) explains the choices in detail.

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

Multimodal M1 uses the same audio branch and generates transcripts offline with
frozen `openai/whisper-tiny`. It mean-pools learned 64-dimensional Whisper-token
embeddings, joins them with the 384-dimensional audio embedding, and uses the
same small classifier. The model has 11,336,130 parameters. Cached transcripts
make training practical, but live inference still pays for autoregressive ASR.

## Experiments

The ablations below use the same split, seed 42, and three-epoch budget. Each checkpoint was selected by validation F1, then evaluated once on the held-out test sample. The [ablation insights](docs/03_ablation_insights.md) and [protocol-v2 comparison CSV](experiments/protocol_v2_seed42/comparison.csv) contain precision, recall, latency, and hard-slice results.

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

What the experiments showed:

- Attention pooling beat mean pooling by 1.52 F1 points and reduced the false-complete rate by 2.63 points.
- Full augmentation gained 0.27 F1 points over original audio, but its false-complete rate was 1.03 points worse. Better recall did not make it safer overall.
- Last-frame pooling had the best AUC and hard-Hinglish proxy F1, despite the initial concern that trailing silence would make the final frame unreliable.
- Broad and short-pause policies are close; long-only pauses weaken false-complete behavior.
- Partial fine-tuning retained nearly all full-tuning F1 with about 4.45M trainable parameters. The fully frozen encoder performed poorly under the fixed budget.
- M1 added 0.25 F1 points over matched E1, raised the false-complete rate by 1.24 points, and made live inference 27.6× slower.
- Single-seed differences are directional, not statistical proof.

### Safety finalist study

E1, E4, and E8 then ran for six epochs across seeds 42, 43, and 44. Each
checkpoint threshold was calibrated only on validation data. All nine runs met
validation FCR ≤10% and recall ≥85%.

| Architecture | Median validation F1 | Median validation FCR |
| --- | ---: | ---: |
| E1: attention, original data | 87.67% | 9.87% |
| E4: last-frame, augmented data | **89.82%** | **9.21%** |
| E8: attention, partial fine-tuning | 88.34% | 9.87% |

Full results: [safety finalist summary](docs/generated/safety_v1_summary.md).

## Final checkpoint results

The bundled production checkpoint is E4 seed 44, the median-performing run for the winning architecture. It is stored at `checkpoints/safety_finalist/best.pt` and uses the validation-selected threshold `0.5777403`.

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

The selected run has warm model-forward latency of **24.71 ms on CPU** and **4.81 ms on GPU**. These benchmarks exclude feature extraction and cold-start loading. Production inference reports end-to-end latency for each request.

## Quick start

### Local demo with bundled weights

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run python app.py
```

Open `http://127.0.0.1:7860`. The demo accepts uploaded audio or microphone input and shows the waveform, completion probability, decision confidence, and latency.

| Setting | Default | Use |
| --- | --- | --- |
| `--checkpoint PATH` | bundled `best.pt` | Select another checkpoint |
| `TURN_DETECTOR_CHECKPOINT` | unset | Checkpoint when the flag is absent |
| `--host HOST` | `127.0.0.1` | Bind address |
| `--port PORT` | `7860` | Server port |
| `SPACES_ZERO_GPU=1` | unset | Enable the managed Spaces GPU decorator |

Checkpoint selection checks the CLI path, then the environment variable, then
the bundled default. A cold start also loads the public Whisper Tiny
configuration and feature-extractor files, so the first run needs Hub access
or a populated cache.

### Download weights from Hugging Face and run

After publishing `best.pt` to model repository:

```powershell
uv run python scripts/download_and_demo.py YOUR_USERNAME/YOUR_MODEL_REPO
```

Private repositories use standard `HF_TOKEN` authentication. The script also accepts `HF_MODEL_REPO_ID`, `--filename`, and `--revision`.

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

The exploration command writes its main report to `docs/data_exploration.md`
and a challenge-requested copy to `notebooks/01_data_exploration.md`.

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

Default comparisons use validation data and the current baseline safety
calibration. These commands reproduce the finalist study without opening the
test set during candidate runs:

```powershell
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 42 --run-tag safety_v1_seed42
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 43 --run-tag safety_v1_seed43
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 44 --run-tag safety_v1_seed44
uv run python scripts/select_safety_finalist.py experiments/safety_v1_seed42 experiments/safety_v1_seed43 experiments/safety_v1_seed44
```

The final command selects from validation results, copies the median-seed
winner, and performs one held-out evaluation. Do not run it again while tuning.

### 5. Reproduce audio+text ablation

```powershell
uv run python scripts/transcribe_dataset.py --batch-size 16
uv run python scripts/run_multimodal_experiment.py
```

The [multimodal report](docs/generated/multimodal_report.md) and
`experiments/M1_audio_plus_text/` contain the outputs. Frozen ASR generation
does not use labels. Audio augmentation is disabled because cached text must
stay aligned with its audio and label.

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
invalid thresholds. Batch latency is amortized per item. Public services should
limit upload size and duration because decoding happens before the final
eight-second crop. Pass `threshold=0.5` only for an explicit override. Results
include the threshold that was applied.

Passing an M1 checkpoint loads frozen Whisper ASR, adds a `transcript` field to
the result, and includes ASR time in end-to-end `latency_ms`.

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

Each experiment directory stores its resolved config, training history,
validation metrics, optional final-test metrics, checkpoint, parameter count,
and CPU/GPU latency. Calibrated configs select the highest validation F1 that
meets the FCR and recall constraints. Test evaluation requires an explicit
`--final-test`.

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

The [documentation index](#documentation-index) links to every project report. The executable notebooks are [data preparation](notebooks/01_data_preparation.ipynb), [experiment design](notebooks/02_experiment_design.ipynb), and [ablations and results](notebooks/03_ablations_and_results.ipynb).

## Documentation index

### Project guide

- [What this repository contains](#what-this-repository-contains)
- [Problem statement](#problem-statement)
- [How the solution was built](#how-the-solution-was-built)
- [Data preparation](#data-preparation)
- [Model architecture](#model-architecture)
- [Experiments](#experiments)
- [Final checkpoint results](#final-checkpoint-results)
- [Quick start](#quick-start)
- [Training and evaluation](#training-and-evaluation)
- [Inference API](#inference-api)
- [Repository layout](#repository-layout)
- [Reproducibility checks](#reproducibility-checks)
- [Limitations](#limitations)
- [Future work](#future-work)

### Documentation

- Study and submission
  - [Documentation guide](docs/README.md)
  - [Data preparation approach](docs/01_data_preparation_approach.md)
  - [Experiment plan](docs/02_experiment_plan.md)
  - [Ablation results and engineering insights](docs/03_ablation_and_insights.md)
  - [Ablation study and experimental insights](docs/03_ablation_insights.md)
  - [Full approach](docs/04_full_approach.md)
  - [Zero-to-mastery tutorial](docs/05_zero_to_mastery_tutorial.md)
  - [Data exploration](docs/data_exploration.md)
  - [Submission checkpoint](docs/submission_checkpoint.md)
- Codebase references
  - [Codebase guide](docs/codebase_guide.md)
  - [Architecture and runtime flow](docs/codebase/ARCHITECTURE.md)
  - [Known concerns](docs/codebase/CONCERNS.md)
  - [Development conventions](docs/codebase/CONVENTIONS.md)
  - [External integrations](docs/codebase/INTEGRATIONS.md)
  - [Technology stack](docs/codebase/STACK.md)
  - [Repository structure](docs/codebase/STRUCTURE.md)
  - [Testing and verification](docs/codebase/TESTING.md)
- Diagrams
  - [System architecture](docs/diagrams/system-architecture.mmd)
  - [Data lifecycle](docs/diagrams/data-lifecycle.mmd)
  - [Module dependencies](docs/diagrams/module-dependencies.mmd)
  - [Inference sequence](docs/diagrams/inference-sequence.mmd)
- Generated reports
  - [Protocol-v2 seed-42 ablation report](docs/generated/protocol_v2_seed42_ablation_report.md)
  - [Multimodal ablation report](docs/generated/multimodal_ablation_report.md)
  - [Audio-only vs audio+text report](docs/generated/multimodal_report.md)
  - [Safety finalist report, seed 42](docs/generated/safety_v1_seed42_ablation_report.md)
- [Safety finalist report, seed 43](docs/generated/safety_v1_seed43_ablation_report.md)
- [Safety finalist report, seed 44](docs/generated/safety_v1_seed44_ablation_report.md)
- [Safety finalist summary](docs/generated/safety_v1_summary.md)

## References

- [Pipecat Smart Turn v3.2 dataset](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train)
- [OpenAI Whisper Tiny](https://huggingface.co/openai/whisper-tiny)
- [Hugging Face Space](https://huggingface.co/spaces/pypi-ahmad/hinglish-turn-detection) and [live demo](https://pypi-ahmad-hinglish-turn-detection.hf.space)
- [uv documentation](https://docs.astral.sh/uv/)
