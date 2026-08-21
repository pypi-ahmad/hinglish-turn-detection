<!-- markdownlint-disable MD013 -->

# Zero to mastery: Hinglish turn detection

## 1. Introduction

A voice assistant needs to know when to answer. Silence alone is not enough. A
speaker may have finished, or may be pausing to find the next word. Turn
detection makes that decision:

- `1`, complete: the assistant may respond.
- `0`, incomplete: the assistant should keep listening.

This sits after voice activity detection. VAD tells us that speech stopped;
turn detection asks why it stopped. A wrong complete decision interrupts the
speaker. A wrong incomplete decision usually adds delay. That is why this
project tracks false-complete rate as well as F1.

Hinglish makes the boundary harder. A speaker may switch languages inside one
clause, pause between Hindi and English words, or use fillers such as "matlab",
"actually", "toh", "yaar", and "haan". A filler often buys thinking time, but
it can also be a complete reply. Prosody matters too: rising intonation can
signal that a thought is unfinished even when the words look complete.

## 2. What this project builds

The project contains a small audio classifier, a repeatable training pipeline,
experiment reports, and a Gradio demo. Its main path is:

```text
audio
  -> mono 16 kHz waveform
  -> final 8 seconds
  -> Whisper log-mel features
  -> Whisper Tiny encoder
  -> pooling and classifier
  -> P(turn complete)
```

The bundled safety-finalist model has 7,901,569 parameters. It uses the Whisper encoder,
not the text decoder. The repository also contains an optional audio-plus-text
model for comparison. The public submission consists of this repository, its
canonical checkpoint, and the Gradio Space linked from the root README.

## 3. The data journey

### Source data

The project uses `pipecat-ai/smart-turn-data-v3.2-train` and the publisher's
separate test repository. The full train repository has 270,946 rows and 23
language tags. Local training used a bounded decoded sample rather than silently
pulling the full 41.4 GB source.

The local source-train sample has 7,517 clips: 3,733 complete and 3,784
incomplete. It includes 721 Hindi clips and 3,646 English clips. The working
split contains 6,613 training rows and 904 validation rows. A separate 4,890-row
publisher-test sample is held out.

### What the source does not provide

The source has no speaker or accent identity, so the split cannot be truly
speaker-disjoint. `spoken_text` is null, which means filler counts and actual
code-switching cannot be measured directly. A Hindi language tag is only a
Hinglish proxy. Some filler flags are nullable, so null means unknown rather
than absent.

These limits matter. The reports never call this a verified human Hinglish
benchmark.

### Cleaning

Each retained clip is decoded, checked for empty or non-finite samples, reduced
to mono, resampled to 16 kHz, and saved as WAV. Long clips keep their final eight
seconds because the turn boundary is near the end. Short clips are left-padded
during feature extraction so their last speech frame stays aligned.

Publisher labels remain authoritative. Duration, silence, language, and filler
metadata do not relabel a clip automatically.

### Targeted augmentation

Augmentation runs during training, so every epoch can draw a different version.
Validation and test audio stay clean.

Silence insertion adds 100 to 800 ms by default. Incomplete clips receive 1.5
times the base probability because hesitation is the expensive error case.
Complete clips can still receive trailing silence, which stops the model from
learning that silence always means incomplete.

Filler injection uses Hindi and Indian-English TTS for words such as "um",
"matlab", "actually", "yaar", "bas", and "ek second". Fillers are inserted
inside examples from both classes without changing the label. Otherwise the
model could learn that a synthetic filler voice means incomplete.

Speed changes stay between 0.9 and 1.1 times. Pitch moves by at most two
semitones, volume by at most 6 dB, and noise is mixed at roughly 5 to 20 dB SNR.
Real noise files take precedence; a synthetic room-noise fallback keeps the
pipeline runnable. That fallback does not prove robustness to Indian streets or
offices.

### Hard examples and splits

The sampler gives extra weight to short complete clips and long incomplete
clips with fillers. These examples weaken an easy duration shortcut. The split
is stratified by language, endpoint label, and source tag because no speaker ID
is available. Validation chooses models. Test data estimates the performance of
an already frozen choice.

## 4. Model architecture

Whisper Tiny is a useful starting point because it is small and already knows
multilingual speech features. The code loads `WhisperModel`, keeps its encoder,
and discards the decoder. The native positional table is shortened to 400
encoder positions, matching the eight-second input window.

The encoder returns 400 vectors, each 384 values wide. Pooling turns that
sequence into one vector:

- Attention pooling learns which frames matter.
- Mean pooling averages valid frames.
- Last-frame pooling uses the final valid representation.

The classifier is `384 -> 256 -> 64 -> 1`, with layer normalization, GELU, and
dropout. Its final number is a logit. Sigmoid converts it to a probability.
Threshold 0.5 was used for historical controlled comparisons. Production
checkpoint stores validation-selected threshold `0.5777403`; inference uses it
automatically unless caller explicitly overrides it.

The multimodal model adds a 64-dimensional average of Whisper transcript-token
embeddings. It has 11,336,130 parameters. Live use must also run autoregressive
ASR, which makes it much slower than the audio-only path.

## 5. Training

Training uses binary cross-entropy with logits. AdamW starts at `5e-5`, with
weight decay `0.01`; a cosine scheduler follows a 20 percent warmup. Mixed
precision reduces GPU memory. Four batches are accumulated before an optimizer
step, gradients are clipped to norm 1.0, and checkpoint selection uses
validation F1 with configured FCR/recall feasibility gate.

One epoch works like this:

1. The dataset loads and optionally augments a waveform.
2. The collator left-pads audio and extracts Whisper features.
3. The model returns one logit per clip.
4. BCE loss measures prediction error.
5. Backpropagation accumulates gradients.
6. The optimizer and scheduler step at the configured interval.
7. Clean validation computes accuracy, precision, recall, F1, AUC, and FCR.
8. Calibration searches validation probabilities for maximum F1 while requiring
   FCR ≤10% and recall ≥85%.
9. Better feasible operating point replaces `best.pt`; threshold and confusion
   counts are stored with weights.

After training, script reloads saved checkpoint and confirms calibrated
validation F1 matches stored value.

## 6. Experiments and lessons

The controlled study changes one main decision at a time. It compares original
and augmented data, three pooling methods, pause ranges, frozen and partially
frozen encoders, and audio with audio-plus-text.

Full augmentation reaches the highest overall F1, 88.14 percent, but its FCR is
worse than the unaugmented control. This is a useful warning: a higher symmetric
score can still increase interruption risk.

Mean pooling loses endpoint detail. Last-frame pooling performs better than the
original hypothesis expected, especially on the hard Hinglish proxy. A fully
frozen encoder fails badly under the fixed budget, while freezing the first two
layers keeps nearly all overall F1 with 4.45 million trainable parameters.

Text adds only 0.25 F1 points over its matched audio control, worsens FCR, and
raises warmed end-to-end mean latency from 8.71 ms to 240.15 ms. It remains an
offline diagnostic rather than the default serving model.

Follow-up repeated E1, E4, and E8 for six epochs across seeds 42/43/44. E4 won
median constrained validation F1 (89.82%) and seed 44 became default. Frozen
held-out evaluation reached 9.84% FCR, 83.26% recall, and 86.29% F1. Lower FCR
therefore came with real recall cost; test result was not used to retune.

## 7. Code walkthrough

`configs/config.py` defines shared paths, audio constants, dataset IDs, fillers,
and voices. Experiment YAML files hold pooling, layer freezing, augmentation,
batching, optimization, and checkpoint settings.

`scripts/prepare_data.py` streams source rows, writes WAV and Parquet files,
creates splits, marks hard cases, and saves examples for listening.

`src/dataset.py` contains audio validation, pause measurements, splits,
augmentation functions, `FillerBank`, `TurnDetectionDataset`, and `collate_fn`.

`src/models.py` defines pooling layers, the audio classifier, the multimodal
classifier, and `build_model`.

`src/train.py` creates balanced loaders, optimizer, scheduler, scaler, training
loop, validation loop, and best checkpoint. `src/evaluate.py` adds slice metrics,
confident errors, model size, and latency.

`src/inference.py` is the integration boundary. `TurnDetector` accepts a file,
NumPy array, or `(waveform, sample_rate)` tuple. `predict_batch` handles several
clips in one forward pass. `app.py` adds upload, microphone, examples, waveform,
confidence, and latency to a queued Gradio interface.

## 8. Running the project

```powershell
uv sync

# Small data preparation run
uv run python scripts/prepare_data.py --train-scan-budget 2000 --test-scan-budget 500

# One-epoch training check
uv run python src/train.py --config configs/smoke.yaml

# Full configured baseline
uv run python src/train.py --config configs/baseline.yaml

# Demo
uv run python app.py
```

Evaluation and experiment commands are listed in the root README. The app binds
to `127.0.0.1:7860` by default. `--checkpoint`, `--host`, and `--port` override
those settings. `TURN_DETECTOR_CHECKPOINT` supplies a checkpoint when the flag
is absent.

Programmatic thresholds must be finite values from 0 to 1. Bare NumPy arrays
are assumed to be 16 kHz. Batch latency is amortized per item. A public service
should limit upload size and duration because file decoding happens before the
final eight-second crop.

## 9. Deployment

The GitHub repository uses Git LFS for the canonical checkpoint. The root README
also acts as the Hugging Face Space card and declares Gradio, Python 3.12, and
`app.py`.

```powershell
hf auth login
hf repos create YOUR_USERNAME/hinglish-turn-detection --type space --sdk gradio --flavor zero-a10g --public
hf upload YOUR_USERNAME/hinglish-turn-detection . --type space
hf spaces info YOUR_USERNAME/hinglish-turn-detection --expand runtime
hf spaces logs YOUR_USERNAME/hinglish-turn-detection --tail 200
```

Cold startup downloads public Whisper Tiny assets unless the cache already has
them. A `RUNNING` status is not enough: open the interface, invoke its generated
`predict` API with real audio, and inspect logs.

## 10. Limits and next work

The current data lacks verified Hinglish labels, speaker IDs, and human pause
annotations. Most training audio is synthetic. The source sample is bounded,
the broad study uses one seed, while finalist study uses three. Production
threshold is validation-calibrated, but held-out recall missed 85% target. The
public demo also has no repository-level upload limit.

The next useful dataset would contain consented human Hinglish conversations,
speaker IDs, transcripts, code-switch labels, filler identity, pause boundaries,
and ambiguity reviews. E4 should then be replicated and recalibrated on a new
speaker-disjoint validation/challenge split. Current held-out result must remain
untouched. Quantization or distillation only makes sense after that evidence.

## 11. Summary

Turn detection is a semantic timing problem, not silence detection. The project
uses a small multilingual encoder and treats false completion as a separate
safety metric. Its strongest lesson is about data and evaluation: augmentation
changes the operating point, and aggregate F1 does not tell us whether a voice
assistant will interrupt people less often.
