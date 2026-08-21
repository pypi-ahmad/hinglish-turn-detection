<!-- markdownlint-disable MD013 MD060 -->

# Data Preparation Approach: Indian Hinglish Turn Detection

## Objective

Predict whether speaker completed a turn (`complete = 1`) or is hesitating,
pausing, or continuing (`incomplete = 0`). Costly error is a false complete:
system interrupts somebody who was not finished.

The [executable data-preparation notebook](../notebooks/01_data_preparation.ipynb)
reproduces local exploration, augmentation previews, hard-case construction,
split audits, and before/after comparisons using production pipeline functions.

This document separates three evidence scopes so sampled observations are not
mistaken for full-dataset facts:

| Scope | Purpose | Size |
|---|---|---:|
| Publisher metadata | Exact repository size and schema | 270,946 rows, 41.4 GB |
| Dataset Viewer statistics | Distribution estimates from current partial statistics scan | 35,915 rows |
| Local decoded subset | Reproducible audio and pipeline analysis | 7,517 train + 4,890 held-out test clips |

Sources: [dataset card](https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train),
[size endpoint](https://datasets-server.huggingface.co/size?dataset=pipecat-ai%2Fsmart-turn-data-v3.2-train),
and [statistics endpoint](https://datasets-server.huggingface.co/statistics?dataset=pipecat-ai%2Fsmart-turn-data-v3.2-train&config=default&split=train).
Statistics checked on 2026-08-21.

## 1. Raw-data findings

### 1.1 Structure and labels

Repository has one `train` split and nine fields:

| Field | Meaning | Preparation consequence |
|---|---|---|
| `audio` | Utterance audio | Decode, validate, convert to mono, resample to 16 kHz |
| `id` | Row identifier | Use for traceability and leakage checks |
| `language` | One language tag per clip | Useful for slicing; cannot identify within-utterance code-switching |
| `endpoint_bool` | Turn-completion target | Authoritative binary label |
| `midfiller`, `endfiller` | Nullable filler annotations | Preserve annotation availability separately from boolean value |
| `synthetic` | Synthetic-versus-recorded indicator | Required evaluation slice for domain-gap analysis |
| `spoken_text` | Declared null type | No supplied transcript for filler or code-switch verification |
| `dataset` | Source-batch tag | Distribution and leakage proxy, not speaker identity |

No speaker ID, accent label, gender field, recording-device field, or
Hinglish/code-switch flag exists. Accent-aware or speaker-disjoint evaluation
cannot be claimed from this schema.

### 1.2 Distribution

Current Dataset Viewer statistics cover 35,915 rows, not all 270,946 rows:

- complete: 17,907 (49.86%); incomplete: 18,008 (50.14%);
- duration: mean 7.62 s, median 7.08 s, standard deviation 3.31 s,
  minimum 0.36 s, maximum 32.60 s;
- synthetic: 29,658 (82.58%); recorded/non-synthetic: 6,257 (17.42%);
- 23 language tags;
- `midfiller` and `endfiller` are each null on 7,310 rows (20.35%).

A full metadata-only projection previously counted English at 65,802 rows
(24.3%) and Hindi at 12,006 rows (4.43%): 5,916 complete and 6,090
incomplete. Bengali (8,006) and Marathi (6,415) provide additional Indian
language variation, but are not substitutes for Hinglish.

Local 7,517-clip training subset contains:

| Measure | Result |
|---|---:|
| Complete / incomplete | 3,733 / 3,784 |
| Hindi | 721 (346 complete / 375 incomplete) |
| English | 3,646 (1,818 complete / 1,828 incomplete) |
| Mean / median duration | 7.84 s / 7.20 s |
| P10 / P90 duration | 3.56 s / 13.12 s |
| Minimum / maximum duration | 0.36 s / 32.60 s |
| Original sample rate | 16 kHz for all retained clips |
| Duplicate IDs / missing audio files | 0 / 0 |

These local counts describe bounded working subset, not complete publisher
dataset.

### 1.3 Silence and pauses

Silence is part of target signal, so it must be measured before deciding
whether to trim it. Local scan uses 20 ms RMS frames with 10 ms hop. A frame
is low energy below RMS 0.01 (-40 dBFS), and a pause requires at least 100 ms
of consecutive low-energy frames.

| Acoustic proxy | Local train subset |
|---|---:|
| Mean low-energy frame share | 17.1% |
| Median low-energy frame share | 17.1% |
| Any pause at least 100 ms | 69.9% |
| Internal pause at least 100 ms | 63.9% |
| Trailing pause at least 100 ms | 45.4% |

Threshold is reproducible, not ground-truth voice activity detection. Quiet
microphones and background noise can move clips across threshold. Result still
establishes that aggressive silence trimming would remove common,
task-relevant information.

Incomplete clips have slightly more internal pauses (65.7% versus 62.1%),
while complete clips have more trailing pauses (48.5% versus 42.3%). Silence
therefore carries information but is not a valid label by itself.

### 1.4 Fillers, code-switching, language, and accent

Local normalized metadata contains 2,445 `midfiller=True` rows and 1,545
`endfiller=True` rows. Values cannot be interpreted as complete filler
prevalence because source annotations are nullable. In Dataset Viewer sample,
20.35% of both filler fields are unknown. Loader output now keeps
`*_annotation_known` flags so unknown is not silently reported as verified
absence.

`spoken_text` is null and `language` contains only one tag per clip.
Consequently:

- code-switching is **unobservable from metadata**, not proven absent;
- lexical counts for “matlab,” “actually,” “tho,” or “yaar” cannot be derived
  without running ASR and accepting its errors;
- Hindi-tagged performance is only a proxy for Hinglish performance;
- accent distribution cannot be measured from supplied fields.

Repository may contain real code-switched speech acoustically, but any stronger
claim requires transcription or manual listening.

## 2. Main challenges for Indian Hinglish

1. **Pause ambiguity.** A 300 ms gap can mean breath, hesitation, word search,
   network delay, or turn completion. Endpoint logic must combine pause with
   preceding prosody and speech context.
2. **Fillers are not labels.** “Um” often signals continuation, but “bas,”
   “haan,” and “wait” can be complete pragmatic turns. Mapping filler presence
   to incomplete creates label noise.
3. **Code-switching is within-utterance.** One language tag cannot represent a
   Hindi clause containing English discourse markers or reverse.
4. **Prosody matters.** Rising pitch, stretched final syllables, incomplete
   syntax, and non-final energy contours may signal continuation. Excessive
   pitch or time augmentation can destroy these cues.
5. **Synthetic-domain gap.** About 82.6% of current statistics sample is
   synthetic. TTS timing, cleanliness, and filler delivery differ from human
   speech and can become shortcuts.
6. **No speaker identity.** Random splitting can place same voice family or
   recording batch on both sides. `dataset` is only a weak proxy.
7. **Asymmetric product cost.** Accuracy can hide interruption failures. Model
   selection must report false-complete rate, especially on Hindi, filler,
   pause, and recorded-audio slices.

## 3. Preparation pipeline

### Step A — Acquire bounded, auditable subset

`stream_filtered_subset` streams rather than materializing 41.4 GB. It keeps
Hindi as primary language, English as Hinglish bridge language, and a small
amount of other-language diversity. Hard row-scan budget makes time and
bandwidth explicit.

Trade-off: Hindi is only 4.43% of unordered dataset. Bounded acquisition is
fast enough for iteration but is not exhaustive or guaranteed representative.
Final training should increase budget or fetch all Hindi/English shards if
resources permit.

### Step B — Validate and normalize audio

For every retained row:

1. require non-null binary `endpoint_bool`;
2. require stable ID and record source tag;
3. decode only mono/stereo audio;
4. reject empty, NaN, or infinite waveforms;
5. collapse channels to mono;
6. resample to 16 kHz and store original sample rate;
7. calculate duration from decoded samples;
8. store filler booleans plus whether each source annotation was known;
9. audit duplicate IDs and missing/corrupt files before splitting.

Do not remove leading, internal, or trailing silence globally. For model input,
keep last eight seconds because endpoint evidence concentrates near boundary.
Left-padding keeps utterance end aligned without deleting pause itself.

### Step C — Define and refine labels

- `endpoint_bool=True` maps to complete (1); false maps to incomplete (0).
- Nullable filler fields remain metadata. Unknown is normalized to false only
  for boolean masks and accompanied by `*_annotation_known=False`. Legacy
  caches that already lost source nulls use null provenance rather than a guess.
- Silence, filler presence, duration, and language never override publisher
  labels automatically.
- Default filler augmentation is internal and label-preserving in **both**
  classes. This avoids teaching “TTS filler voice = incomplete.”
- Truncating complete clip and appending filler remains optional weak-label
  primitive, but default dataset path does not use it. It needs manual
  listening or dedicated ablation before use.

Ambiguous or contradictory examples should enter review queue. Useful future
protocol: two independent annotators plus adjudication using question: “Would
interrupting immediately after this audio clip be natural?”

### Step D — Apply targeted training-only augmentation

Validation and test audio remain clean. Training transforms are sampled online
each epoch, avoiding large static copy and providing more variation.

| Augmentation | Policy | Expected impact | Main risk / control |
|---|---|---|---|
| Pause insertion | 100–800 ms; 1.5× probability for incomplete rows; internal or trailing for incomplete, trailing for complete | Reduces false completion during hesitation; prevents “trailing silence = complete” shortcut | Synthetic zero-energy gaps can sound unnatural; compare short-versus-long pause ablations |
| Hinglish filler insertion | “um”, “uh”, “matlab”, “actually”, “tho/toh”, “yaar”, “bas”, “wait”, “ek second”, “haan”, etc.; inserted internally into both classes; label unchanged | Teaches robustness to Hindi/English discourse markers and some code-switched lexical content | TTS voice and splice may become artifacts; use Indian voices, crossfades, both labels, and manual previews |
| Speed perturbation | 0.9–1.1× | Robustness to speaking rate | Alters pause duration and timing; keep range narrow |
| Pitch shift | ±2 semitones | Robustness to voice register | Can distort final intonation; keep contour-preserving range and ablate |
| Background noise | 5–20 dB SNR | Robustness to office/street conditions | Synthetic fallback is not authentic Indian ambience; report honestly and prefer licensed recordings |
| Volume perturbation | ±6 dB | Robustness to mic gain and distance | Clipping; output is clipped to valid range |

Augmentation is not proof of real Hinglish coverage. Highest-value data
addition remains consented, human-recorded Hinglish containing natural fillers,
interruptions, and hesitation.

### Step E — Mine hard examples

Current rule identifies:

- short complete clips (≤1.5 s), which break “short means unfinished”; and
- long incomplete clips (≥4 s) with known filler flag, which break “long means
  complete after enough speech.”

Rows receive extra sampler weight rather than being copied. Sampler
renormalizes total mass per endpoint class, preventing global class imbalance.
Current train split yields 1,724 candidates: 17 short complete and 1,707 long
incomplete-with-filler. This asymmetry matters: fixed thresholds capture many
more incomplete examples. Treat it as false-complete-focused curriculum, not a
balanced benchmark.

After baseline exists, replace purely heuristic mining with model-driven
mining:

1. collect high-confidence false completes and false incompletes on clean
   validation data;
2. slice by language, synthetic status, filler availability, duration, and
   pause profile;
3. manually review highest-loss Hindi/English clips;
4. upweight confirmed hard cases only in training;
5. keep fixed, untouched challenge set for honest comparison.

### Step F — Split without overclaiming

Publisher's separate `smart-turn-data-v3.2-test` repository is final held-out
set. Train and validation are stratified by
`(language, endpoint_bool, source_dataset)` so rare language/class/source
combinations remain represented.

Current local split:

| Split | Rows | Complete | Hindi | English |
|---|---:|---:|---:|---:|
| Train | 6,613 | 3,285 | 634 | 3,207 |
| Validation | 904 | 448 | 87 | 439 |
| Held-out test | 4,890 | 2,461 | 254 | 1,507 |

ID overlap between all local splits is zero. All 12 observed source tags occur
across train, validation, and test, so evaluation is **not** source-disjoint or
speaker-disjoint.

Why not group by source now? Several source tags are language-specific; source
group split can remove entire language slices from one partition. Current
stratified split is safer small-data compromise. If speaker or voice IDs become
available, group by those identities first, then stratify at group level. Also
add waveform fingerprints or perceptual hashes to detect duplicate audio with
different IDs.

## 4. Before-and-after preparation contract

“After” means auditable preparation and training-time transformations, not
claiming synthetic examples are equivalent to newly collected human speech.

| Raw state | Prepared state | Reason |
|---|---|---|
| Decoded audio with variable channel assumptions | Validated finite mono waveform, resampled to 16 kHz | Stable model input and comparable duration statistics |
| Publisher boolean | Explicit `complete=1`, `incomplete=0`; no heuristic relabeling | Preserve task semantics and avoid silent label corruption |
| Nullable filler fields | Boolean mask plus annotation-known provenance | Unknown annotation is not verified filler absence |
| Silence present but unquantified | Reproducible frame-energy proxy; silence preserved | Measure pause signal without treating it as ground truth |
| Single language tag, no Hinglish label | Hindi/English proxy slices plus honest evidence boundary | Avoid overstating code-switch coverage |
| Duration and filler shortcuts available | Dedicated short-complete and long-incomplete hard set | Expose shortcut learning instead of reinforcing it |
| One publisher train split | Seeded stratified train/validation split plus separate publisher test | Comparable experiments with untouched final evaluation |
| Clean waveform only | Online, train-only pause/filler/speed/pitch/noise/volume variants | Improve robustness without contaminating validation or test |

Notebook compares waveform duration, RMS, peak amplitude, and low-energy share
before and after each targeted transform. Dataset-level “after” statistics are
reported as a strategy audit because online augmentation changes across epochs;
a single materialized augmented corpus would misrepresent that distribution.

## 5. Evaluation slices required by this strategy

Overall accuracy and F1 are insufficient. Report:

- precision, recall, F1, ROC-AUC, and false-complete rate;
- Hindi and English separately;
- synthetic versus recorded audio;
- known mid-filler and end-filler rows, excluding unknown annotations;
- internal-pause and trailing-pause buckets;
- short complete and long incomplete hard cases;
- duration buckets and source-dataset tags;
- manually curated Hinglish challenge set once available.

Tune decision threshold on validation data for product cost, not fixed 0.5 by
habit. Test data remains untouched until model and threshold are frozen.

## 6. Expected outcomes and trade-offs

| Decision | Expected benefit | Accepted cost |
|---|---|---|
| Preserve silence | Retains primary turn-taking signal | Longer inputs and sensitivity to background noise |
| Keep last 8 seconds | Focuses compute near endpoint | Loses earlier lexical context on long utterances |
| Preserve filler labels | Avoids semantic label corruption and TTS-label shortcut | Does not manufacture extra incomplete labels |
| Online augmentation | More variability with little disk growth | Exact sample changes by epoch; seeds and configs must be logged |
| Bounded language-focused scan | Faster iteration on Hindi/English | Sampling bias and incomplete Hindi coverage |
| Stratify by source tag | Stable slice coverage | Does not measure unseen-speaker generalization |
| Synthetic noise fallback | Pipeline works without unlicensed assets | Does not represent real Indian street/office acoustics |

## 7. Reproducibility

```powershell
uv run python scripts/prepare_data.py
uv run python scripts/explore_data.py
uv run --with jupyterlab jupyter lab notebooks/01_data_preparation.ipynb
uv run python -m unittest discover -s tests
```

Generated artifacts:

- `data/subset/train_meta_raw.parquet`: retained metadata and local paths;
- `data/subset/train_split.parquet`, `val_split.parquet`, `test_split.parquet`;
- `data/subset/hard_negative_split.parquet`: inspectable hard-case manifest;
- `data/samples/`: labeled clips and before/after augmentation previews;
- `notebooks/01_data_preparation.ipynb`: executable analysis with saved outputs;
- `docs/data_exploration.md`: statistics generated from current local
  artifacts.

Every reported number should identify whether it came from full metadata,
partial Dataset Viewer statistics, or local decoded subset. This prevents a
central data-quality failure: precise-looking numbers whose sampling scope is
unclear.
