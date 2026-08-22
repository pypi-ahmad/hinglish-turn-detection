<!-- markdownlint-disable MD013 MD060 -->

# Building a tiny turn detection model for Indian Hinglish

## 1. Problem Understanding

Turn detection decides whether the system should respond after the latest audio
or keep listening. This project uses `complete = 1` and `incomplete = 0`. A
false complete is the more costly error because it interrupts someone who is
still forming a thought.

This is different from voice activity detection. A VAD can tell me that speech stopped, but not whether the speaker finished. Silence after "haan, actually..." and silence after "haan, that is correct" look similar at the waveform level while carrying opposite conversational meaning.

### Why Hinglish makes the boundary difficult

Hinglish combines acoustic, pragmatic, and lexical uncertainty:

- A speaker may switch language inside one clause: "haan, but actually mujhe..." A single clip-level language tag cannot represent that transition.
- Fillers such as "um," "matlab," "actually," "toh," and "yaar" often buy planning time, but they are not reliable incomplete labels. "Bas" or "haan" can be complete turns by themselves.
- Natural pauses are not fixed timers. A short pause may end a turn; a long pause may be word search, hesitation, breath, or network delay.
- Indian English and Hindi prosody can encode continuation through pitch, syllable length, and energy even when the words sound locally complete.
- Transcription is an imperfect fallback. Hindi-English switching, transliterated words, accents, and fillers are exactly where a small ASR model is likely to make mistakes.

The model must combine weak signals: recent acoustic context, endpoint prosody,
timing, filler behavior, and possibly semantics. None is sufficient alone.

### Failure modes I designed around

| Failure mode | Naive shortcut | Product failure |
| --- | --- | --- |
| Mid-sentence hesitation | Silence means complete | Assistant interrupts during word search |
| Trailing silence after a complete sentence | Silence means incomplete | Assistant waits unnecessarily |
| Filler inside a complete turn | Filler means incomplete | Valid answer is ignored |
| Short complete response | Short means incomplete | "haan," "no," or "bas" is missed |
| Long incomplete clause | Long means complete | Length is mistaken for semantic completion |
| Code-switching | One language representation is enough | Boundary cues around switch are missed |
| Synthetic speech | Clean TTS timing is representative | Model fails on real microphones and human hesitation |
| Text dependency | Transcript is fast and correct | ASR latency delays response and ASR errors compound |

This changes the evaluation target. Accuracy alone is not enough. The project
tracks precision, recall, F1, ROC-AUC, and especially false-complete rate (FCR),
then repeats those measurements for Hindi, filler, pause, recorded-audio, and
hard-example slices.

## 2. Data Strategy

### What the supplied data actually supports

The source repository contains 270,946 rows and about 41.4 GB of audio. Current Dataset Viewer statistics cover 35,915 rows and show a nearly balanced endpoint label distribution. The full schema is useful, but limited for this challenge:

- `endpoint_bool` provides the binary target;
- `language` provides one tag per clip;
- nullable `midfiller` and `endfiller` flags provide partial filler information;
- `synthetic` identifies synthetic versus recorded speech;
- `spoken_text` is null;
- speaker, accent, gender, device, and code-switch identifiers are absent.

I used a bounded, decoded local sample so every audio-dependent claim can be
reproduced without an implicit 41 GB download. It contains 7,517 source-train
clips and 4,890 publisher-test clips. The source-train sample is nearly balanced:
3,733 complete and 3,784 incomplete. It includes 721 Hindi and 3,646 English
clips, but a Hindi tag is only a Hinglish proxy, not proof of code-switching.

The working split contains 6,613 training clips and 904 validation clips; the 4,890 publisher-test clips remain separate. IDs do not overlap. Splitting is stratified by language, endpoint label, and source-dataset tag. I cannot claim speaker-disjoint evaluation because no speaker identity exists.

### Gaps and risks found during exploration

The dataset gaps are more informative than the raw row count:

1. **No verified Hinglish annotation.** One language tag cannot reveal within-utterance switching.
2. **No source transcript.** Lexical filler counts require ASR or listening, both of which introduce uncertainty.
3. **Incomplete filler annotation.** About one fifth of sampled source filler fields are null. I preserve annotation provenance instead of silently turning unknown into confirmed absence.
4. **Synthetic-domain bias.** Most available audio is synthetic. TTS may encode cleaner timing, different prosody, and identifiable voices.
5. **No speaker identity.** Random or stratified splits may contain related voices or synthesis systems on both sides.
6. **Long-tail duration.** Local clips range from 0.36 to 32.60 seconds, while the model can only afford a small real-time context window.

Pause analysis also argues against aggressive trimming. With a reproducible
low-energy proxy, 20 ms RMS frames, 10 ms hop, below -40 dBFS, and a 100 ms
minimum, 63.9% of local training clips contain an internal pause and 45.4%
contain trailing silence. Silence occurs in both labels. Removing it would lose
signal, while treating it as a label would create a shortcut.

### Cleaning and normalization

The preparation pipeline performs only operations that preserve endpoint evidence:

1. validate label, ID, decoded samples, and file existence;
2. reject empty, NaN, or infinite waveforms;
3. collapse stereo to mono and normalize to 16 kHz;
4. retain filler value and whether that value was actually annotated;
5. compute duration from decoded samples;
6. retain the last eight seconds for long clips;
7. left-pad short clips so the speech endpoint stays aligned.

The pipeline does not globally trim leading, internal, or trailing silence. For
this task, silence is part of the observation.

### Targeted augmentations and why they exist

All augmentations run online on training data only. Validation and test remain clean. Online sampling avoids storing duplicate audio and presents different perturbations each epoch.

| Augmentation | Policy | Intended lesson | Main risk |
| --- | --- | --- | --- |
| Silence insertion | Default 100-800 ms; incomplete rows receive 1.5× probability | A pause inside speech does not necessarily end a turn | Zero-energy gaps may sound artificial or shift calibration |
| Filler insertion | Internal Hinglish filler audio in both classes; label unchanged | Filler presence alone is not the target | TTS voice or splice boundary may become shortcut |
| Speed | 0.9-1.1× | Speaking rate should not determine completion | Changes natural timing |
| Pitch | ±2 semitones | Voice register should not determine completion | Can damage endpoint intonation if too strong |
| Noise | 5-20 dB SNR | Robustness to office/street-like conditions | Synthetic fallback is not authentic Indian ambience |
| Volume | ±6 dB | Robustness to microphone gain and distance | Clipping and loudness artifacts |

The filler policy preserves the publisher label. Appending "matlab" or
"actually" and changing an example to incomplete would be unsafe: some fillers
and discourse markers can complete a pragmatic turn, and a synthetic filler
voice could become a label watermark. Instead, fillers are inserted internally
into both classes without changing the endpoint label.

Pause augmentation is also applied to both labels. Complete clips can receive trailing silence; incomplete clips can receive internal or trailing silence. This prevents the model from learning either "any silence means incomplete" or "trailing silence means complete." The class probabilities are still asymmetric because false completion is the costlier error, so this choice must be checked through FCR ablations rather than assumed beneficial.

### Hard-example creation

A model can achieve plausible aggregate accuracy with a bad duration rule. To attack that shortcut, I oversample:

- complete clips at or below 1.5 seconds; and
- incomplete clips at or above 4 seconds with a known filler flag.

The current train split yields 1,724 candidates: 17 short complete and 1,707 long incomplete-with-filler clips. That imbalance is informative. This is a false-complete-focused curriculum, not a balanced hard benchmark. Rows receive extra sampler weight rather than physical duplication, and class mass is renormalized so hard mining does not create global label imbalance.

The next version should move from heuristic mining to model-driven mining: collect high-confidence mistakes on clean validation data, listen to the highest-loss Hindi/English clips, distinguish bad labels from genuinely hard examples, and only then upweight confirmed cases.

## 3. Modeling Approach

### Why I started with Whisper Tiny

Whisper Tiny is a strong starting point for three practical reasons.

First, it supplies multilingual acoustic representations learned from much more speech than this bounded training sample. Starting with a small randomly initialized network would force limited local data to teach both speech representation and endpoint behavior.

Second, only the encoder is required. The decoder exists for text generation, but turn detection needs a compact sequence of acoustic states. Removing the decoder leaves an approximately 8.0M-parameter audio model, below the 15M target and small enough for batch-one inference.

Third, words are only part of the signal. Whisper's log-mel encoder retains
pitch, pace, hesitation, and silence, all of which disappear from a text-only
input.

I shorten Whisper's native context to the task: the final eight seconds become 800 mel frames and 400 encoder positions. Existing positional weights are retained rather than reinitialized. This keeps compute focused near the response boundary while preserving the pretrained prior.

The audio path is:

```text
audio → mono 16 kHz → final 8 s → Whisper log-mel features
      → Whisper-tiny encoder → temporal pooling
      → Linear(384, 256) → LayerNorm → GELU → Dropout
      → Linear(256, 64) → GELU → Linear(64, 1)
      → sigmoid → P(turn complete)
```

### Pooling and encoder adaptation

I tested three ways to reduce the encoder sequence to one vector:

- **Mean pooling** asks every real frame to contribute equally.
- **Attention pooling** learns which frames matter.
- **Last-frame pooling** assumes the strongest boundary evidence is closest to the end.

I also varied encoder adaptation. Full fine-tuning updates all four Whisper encoder blocks. Partial fine-tuning freezes the first two and updates the later two. A fully frozen encoder tests whether generic ASR features already contain enough turn-boundary information.

This separates two questions that are often conflated: how much representation capacity the model has, and how much of that representation must adapt to endpoint prosody.

### Alternatives considered

| Alternative | Why it is attractive | Why it was not the first implementation |
| --- | --- | --- |
| VAD plus silence timeout | Extremely small and easy to stream | Detects speech absence, not conversational completion |
| Small CNN/CRNN on log-mel features | Potentially faster and easier to quantize | More local data needed to learn robust multilingual/prosodic features from scratch |
| Tiny Conformer | Streaming-friendly local and global context | Adds architecture and training risk before data assumptions are understood |
| Wav2Vec2/HuBERT encoder | Strong self-supervised acoustic features | Common checkpoints are larger; no clear size advantage for this experiment |
| Text-only classifier | Direct access to incomplete syntax and discourse markers | Misses prosody and requires accurate, low-latency Hinglish ASR |
| Audio + text fusion | Combines prosody with semantics | ASR errors and autoregressive latency sit directly on response path |
| Two-stage acoustic model with semantic fallback | Can reserve text for ambiguous cases | More operational complexity; needs calibrated uncertainty before it is justified |

These remain useful follow-ups, but changing architecture before understanding
the data would make failures harder to diagnose. Whisper Tiny provides a
low-parameter reference for isolating data, pooling, and fine-tuning choices.

### Size, accuracy, and latency trade-offs

Attention audio model has 8,000,386 parameters; selected last-frame model has 7,901,569 parameters and a 30.14 MiB FP32 footprint. Partial freezing reduces trainable parameters to 4.45M, which lowers optimizer-state and gradient memory, but does not reduce inference size by itself. Pooling choice saves little compared with encoder.

Measured audio-only batch-one inference is in the single-digit millisecond range on the test GPU and roughly 25-33 ms on CPU, depending on pooling and run noise. Those figures exclude audio capture and application overhead. The multimodal classifier also looks fast when transcripts are cached, but live end-to-end measurement exposes the real cost: ASR raises mean latency from 8.71 ms to 240.15 ms.

## 4. Experimental Process

### How I kept comparisons interpretable

Experiments use the same split, seed, optimizer, scheduler, three-epoch budget, class sampler, checkpoint rule, and threshold unless the experiment explicitly changes one factor. Checkpoints are selected by validation F1. Test evaluation is a separate explicit step after training; test metrics are not used to select the checkpoint.

The study followed this sequence:

1. establish an unaugmented attention-pooling baseline;
2. enable the complete targeted augmentation policy;
3. hold data constant and compare attention, mean, and last-frame pooling;
4. hold architecture constant and compare short, broad, long, and absent silence insertion;
5. compare full, partial, and frozen encoder training;
6. evaluate cached audio + text fusion against the matched unaugmented audio model;
7. inspect Hindi, filler, internal-pause, and hard-Hinglish proxy errors rather than ranking only by overall F1.

### Historical protocol-v2 held-out results

| Experiment | Main change | F1 | FCR | Main learning |
| --- | --- | ---: | ---: | --- |
| E1 | Original audio, attention pooling | 87.87% | **13.96%** | Strongest overall interruption safety |
| E2 | Full augmentation, attention pooling | **88.14%** | 14.99% | Recall improves, but safety does not |
| E3 | Mean pooling | 86.62% | 17.62% | Equal frame weighting dilutes endpoint evidence |
| E4 | Last-frame pooling | 87.72% | 15.36% | Best AUC and hard-Hinglish proxy F1 |
| E5 | Short 50-250 ms pauses | 88.00% | 14.74% | Too close to broad policy for a firm claim |
| E6 | Long 600-1500 ms pauses | 87.78% | 16.43% | High recall, weaker false-complete behavior |
| E7 | Fully frozen encoder | 73.54% | 35.61% | Generic ASR features need task adaptation |
| E8 | First two encoder layers frozen | 87.97% | 16.18% | Nearly full F1 with 4.45M trainable parameters |
| E11 | No silence insertion | 87.38% | 14.45% | Silence helps hard cases, but effect is not simple |
| M1 | Audio + cached transcript | 88.12% | 15.19% | Small F1 gain cannot justify live ASR cost |

These results use one seed. Differences below roughly one percentage point
suggest directions for later confirmation rather than settle a ranking.

### Insights that changed the approach

#### Augmentation was not a universal improvement

Full augmentation improved overall F1 by 0.27 percentage points and hard-Hinglish proxy F1 by 1.82 points relative to E1, but overall FCR worsened by 1.03 points and hard-proxy FCR by 4.67 points. The augmentation pipeline increased completion recall, but shifted the operating point toward more premature completions.

This changed the conclusion from "more targeted augmentation is better" to
"augmentation policy and decision threshold must be designed together." E2
remains a useful high-recall candidate, but it is not an automatic production
winner.

#### Mean pooling failed; last-frame pooling surprised me

Attention beat mean pooling by 1.52 points overall F1 and reduced FCR by 2.63 points. This supports the idea that endpoint evidence is temporally concentrated.

I expected last-frame pooling to fail on trailing silence. Instead, E4 produced the best AUC and the best hard-Hinglish proxy F1, 88.05%. That result did not prove why it worked, but it invalidated the simple assumption that more temporal aggregation is always safer and motivated the completed three-seed finalist study.

#### Some encoder adaptation is essential; full adaptation may not be

The fully frozen model collapsed to 73.54% F1 and 35.61% FCR under the fixed budget. Whisper's generic ASR representation is not enough by itself. Partial fine-tuning, however, stayed within 0.17 points of full-tuning F1 while updating only 4.45M parameters and improved the hard-proxy trade-off relative to E2.

E8 became the efficiency finalist. Freezing early layers saves training memory
without assuming that the whole encoder is task-ready.

#### Semantics helped the wrong objective

M1 improved Hindi and hard-proxy F1, showing that transcript semantics carry useful information. It also increased false completes and made the live path 27.6 times slower. For a real-time endpoint system, that is a poor exchange.

The text branch is therefore an offline diagnostic, not the main solution. If semantics return, I would use them asynchronously or only for uncertain cases rather than blocking every endpoint decision on ASR.

### What I learned from error analysis

No model dominates all hard slices:

- E1 has the lowest overall and hard-Hinglish FCR.
- E4 has the highest Hindi, Hindi filler-plus-pause, and hard-Hinglish F1.
- E8 offers a strong hard-case compromise with half the trainable parameters.
- Mid-filler performance remains weaker than aggregate performance.
- Some errors are confidently wrong, with complete clips near probability 0.01 and incomplete clips near 0.98. Threshold tuning alone will not repair those examples.

I compare FCR, recall, hard-case F1, size, and latency together rather than name
a winner from one metric.

A deterministic qualitative audit made this trade-off concrete. Four finalists
rescored 35 Hindi filler/pause clips. All four correctly handled both an
incomplete mid/end-filler clip with an internal pause and a complete
mid-filler clip with an internal pause. This confirms that neither filler nor
pause is a valid label shortcut. They also shared two confident failures: one
incomplete filler clip was declared complete by every model, while one
complete filler-plus-pause clip was declared incomplete by every model. A
third incomplete clip split models by training policy: E1 and E4 stayed below
the completion threshold, while augmented E2 and partially tuned E8 crossed
it. This supports the interpretation that augmentation changes calibration as
well as representation. It also identifies examples that need human label and
prosody review before they are reused for hard mining.

## 5. Final Solution & Honest Limitations

### Current final system

The deployable system is an audio-only Whisper Tiny encoder with an eight-second endpoint-aligned window, a small pooling/classification head, and a configurable decision threshold. It accepts files or arrays, normalizes them to mono 16 kHz, supports batch inference, and returns both `P(complete)` and a complete/incomplete decision.

Promotion study repeated E1, E4, and E8 across seeds 42/43/44 for six epochs.
Thresholds were selected only on validation for FCR ≤10%, recall ≥85%, then
maximum F1. E4 won median constrained F1 (89.82%) and median-performing seed 44
became default at threshold `0.5777403`.

One frozen held-out evaluation reached 9.84% FCR, 83.26% recall, and 86.29% F1.
It reduced interruption risk but missed held-out recall guardrail. That result
is informational; changing threshold after seeing it would leak test evidence.

### Strengths of the final system

- Keeps model below 15M parameters; audio variants are about 8M.
- Uses acoustic evidence directly, preserving prosody, fillers, and pause timing.
- Runs fast enough for interactive batch-one inference without ASR.
- Handles raw files, NumPy audio, resampling, batching, and confidence output.
- Avoids simple filler and duration shortcuts through label-preserving augmentation and hard-example sampling.
- Records exact configs, data fingerprints, histories, metrics, sizes, and latency.
- Evaluates interruption risk and hard slices rather than hiding behind accuracy.

### Remaining weaknesses

1. **Hinglish is only approximated.** Hindi tags and synthetic filler insertion do not form a verified code-switch benchmark.
2. **Human prosody is underrepresented.** Synthetic speech dominates available data and may make the measured problem easier than real conversation.
3. **Speaker leakage cannot be ruled out.** No speaker or voice identity is provided.
4. **Pause labels are heuristic.** Low energy is not the same as hesitation, and microphone gain affects the proxy.
5. **Aggregate FCR hides slice failures.** Overall held-out FCR is 9.84%, while human-audio FCR remains 12.31%.
6. **Three seeds are still limited.** Finalist evidence improved, but uncertainty intervals and human-speaker replication remain absent.
7. **The eight-second window loses earlier semantics.** Long-range syntax may matter for clauses whose completion depends on more distant context.
8. **Calibration did not fully transfer.** Validation recall exceeded 85%, but held-out recall was 83.26%.

### What I would do with more time

Priority matters. I would improve evidence before adding model complexity:

1. Collect consented, human-recorded Hinglish conversations with speaker IDs, code-switch transcripts, filler identity, pause boundaries, and ambiguity labels.
2. Build a fixed speaker-disjoint challenge set containing short complete replies, long incomplete clauses, natural fillers, rising intonation, background noise, and real conversational overlap.
3. Repeat E4 on larger human data and report confidence intervals.
4. Recalibrate on a new validation set; never tune against current held-out result.
5. Review high-confidence errors manually and use confirmed mistakes for model-driven hard mining.
6. Add licensed Indian street, office, phone, and household noise instead of relying on synthetic ambience.
7. Test attention-plus-last-frame fusion only on a newly registered validation protocol.
8. Quantize or distill the selected audio model for edge deployment, then
   measure complete request latency as well as classifier time.
9. Revisit semantic features only as an asynchronous or uncertainty-gated fallback.

This work does not reduce to one accuracy number. Data quality and evaluation
design matter at least as much as the encoder choice. Each apparent gain still
needs to be checked against premature interruption, real Hinglish coverage,
and live latency.

### Supporting evidence

- [Documentation guide](README.md)
- [Codebase guide](codebase_guide.md)
- [Data preparation analysis](01_data_preparation_approach.md)
- [Experiment plan](02_experiment_plan.md)
- [Ablation results and error analysis](03_ablation_insights.md)
- [Executed ablation notebook](../notebooks/03_ablations_and_results.ipynb)
- [Structured protocol-v2 results](../experiments/protocol_v2_seed42/all_results.json)
- [Multimodal comparison](../experiments/multimodal_comparison.json)
