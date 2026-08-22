<!-- markdownlint-disable MD013 MD060 -->

# Ablation study and experimental insights

## Safety finalist update

The later promotion study repeated the relevant architectures and calibrated
their thresholds. E1, E4, and E8 trained for six epochs with seeds 42, 43, and
44. Each run chose its threshold on validation data, maximizing F1 subject to
FCR ≤10% and recall ≥85%. All nine selected checkpoints met those validation
constraints.

| Architecture | Seed 42 F1/FCR | Seed 43 F1/FCR | Seed 44 F1/FCR | Median F1/FCR |
| --- | ---: | ---: | ---: | ---: |
| E1 | 88.59 / 9.87 | 87.19 / 9.87 | 87.67 / 9.65 | 87.67 / 9.87 |
| E4 | 88.08 / 9.87 | 89.89 / 9.21 | 89.82 / 8.55 | **89.82 / 9.21** |
| E8 | 88.21 / 9.87 | 88.34 / 9.87 | 88.81 / 8.55 | 88.34 / 9.87 |

E4 had the best median result. Its median-performing run, seed 44, was frozen
at threshold `0.5777403` before one held-out test evaluation.

| Operating point | Accuracy | Precision | Recall | F1 | AUC | FCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated | 86.69% | 89.55% | 83.26% | 86.29% | 94.47% | **9.84%** |
| Same checkpoint at 0.5 | 87.10% | 88.74% | 85.17% | 86.92% | 94.47% | 10.95% |

Held-out FCR target passed, but recall missed 85% by 1.74 points. This test
result cannot be used for another threshold adjustment. Full record:
[safety summary](generated/safety_v1_summary.md).

## Historical protocol-v2 analysis

## Summary of historical protocol version two results

In the historical fixed-0.5 matrix, no experiment performed best on every
objective. Three candidates had different trade-offs:

- **E1, no augmentation:** safest historical fixed-0.5 model, with **13.96% false-complete
  rate (FCR)**.
- **E4, last-frame pooling:** strongest hard-Hinglish proxy model, with
  **88.05% F1** on that slice and best overall AUC (**93.86%**).
- **E8, partial fine-tuning:** strongest efficiency candidate. It stays within
  **0.17 percentage points (pp)** of fully tuned E2 overall F1 while updating
  4.45M rather than 8.00M parameters and improving E2's hard-case trade-off.

E2 full augmentation reaches best overall F1 (**88.14%**) but worsens FCR.
M1 audio plus text adds only 0.25 pp F1 over its audio-only control, worsens
FCR, and makes live inference 27.6 times slower. Neither becomes default.

The clearest result is that encoder adaptation made the largest quality change,
while data policy moved the safety operating point. Synthetic augmentation can
improve recall-oriented F1 without making interruptions less likely.

The [executed results notebook](../notebooks/03_ablations_and_results.ipynb)
recomputes every table, validates artifacts, plots training and paired effects,
and embeds five qualitative audio cases.

## Evidence and controls

Nine audio ablations and one multimodal experiment were executed under
protocol version 2. Audio runs share seed 42, three epochs, prepared manifests,
optimizer, checkpoint selection by validation F1, and test threshold 0.5.

| Control | Fixed value |
|---|---|
| Train / validation / test | 6,613 / 904 / 4,890 rows |
| Audio input | Mono 16 kHz; final 8 seconds; left-padded |
| Encoder | `openai/whisper-tiny` |
| Optimizer | AdamW, `5e-5`, cosine schedule, 20% warmup |
| Effective batch | 96 |
| Selection | Highest validation F1 |
| Hardware | RTX 4060 Laptop GPU; batch-one CPU/GPU timing |

E1 disables online augmentation but retains balanced and hard-example
sampling. E2 is full-augmentation control for pooling, pause, and encoder
ablations. M1 compares with E1 because cached text would not remain aligned
with dynamically augmented audio.

The held-out test set has been evaluated. Its results describe the current
evidence, but they must not drive more tuning on the same split.

## Overall held-out results

Percentages are shown for classification metrics. Lower FCR is better.
Classifier-path latency excludes feature extraction; M1 live latency is
reported separately below.

| Experiment | Accuracy | Precision | Recall | F1 | AUC | FCR ↓ | FP32 MiB | GPU ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E1 no augmentation | 87.61 | 86.62 | 89.15 | 87.87 | 93.64 | **13.96** | 30.5 | 6.7 |
| E2 full augmentation | **87.75** | 85.95 | 90.45 | **88.14** | 93.79 | 14.99 | 30.5 | 6.3 |
| E3 mean pooling | 86.05 | 83.76 | 89.68 | 86.62 | 92.98 | 17.62 | 30.1 | 5.2 |
| E4 last-frame pooling | 87.32 | 85.58 | 89.96 | 87.72 | **93.86** | 15.36 | 30.1 | 5.3 |
| E5 short pauses | 87.65 | 86.09 | 90.00 | 88.00 | 93.58 | 14.74 | 30.5 | 5.6 |
| E6 long pauses | 87.26 | 84.86 | 90.90 | 87.78 | 93.35 | 16.43 | 30.5 | 5.7 |
| E7 frozen encoder | 71.53 | 69.10 | 78.59 | 73.54 | 77.51 | 35.61 | 30.5 | 6.3 |
| E8 freeze first two layers | 87.46 | 85.08 | **91.06** | 87.97 | 93.32 | 16.18 | 30.5 | 7.0 |
| E11 no silence | 87.12 | 86.14 | 88.66 | 87.38 | 93.53 | 14.45 | 30.5 | 4.9 |
| M1 audio plus text | 87.71 | 85.80 | 90.57 | 88.12 | 93.94 | 15.19 | 43.2 | 6.0* |

`*` M1 classifier timing assumes cached text. Warm end-to-end live latency is
240.15 ms versus 8.71 ms for E1 because M1 must run autoregressive ASR.

Differences below roughly one point remain directional. One seed cannot prove
repeatability or significance.

## Hypothesis outcomes

| Comparison | Pre-registered expectation | Observed result | Verdict |
|---|---|---|---|
| E2 vs E1 | Augmentation lowers hard-case FCR without recall collapse | F1 +0.28 pp, recall +1.30 pp, but overall FCR +1.03 pp and hard-proxy FCR +4.67 pp | Not supported |
| E3 vs E2 | Mean pooling loses local endpoint evidence | F1 -1.52 pp and FCR +2.63 pp | Supported |
| E4 vs E2 | Last frame fails around silence | Only -0.42 pp overall F1; hard-proxy F1 +3.24 pp and FCR -1.87 pp | Not supported |
| E5/E6 vs E2 | Broad pauses generalize best | Broad slightly leads F1; long-only worsens FCR; exact range unresolved | Directional only |
| E11 vs E2 | Silence insertion lowers pause FCR by at least 2 pp | Hard-proxy F1 +5.07 pp, but internal-pause FCR improves only 0.19 pp | F1 benefit; FCR criterion failed |
| E7 vs E2 | Fully frozen encoder underperforms | F1 -14.60 pp and FCR +20.62 pp | Strongly supported |
| E8 vs E2 | Partial tuning stays within 1 pp F1 and 2 pp FCR | F1 -0.17 pp and FCR +1.19 pp; hard-proxy balance improves | Supported |
| M1 vs E1 | Text gives ≥2 pp Hinglish benefit with viable latency | Overall F1 +0.25 pp, worse FCR, 27.6× live latency | Not supported |

## 1. Data augmentation: more robust, not automatically safer

### Problem

Natural hesitation, fillers, channel variation, and speech rate are
underrepresented. Targeted transforms should broaden coverage, but they can
also alter label-calibration relationship.

### Result

E2 improves overall F1 by 0.28 pp, internal-pause F1 by 0.77 pp, and
hard-Hinglish proxy F1 by 1.82 pp over E1. Same model also worsens overall FCR
by 1.03 pp and hard-proxy FCR by 4.67 pp.

### Why

Transforms appear to teach greater tolerance for varied completion evidence,
raising completion recall. They do not create cleaner separation between
natural continuation and completion. Class-conditional pause weighting may
also shift scores toward `complete`. This is an inference from paired metric
movement, not a proven causal mechanism.

### Decision

Keep E2 as high-recall comparator, not safety default. Tune transform mix and
threshold on validation with explicit FCR and recall constraints. Next data
investment should be real ambiguous Hinglish, not more synthetic volume alone.

## 2. Pooling: mean loses; last frame surprises

Mean pooling averages speech, padding, silence, and short endpoint cues. E3
loses 1.52 pp F1 and worsens FCR by 2.63 pp versus attention. This is clearest
architecture result after frozen-encoder failure.

Last-frame pooling contradicts original prediction. E4 is close to attention
overall, achieves best AUC, and leads Hindi filler-plus-pause and hard-proxy
F1. Endpoint evidence may concentrate near final encoded frames after
right-aligned cropping. That explanation needs attention-map or temporal
occlusion evidence before becoming a claim.

Decision: drop mean pooling. Repeat E2 and E4 across seeds. Test gated
attention/last fusion only if E4's hard-case advantage repeats; do not add
complexity from one seed.

## 3. Silence policy: useful signal, weak duration conclusion

E2 broad 100-800 ms and E5 short 50-250 ms are effectively tied. E6 long-only
600-1500 ms increases recall but worsens FCR. Removing silence in E11 costs
5.07 pp hard-proxy F1, yet changes internal-pause FCR by only 0.19 pp.

Silence insertion therefore helps difficult-case sensitivity, but current
evidence does not prove optimal length or strong safety improvement. Synthetic
zero-energy gaps are imperfect models of breathing, room noise, or hesitation.

Decision: retain broad range provisionally, reduce incomplete-class pause
weight in next validation-only study, and evaluate real pause-duration bins.

## 4. Encoder tuning: largest effect in study

| Variant | Trainable parameters | Overall F1 | FCR | Hard-proxy F1 | Hard-proxy FCR |
|---|---:|---:|---:|---:|---:|
| E2 full tune | 8,000,386 | 88.14 | 14.99 | 84.81 | 14.95 |
| E8 freeze first two | 4,452,226 | 87.97 | 16.18 | **87.18** | **12.15** |
| E7 freeze all | 214,402 | 73.54 | 35.61 | 64.43 | 24.30 |

Fully frozen Whisper features are inadequate under fixed learning rate and
three-epoch budget. Turn completion depends on prosody/timing that ASR
pretraining may not expose linearly. E7 proves need for adaptation under this
protocol, not impossibility of every frozen-feature system; larger head LR or
longer training could change its ceiling.

E8 shows lower layers need not all update. It nearly matches E2 overall while
improving hard-proxy F1 and FCR.

**Historical decision:** advance E8 as efficiency finalist. Later safety study
repeated E8 alongside E1/E4; E4 won median constrained validation F1.

## 5. Text semantics: useful signal, wrong deployment trade-off

M1 improves Hindi and hard-proxy F1, suggesting transcript tokens contain some
semantic incompleteness signal. It also worsens overall, Hindi, and hard-proxy
FCR. Live latency rises from 8.71 ms to 240.15 ms; p95 reaches 389.58 ms.

Turn detector sits on response critical path. Autoregressive ASR cost and
transcription noise outweigh small F1 improvement.

Decision: keep M1 for offline diagnosis only. Any future semantic feature
must be asynchronous, confidence-gated, or distilled into non-autoregressive
signal.

## Filler and pause slices

Values are `F1 / FCR`, in percent. Slice support is fixed across models: Hindi
254, mid-filler 1,830, internal pause 3,124, Hindi filler plus pause 159, and
hard-Hinglish proxy 182.

| Experiment | Hindi | Mid-filler | Internal pause | Hindi filler + pause | Hard proxy |
|---|---:|---:|---:|---:|---:|
| E1 | 89.21 / **11.50** | 84.73 / 16.79 | 87.78 / **12.39** | 84.72 / **9.52** | 82.99 / **10.28** |
| E2 | 90.34 / 15.93 | 84.19 / 17.76 | **88.55** / 12.83 | 87.01 / 14.29 | 84.81 / 14.95 |
| E3 | 88.97 / 17.70 | 83.31 / 17.76 | 86.91 / 15.44 | 85.16 / 16.67 | 83.54 / 15.89 |
| E4 | **91.72** / 14.16 | 84.94 / 17.33 | 87.62 / 13.66 | **89.74** / 13.10 | **88.05** / 13.08 |
| E5 | 89.66 / 16.81 | 84.26 / 17.01 | 87.84 / 12.83 | 86.27 / 14.29 | 84.08 / 14.95 |
| E6 | 90.78 / 16.81 | **85.13** / 18.19 | 88.28 / 14.74 | 87.18 / 15.48 | 85.00 / 15.89 |
| E7 | 73.13 / 25.66 | 65.29 / 36.71 | 73.64 / 32.78 | 66.67 / 25.00 | 64.43 / 24.30 |
| E8 | 91.67 / 13.27 | 84.67 / 18.41 | 88.18 / 14.49 | 88.89 / 11.90 | 87.18 / 12.15 |
| E11 | 87.72 / 16.81 | 83.93 / **16.15** | 87.82 / 13.02 | 81.88 / 15.48 | 79.74 / 15.89 |

Fillers are not labels. Same filler-plus-pause pattern appears in both endpoint
classes. Models must combine it with timing and prosody; forcing
`filler → incomplete` would create another shortcut.

## Qualitative evidence

Four finalists rescored 35 deterministic Hindi filler/pause examples. This is
an inspection set, not another metric estimate. Full probabilities and paths
are stored in
[`qualitative_examples.json`](../experiments/protocol_v2_seed42/qualitative_examples.json).

| Case | Truth and metadata | E1 | E2 | E4 | E8 | Learning |
|---|---|---:|---:|---:|---:|---|
| `00012542` | Incomplete; internal pause; mid+end filler | 0.011 | 0.013 | 0.011 | 0.015 | All models correctly resist completion despite filler and pause |
| `003d8a0b` | Complete; internal pause; mid-filler | 0.897 | 0.844 | 0.910 | 0.851 | Same surface cues can still form complete turn |
| `08888b79` | Incomplete; mid-filler; no detected internal pause | 0.868 | 0.770 | 0.845 | 0.703 | Consensus false complete; product-critical interruption failure |
| `174e2874` | Complete; internal pause; mid-filler | 0.014 | 0.024 | 0.019 | 0.023 | Consensus false incomplete; may reflect prosody failure or label ambiguity |
| `05f3ceab` | Incomplete; internal pause; mid+end filler | 0.196 | 0.763 | 0.433 | 0.786 | E1/E4 correct, E2/E8 interrupt; augmentation/tuning shifts calibration |

Probabilities are `P(complete)` at threshold 0.5. Notebook embeds waveforms and
audio controls for listening. Dataset supplies no transcript or human ambiguity
rating, so linguistic interpretations remain deliberately limited.

## Per-experiment decision record

| ID | What worked | What failed | Decision |
|---|---|---|---|
| E1 | Lowest overall and hard-proxy FCR | Lower hard-case recall/F1 than E4/E8 | Safety finalist |
| E2 | Best overall F1 and recall | Safety hypothesis failed | High-recall control |
| E3 | Slightly faster/smaller pooling | Worst trainable pooling result | Drop |
| E4 | Best AUC and hard-proxy F1 | One seed; FCR above E1 | Hard-case finalist |
| E5 | Competitive F1 and FCR | No clear gain over broad range | Do not prefer |
| E6 | Highest recall among pause variants | Worse FCR | Drop long-only policy |
| E7 | Reveals adaptation requirement | Severe fixed-budget failure | Drop |
| E8 | Near-full quality, fewer trainable parameters | Overall FCR above E1 | Efficiency finalist |
| E11 | Fast and reasonable global FCR | Weakest trainable hard-proxy F1 | Keep silence insertion |
| M1 | Hindi/hard-case F1 improves | Worse FCR and 27.6× latency | Offline diagnostic |

## How findings change next work

1. Completed: E1, E4, and E8 repeated over seeds 42/43/44; E4 won median
   constrained validation F1.
2. Completed: validation-only calibration targeted FCR ≤10% and recall ≥85%;
   E4 seed 44 threshold was frozen before held-out evaluation.
3. Build H1: speaker-disjoint human Hinglish with fillers, code-switch labels,
   rising intonation, internal pause boundaries, short complete replies, and
   long incomplete clauses.
4. Compare finalists on H1 as Pareto set over FCR, recall, hard-case F1, and
   latency rather than one leaderboard score.
5. Only after repeatability, test attention/last-frame fusion and lower
   incomplete-class silence weighting.

Data collection and annotation now have higher expected value than another
small classification-head variant.

## Limits

1. Broad matrix has one seed; finalist study has three seeds but no confidence intervals.
2. Hindi, filler, and energy-based pause slices are Hinglish proxies.
3. No speaker identities; split cannot prove speaker disjointness.
4. Most data is synthetic, so TTS cadence may remain shortcut.
5. Historical protocol-v2 comparisons use 0.5; safety finalists use validation-selected thresholds.
6. Equal three-epoch budget does not guarantee every method reached own optimum.
7. Final test has been opened; further tuning requires validation or new H1.

## Reproduction and evidence

```powershell
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 42 --run-tag safety_v1_seed42
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 43 --run-tag safety_v1_seed43
uv run python scripts/run_experiments.py --only E1_no_augmentation E4_last_pool E8_partial_finetune --epochs 6 --seed 44 --run-tag safety_v1_seed44
uv run python scripts/select_safety_finalist.py experiments/safety_v1_seed42 experiments/safety_v1_seed43 experiments/safety_v1_seed44
```

Historical protocol-v2 artifacts below preserve their resolved configs and
fixed-0.5 outputs. Current `configs/baseline.yaml` enables calibration, so old
command text would not recreate that historical operating point exactly.

- [Experiment manifest](../experiments/protocol_v2_seed42/experiment_manifest.json)
- [Structured results](../experiments/protocol_v2_seed42/all_results.json)
- [Comparison CSV](../experiments/protocol_v2_seed42/comparison.csv)
- [Qualitative audit](../experiments/protocol_v2_seed42/qualitative_examples.json)
- [Multimodal comparison](../experiments/multimodal_comparison.json)
- [Multimodal latency](../experiments/multimodal_latency.json)
