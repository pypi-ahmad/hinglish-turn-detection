<!-- markdownlint-disable MD013 MD060 -->

# Ablation Results and Engineering Insights

## Executive decision

No single model wins every objective.

- **Best overall F1:** E2, full augmentation, at **88.14%**.
- **Lowest false-complete rate (FCR):** E1, unaugmented audio, at **13.96%**.
- **Best ROC-AUC and hard-Hinglish proxy F1:** E4, last-frame pooling, at **93.86% AUC** and **88.05% hard-proxy F1**.
- **Best capacity trade-off:** E8, two frozen encoder layers, stays within **0.17 percentage points (pp)** of E2 overall F1 while using **4.45M instead of 8.00M trainable parameters**. It also improves hard-proxy F1 and FCR relative to E2.
- **Audio + text is not justified for live turn detection:** M1 gains only **0.25 pp** F1 over E1, worsens FCR by **1.24 pp**, and is **27.6x slower** end to end.

Recommended next step is not to declare a universal winner. Advance E1 as the safety-oriented control, E4 as the strongest hard-Hinglish model, and E8 as the efficient finalist. Repeat these with two additional seeds, choose an operating threshold on validation data, then evaluate once on a real speaker-disjoint Hinglish challenge set.

## What was run

This report uses protocol version 2. Exact configurations, data fingerprints, checkpoints, validation histories, and final-test outputs are stored in [`experiments/protocol_v2_seed42`](../experiments/protocol_v2_seed42/).

| Control | Value |
| --- | --- |
| Dataset | `pipecat-ai/smart-turn-data-v3.2-train` local prepared subset |
| Split sizes | 6,613 train / 904 validation / 4,890 untouched test |
| Seed | 42 |
| Audio | Mono 16 kHz; last 8 seconds; left-padded |
| Encoder | `openai/whisper-tiny` encoder |
| Classifier | Pooling + 384→256→64→1 MLP |
| Optimization | AdamW, `5e-5`, cosine schedule, 20% warmup, 3 epochs |
| Selection | Highest validation F1; threshold fixed at 0.5 |
| Hardware | NVIDIA RTX 4060 Laptop GPU; CPU and GPU latency measured at batch 1 |

E1 means original audio with no online augmentation. Class balancing and hard-example oversampling remain enabled, so it is an augmentation control rather than an untouched-frequency control. E2–E8 and E11 change one declared factor at a time. All nine selected checkpoints were evaluated once on the test split after validation work finished.

Pause slices are reproducible acoustic proxies: a pause is at least 100 ms below -40 dBFS, measured with 20 ms RMS frames and 10 ms hops. They are not human voice-activity annotations. `hard_hinglish_proxy` combines Hindi hard examples with Hindi filler-plus-internal-pause examples; it is not a verified code-switching benchmark.

## Overall held-out results

All values except size and latency are percentages. Lower FCR is safer because it means fewer incomplete turns were incorrectly declared complete. GPU latency is classifier-path latency; M1's separate live latency includes ASR.

| Experiment | Accuracy | Precision | Recall | F1 | AUC | FCR ↓ | FP32 MiB | CPU ms | GPU ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 no augmentation | 87.61 | 86.62 | 89.15 | 87.87 | 93.64 | **13.96** | 30.5 | 32.5 | 6.7 |
| E2 full augmentation | 87.75 | 85.95 | 90.45 | **88.14** | 93.79 | 14.99 | 30.5 | 29.1 | 6.3 |
| E3 mean pooling | 86.05 | 83.76 | 89.68 | 86.62 | 92.98 | 17.62 | 30.1 | 28.1 | 5.2 |
| E4 last-frame pooling | 87.32 | 85.58 | 89.96 | 87.72 | **93.86** | 15.36 | 30.1 | 28.4 | 5.3 |
| E5 short pauses | 87.65 | 86.09 | 90.00 | 88.00 | 93.58 | 14.74 | 30.5 | 28.8 | 5.6 |
| E6 long pauses | 87.26 | 84.86 | 90.90 | 87.78 | 93.35 | 16.43 | 30.5 | 29.6 | 5.7 |
| E7 frozen encoder | 71.53 | 69.10 | 78.59 | 73.54 | 77.51 | 35.61 | 30.5 | 29.9 | 6.3 |
| E8 freeze first two layers | 87.46 | 85.08 | **91.06** | 87.97 | 93.32 | 16.18 | 30.5 | 32.1 | 7.0 |
| E11 no silence insertion | 87.12 | 86.14 | 88.66 | 87.38 | 93.53 | 14.45 | 30.5 | 25.1 | 4.9 |
| M1 audio + cached text | 87.71 | 85.80 | 90.57 | 88.12 | 93.94 | 15.19 | 43.2 | 29.5 | 6.0* |

`*` M1's 6.0 ms is only fused-classifier latency with cached text. Warm end-to-end batch-1 latency is 240.15 ms versus 8.71 ms for E1 because live M1 must also transcribe audio.

Differences below roughly one point should be treated as directional, not conclusive: this is one seed with no confidence interval. E11 illustrates selection risk. It led validation F1 at 89.30% but fell to 87.38% on test, so validation rank did not transfer.

## Hard-case error analysis

Slice notation is `F1 / FCR`; all values are percentages. Slice sizes are fixed across models: Hindi `n=254`, mid-filler `n=1,830`, internal pause `n=3,124`, Hindi filler + pause `n=159`, and hard-Hinglish proxy `n=182`.

| Experiment | Hindi | Mid-filler | Internal pause | Hindi filler + pause | Hard-Hinglish proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 | 89.21 / **11.50** | 84.73 / 16.79 | 87.78 / **12.39** | 84.72 / **9.52** | 82.99 / **10.28** |
| E2 | 90.34 / 15.93 | 84.19 / 17.76 | **88.55** / 12.83 | 87.01 / 14.29 | 84.81 / 14.95 |
| E3 | 88.97 / 17.70 | 83.31 / 17.76 | 86.91 / 15.44 | 85.16 / 16.67 | 83.54 / 15.89 |
| E4 | **91.72** / 14.16 | **84.94** / 17.33 | 87.62 / 13.66 | **89.74** / 13.10 | **88.05** / 13.08 |
| E5 | 89.66 / 16.81 | 84.26 / 17.01 | 87.84 / 12.83 | 86.27 / 14.29 | 84.08 / 14.95 |
| E6 | 90.78 / 16.81 | 85.13 / 18.19 | 88.28 / 14.74 | 87.18 / 15.48 | 85.00 / 15.89 |
| E7 | 73.13 / 25.66 | 65.29 / 36.71 | 73.64 / 32.78 | 66.67 / 25.00 | 64.43 / 24.30 |
| E8 | 91.67 / 13.27 | 84.67 / 18.41 | 88.18 / 14.49 | 88.89 / 11.90 | 87.18 / 12.15 |
| E11 | 87.72 / 16.81 | 83.93 / **16.15** | 87.82 / 13.02 | 81.88 / 15.48 | 79.74 / 15.89 |

### What errors remain

- **Fillers are not automatically evidence of incompleteness.** E1 has the safest Hindi filler-plus-pause FCR, while augmented models often gain recall and F1 by accepting more false completes. Label-preserving filler injection prevented a direct `filler → incomplete` shortcut, but did not solve calibration.
- **Pause augmentation mostly moves the operating point.** Longer pauses increase recall, but also increase false completes. This suggests models learn broader completion tolerance rather than a clean pause-duration rule.
- **Last-frame evidence matters.** E4 is strongest on Hindi, Hindi filler-plus-pause, and hard-Hinglish F1. The original hypothesis predicted trailing-silence failure, but held-out data contradicts that simple story.
- **Frozen ASR features miss endpoint prosody.** E7 fails across every slice. A classifier on fully frozen Whisper representations is insufficient under the fixed learning-rate and epoch budget.
- **The hardest errors remain highly confident.** Stored examples include complete clips assigned probabilities near 0.01 and incomplete clips near 0.98. Threshold adjustment alone cannot repair those representation or label failures.

End-filler rows are not used for binary F1 conclusions because that slice contains only incomplete labels. FCR remains descriptive there. Legacy filler flags also lack complete annotation provenance for some negative flags, so positive slices are usable but filler prevalence is not trustworthy.

## 1. Original data versus targeted augmentation

**Problem.** Natural fillers and pauses can make an incomplete utterance sound locally complete. Generic training may miss these cases, but aggressive augmentation can also teach the model to tolerate genuine endpoint evidence.

**Hypothesis.** Label-preserving fillers, silence, speed, pitch, noise, and volume transforms will reduce false completes on pause-heavy Hinglish cases without harming overall F1.

**Experiment.** Compare E1 and E2. Architecture, split, optimizer, hard-example sampler, seed, and training budget are identical. E2 alone enables all targeted transforms.

**Result.** E2 gains 0.27 pp overall F1, 1.30 pp recall, 0.77 pp internal-pause F1, and 1.82 pp hard-proxy F1. It loses 0.67 pp precision and worsens overall FCR by 1.03 pp and hard-proxy FCR by 4.67 pp.

**Learning.** Augmentation improves sensitivity to completion but not safety. The pre-registered E2 criterion required lower hard-slice FCR, so the hypothesis **fails** even though F1 improves. Data work matters, but current mixture shifts calibration toward `complete` rather than learning a uniformly better boundary.

**Next decision.** Keep E2 as a high-recall candidate, not production default. Tune transform probabilities and class-conditional pause weighting on validation FCR. Calibrate threshold on validation only, with a recall floor, before another locked evaluation.

## 2. Pooling: mean versus attention versus last frame

**Problem.** Eight-second windows contain speech, padding, fillers, and silence. Pooling determines whether endpoint cues remain visible to the classifier.

**Hypothesis.** Attention should beat mean pooling by selecting relevant frames, while last-frame pooling should fail when trailing silence separates speech from window end.

**Experiment.** E2 uses attention, E3 mean, and E4 last frame. All use identical augmented data and full fine-tuning.

**Result.** Attention beats mean by 1.52 pp overall F1, 1.64 pp internal-pause F1, and 1.27 pp hard-proxy F1, while reducing overall FCR by 2.63 pp. This passes the pre-registered attention-versus-mean criterion. Last-frame pooling is only 0.42 pp below attention overall, has the best AUC, and beats attention by 3.24 pp hard-proxy F1 with 1.87 pp lower hard-proxy FCR. Attention does reduce internal-pause FCR by 0.83 pp, but last-frame recall and hard-slice behavior refute the predicted broad failure.

**Learning.** Mean pooling dilutes short endpoint evidence. Attention is the safest general pooling choice. Last-frame pooling is unexpectedly valuable for Hinglish hard cases, likely because completion evidence is concentrated near the end of each cropped clip. That explanation is an inference, not proven attribution.

**Next decision.** Repeat E2 and E4 across seeds. Inspect attention weights and stratify by trailing-pause duration. A small gated fusion of attention and last-frame vectors is worth testing only after repeatability is established.

## 3. Silence insertion strategy

**Problem.** Pauses range from brief breaths to long hesitations. A single synthetic range may overfit pause duration or make silence a label shortcut.

**Hypothesis.** Broad 100–800 ms insertion should generalize better than short-only 50–250 ms, long-only 600–1500 ms, or no inserted silence.

**Experiment.** E2 uses 100–800 ms; E5 short-only; E6 long-only; E11 disables silence while retaining filler, speed, pitch, noise, volume, and hard mining.

**Result.** E2, E5, and E6 are close overall: 88.14%, 88.00%, and 87.78% F1. E2 and E5 tie at 12.83% internal-pause FCR; E6 worsens it to 14.74%. E2 beats E11 by 0.76 pp overall F1, 0.73 pp internal-pause F1, and 5.07 pp hard-proxy F1. However, E2 improves internal-pause FCR over E11 by only 0.19 pp, far below the pre-registered 2 pp target.

**Learning.** Some silence augmentation is useful for hard-Hinglish recall, but exact duration is not resolved by one seed. Long-only insertion is the weakest safety trade-off. The broad-range hypothesis is directionally supported for F1, not for the planned FCR criterion.

**Next decision.** Retain broad 100–800 ms as current default, reduce incomplete-class pause oversampling, and add evaluation bins for 100–250, 250–500, 500–800, and >800 ms. Do not claim duration robustness until real pause boundaries are annotated.

## 4. Frozen versus partial versus full fine-tuning

**Problem.** Full fine-tuning may be unnecessary, but frozen ASR features may omit endpoint prosody.

**Hypothesis.** A fully frozen encoder will underperform; freezing the first two layers will retain most quality with fewer trainable parameters.

**Experiment.** E2 updates all four encoder layers, E8 freezes the first two, and E7 freezes all four. Data, head, optimizer, seed, and budget stay fixed.

| Model | Trainable parameters | Test F1 | Test FCR | Hard-proxy F1 | Hard-proxy FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E2 full tune | 8,000,386 | 88.14 | 14.99 | 84.81 | 14.95 |
| E8 freeze first two | 4,452,226 | 87.97 | 16.18 | **87.18** | **12.15** |
| E7 freeze all | 214,402 | 73.54 | 35.61 | 64.43 | 24.30 |

**Result.** Full tuning beats fully frozen by 14.60 pp overall F1 and 20.38 pp hard-proxy F1. E8 stays within 0.17 pp overall F1 and 1.19 pp FCR of E2, satisfying the pre-registered efficiency criterion. E8 also improves hard-proxy F1 by 2.37 pp and FCR by 2.80 pp.

**Learning.** Task-specific encoder adaptation is essential, but lower layers need not all move. E8 is the strongest efficiency result. E7 is a fixed-budget ablation, not a fully tuned frozen-feature baseline: a larger head learning rate or longer schedule might improve it.

**Next decision.** Advance E8. Measure training memory and throughput explicitly, then repeat E2/E8 across seeds. If E8 remains stable, prefer it over full tuning for reproducibility and lower optimizer-state memory.

## 5. Audio only versus audio + text

**Problem.** Acoustics can miss incomplete syntax; transcripts may expose discourse markers and semantic incompleteness. Live ASR also adds latency and transcription errors.

**Hypothesis.** Cached Whisper transcripts will improve Hindi and hard-Hinglish performance enough to justify added deployment cost.

**Experiment.** M1 uses the same unaugmented split and attention audio encoder as E1, then concatenates a learned 64-dimensional token-average embedding. Two of 12,407 cached transcripts were empty. Classifier metrics use cached text; live latency separately includes feature extraction, ASR generation, tokenization, and classification.

**Result.** M1 improves overall F1 by 0.25 pp, Hindi F1 by 2.32 pp, and hard-proxy F1 by 3.43 pp. It worsens overall FCR by 1.24 pp, Hindi FCR by 5.31 pp, and hard-proxy FCR by 5.61 pp. End-to-end latency rises from 8.71 ms to 240.15 ms; p95 reaches 389.58 ms.

**Learning.** Text raises recall-oriented F1 but increases the exact premature-completion risk that turn detection must control. It also violates the latency objective. M1 fails the adoption criterion.

**Next decision.** Keep M1 as an offline diagnostic, not live default. If semantics are revisited, use an asynchronous transcript signal or tiny intent/incompleteness feature with a confidence gate; never block endpoint detection on autoregressive ASR.

## Per-experiment verdicts

| ID | What worked | What did not | Decision |
| --- | --- | --- | --- |
| E1 | Best overall and hard-proxy FCR; strong simple baseline | Lower hard-proxy recall/F1 than E4/E8 | Safety finalist |
| E2 | Best overall F1; better recall and pause F1 | FCR worsened, especially hard-Hinglish | High-recall comparator |
| E3 | Lowest parameter count and fast pooling | Worst learned pooling among trainable models | Drop |
| E4 | Best AUC, Hindi F1, and hard-proxy F1 | Overall FCR above E1; one-seed result | Hard-case finalist |
| E5 | Competitive overall F1 and FCR | No clear gain over broad pauses | Do not prefer yet |
| E6 | High recall | Worse FCR and no decisive F1 gain | Drop long-only policy |
| E7 | Establishes adaptation necessity | Severe failure under fixed budget | Drop; caveat LR/budget |
| E8 | Half trainable parameters; strong hard-case trade-off | Overall FCR above E1/E2 | Efficiency finalist |
| E11 | Good validation F1 and fast measured latency | Validation rank failed on test; weakest trainable hard-proxy F1 | Keep silence augmentation |
| M1 | Hindi and hard-proxy F1 gains | Worse FCR; 27.6x live latency | Offline diagnostic only |

## Limits on conclusions

1. **One seed:** no standard deviations or paired confidence intervals. Small differences are hypotheses for confirmation.
2. **Proxy Hinglish labels:** Hindi tags do not prove Hindi-English code-switching, accent, filler type, or rising intonation.
3. **Heuristic pauses:** low energy is not equivalent to a linguistic hesitation; noise suppression and microphone gain can alter it.
4. **No speaker-disjoint challenge set:** split logic is stratified where speaker identity is absent. Leakage by voice or synthesis system remains possible.
5. **Test used once, but now used:** these results must not drive repeated tuning against the same test split. Further selection belongs on validation or a new challenge set.
6. **Threshold fixed at 0.5:** FCR/recall trade-offs may change after validation-only calibration.
7. **Three-epoch budget:** sufficient for controlled comparison, not proof each architecture reached its own optimum.

## Final learning and next decisions

1. Repeat E1, E4, and E8 with seeds 43 and 44. Promote only effects with consistent direction in at least two of three seeds.
2. Calibrate thresholds on validation data for explicit operating points: FCR ≤10%, recall ≥85%, and maximum F1. Freeze thresholds before new evaluation.
3. Build H1: speaker-disjoint, human-recorded Hinglish with fillers, internal pauses, rising intonation, short complete replies, and long incomplete clauses.
4. Annotate pause start/end, filler identity, code-switch presence, and ambiguity. Report bootstrap intervals and per-speaker failures.
5. Compare E1/E4/E8 on H1. Keep a Pareto view—FCR, recall, hard-case F1, latency—not a single leaderboard number.
6. Only after repeatability, test attention + last-frame fusion and lower incomplete-class silence weighting.

## Reproduction and evidence

Core commands used:

```powershell
uv run python scripts/run_experiments.py --only E1_no_augmentation E2_augmented E3_mean_pool E4_last_pool E5_short_pauses E6_long_pauses E7_frozen_encoder E8_partial_finetune E11_no_silence --epochs 3 --run-tag protocol_v2_seed42

uv run python scripts/run_experiments.py --only E1_no_augmentation E2_augmented E3_mean_pool E4_last_pool E5_short_pauses E6_long_pauses E7_frozen_encoder E8_partial_finetune E11_no_silence --epochs 3 --run-tag protocol_v2_seed42 --reuse-existing --final-test

uv run python scripts/run_multimodal_experiment.py --reuse-existing --baseline-result experiments/protocol_v2_seed42/E1_no_augmentation/result.json
```

Evidence files:

- [Experiment manifest](../experiments/protocol_v2_seed42/experiment_manifest.json)
- [Structured results](../experiments/protocol_v2_seed42/all_results.json)
- [Comparison CSV](../experiments/protocol_v2_seed42/comparison.csv)
- [Multimodal comparison](../experiments/multimodal_comparison.json)
- [Multimodal latency](../experiments/multimodal_latency.json)
