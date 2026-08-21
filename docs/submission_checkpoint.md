<!-- markdownlint-disable MD013 -->

# Bundled submission checkpoint

## Purpose

`checkpoints/safety_finalist/best.pt` is default Gradio checkpoint. It is E4 last-frame pooling, seed 44, chosen by a predeclared three-architecture/three-seed validation protocol. Historical six-epoch attention checkpoint remains at `checkpoints/baseline_attention_augmented/best.pt` for comparison.

## Selection protocol

- Train: 6,613 rows from bounded Pipecat v3.2 train subset
- Validation: 904 stratified rows
- Test: 4,890 rows from separate Pipecat v3.2 test repository
- Candidates: E1, E4, E8; six epochs; seeds 42, 43, 44
- Per-run threshold: maximize validation F1 subject to FCR ≤10% and recall ≥85%
- Architecture: highest median constrained validation F1
- Deployment seed: median-F1 run within winning architecture
- Winner: E4 seed 44, threshold `0.5777403116226196`
- Size: 7,901,569 parameters; 30.14 MiB FP32 parameter footprint

All nine runs met validation constraints. Architecture medians:

| Architecture | Median validation F1 | Median validation FCR |
| --- | ---: | ---: |
| E1 | 87.67% | 9.87% |
| E4 | **89.82%** | **9.21%** |
| E8 | 88.34% | 9.87% |

## Final results

Held-out test was evaluated once after architecture, seed, epoch, and threshold were frozen.

| Split / operating point | Accuracy | Precision | Recall | F1 | AUC | FCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation, calibrated | 90.04% | 91.06% | 88.62% | 89.82% | 95.53% | 8.55% |
| Held-out test, calibrated | 86.69% | 89.55% | 83.26% | 86.29% | 94.47% | 9.84% |
| Held-out test, fixed 0.5 control | 87.10% | 88.74% | 85.17% | 86.92% | 94.47% | 10.95% |

Held-out slices at calibrated threshold:

| Slice | n | Accuracy | Recall | FCR |
| --- | ---: | ---: | ---: | ---: |
| Hindi | 254 | 87.01% | 78.01% | 1.77% |
| Mid-filler | 1,830 | 84.86% | 79.58% | 10.01% |
| End-filler | 1,200 | 94.75% | n/a | 5.25% |
| Hindi filler + pause | 159 | 88.05% | 76.00% | 1.19% |
| Human audio | 1,071 | 90.01% | 92.34% | 12.31% |
| Hard examples | 1,521 | 92.24% | 72.73% | 7.62% |

Calibration reduced same-checkpoint held-out FCR by 1.11 percentage points versus threshold 0.5. Cost was 1.91 points of recall. Held-out recall missed desired 85% by 1.74 points, so result is a safety/latency trade-off rather than universal quality improvement. Test result must not be used for another threshold adjustment.

## Runtime behavior

Checkpoint stores `decision_threshold` and calibration evidence. `TurnDetector.predict()` and `predict_batch()` use it automatically; callers may pass explicit threshold override. Evaluation reports both deployed and fixed-0.5 metrics. Legacy checkpoints without threshold fall back to 0.5.

## Scope and limitations

- Local training uses bounded 7,517-row sample, not all 270,946 source rows.
- Speaker/accent IDs unavailable; speaker-disjoint evaluation impossible.
- Controlled TTS filler mixing does not replace human Hinglish evaluation.
- Mid-filler recall remains weak; human-audio FCR exceeds 10%.
- Three seeds improve evidence over original one-seed study but do not provide narrow uncertainty bounds.
- Full run details: [safety summary](generated/safety_v1_summary.md) and [ablation insights](03_ablation_insights.md).
