<!-- markdownlint-disable MD013 -->

# Bundled Submission Checkpoint

## Purpose

`checkpoints/baseline_attention_augmented/best.pt` is the bundled six-epoch checkpoint used by the default Gradio demo. It is separate from the equal-budget, three-epoch protocol-v2 ablations documented in [Ablation Study and Experimental Insights](03_ablation_insights.md).

## Setup

- Train: 6,613 rows from the bounded Pipecat v3.2 train subset
- Validation: 904 stratified rows
- Test: 4,890 rows from the separate Pipecat v3.2 test repository
- Model: Whisper Tiny encoder, eight-second window, attention pooling, binary head
- Size: 8,000,386 parameters; 30.52 MiB FP32 parameter footprint
- Selection: highest validation F1, defined in `configs/baseline.yaml`

## Results

| Split | Accuracy | Precision | Recall | F1 | AUC | False-complete rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 89.16% | 87.55% | 91.07% | 89.28% | 94.45% | 12.72% |
| Held-out test | 87.59% | 88.05% | 87.16% | 87.60% | 94.21% | 11.98% |

Test slices:

| Slice | n | Accuracy | False-complete rate |
| --- | ---: | ---: | ---: |
| Hindi | 254 | 88.58% | 8.85% |
| Mid-filler | 1,830 | 84.10% | 13.99% |
| End-filler | 1,200 | 93.42% | 6.58% |
| Hindi mid-filler | 169 | 83.43% | 14.08% |
| Hindi end-filler | 85 | 98.82% | 1.18% |
| Human audio | 1,071 | 90.66% | 10.82% |
| Hard examples | 1,521 | 89.28% | 10.73% |

## Hardening applied before this checkpoint

The initial implementation used Whisper's 30-second ASR window, costing about 111 ms per CPU model forward. The corrected model uses an eight-second/400-position window while retaining pretrained positional weights. Measured model-forward latency became 23.74 ms on CPU and 4.83 ms on GPU.

Other corrected defects included contradictory filler label flips, augmentations cropped out of long clips, stale augmented metadata, zero-fraction split data loss, unbalanced hard-example sampling, final gradient-accumulation scaling, scheduler step count, very-short-audio masks, malformed inference inputs, missing requested fillers, missing parameter-budget enforcement, and failure to revalidate the reloaded best checkpoint.

## Scope and limitations

- This checkpoint uses a bounded 7,517-row local train sample, not all 270,946 source rows.
- Speaker and accent IDs are unavailable; speaker-disjoint evaluation is impossible.
- Controlled TTS filler mixing is not a substitute for human Hinglish evaluation.
- Background-noise fallback is synthetic room-like noise, not a licensed Indian street/office corpus.
- Mid-filler remains the weakest measured slice.
- Use [protocol-v2 ablations](03_ablation_insights.md) for controlled model comparisons; do not compare this six-epoch result directly with three-epoch rows as if training budgets matched.
