<!-- markdownlint-disable MD013 MD060 -->

# Audio-only vs Audio+Text

This comparison holds the unaugmented train, validation, and test splits fixed. Both runs use seed 42, the same optimizer, attention pooling, and a 3-epoch budget. M1 adds transcripts from frozen Whisper Tiny ASR, a learned 64-dimensional token embedding, and feature concatenation.

| Model | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | GPU classifier ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 87.61% | 86.62% | 89.15% | 87.87% | 93.64% | 13.96% | 8,000,386 | 6.70 |
| M1_audio_plus_text | 87.71% | 85.80% | 90.57% | 88.12% | 93.94% | 15.19% | 11,336,130 | 5.99 |

## Overall difference

M1 versus E1: accuracy +0.10 pp, F1 +0.25 pp, AUC +0.30 pp, and false-complete +1.24 pp.

## Hindi / Hinglish-proxy slices

The dataset has Hindi language tags but no verified code-switch or Hinglish annotation. The Hindi and Hindi+filler slices are proxies rather than a Hinglish benchmark.

| Slice | n | E1 F1 | M1 F1 | F1 delta | E1 false-complete | M1 false-complete | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hindi | 254 | 89.21% | 91.53% | +2.32 pp | 11.50% | 16.81% | +5.31 pp |
| Hindi + mid-filler | 169 | 86.01% | 88.46% | +2.45 pp | 16.90% | 25.35% | +8.45 pp |
| Hindi + end-filler | 85 | n/a | n/a | n/a | 5.88% | 7.06% | +1.18 pp |

`n/a` means slice contains only one ground-truth class; binary F1 is not meaningful there.

## Latency scope

Classifier latency does not include transcript generation. Cached text is suitable for this offline ablation, but live deployment must include autoregressive ASR latency. That removes the speed advantage of the small audio-only model.

Batched ASR cache run: 12,407 rows; 2 empty transcripts; 32.25 ms/clip amortized generation time on cuda (not single-request latency).

| End-to-end batch-1 path | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Audio-only | 8.71 ms | 8.66 ms | 9.92 ms |
| Audio+text | 240.15 ms | 232.83 ms | 389.58 ms |

Across 20 held-out clips, warm live M1 is 27.6x slower. The measurement includes feature extraction, ASR generation, tokenization, and classification, but excludes model loading.
