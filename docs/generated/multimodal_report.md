<!-- markdownlint-disable MD013 MD060 -->

# Audio-only vs Audio+Text

Controlled comparison: same unaugmented train/validation/test splits, seed 42, optimizer, attention pooling, and 3-epoch budget. M1 adds frozen Whisper-tiny ASR transcripts, a learned 64-dimensional token embedding, and feature concatenation.

| Model | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | GPU classifier ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 87.61% | 86.62% | 89.15% | 87.87% | 93.64% | 13.96% | 8,000,386 | 6.70 |
| M1_audio_plus_text | 87.71% | 85.80% | 90.57% | 88.12% | 93.94% | 15.19% | 11,336,130 | 5.99 |

## Overall difference

M1 versus E1: accuracy +0.10 pp, F1 +0.25 pp, AUC +0.30 pp, and false-complete +1.24 pp.

## Hindi / Hinglish-proxy slices

Dataset has Hindi language tags but no verified code-switch/Hinglish annotation. These Hindi and Hindi+filler slices are proxies, not a claimed Hinglish benchmark.

| Slice | n | E1 F1 | M1 F1 | F1 delta | E1 false-complete | M1 false-complete | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hindi | 254 | 89.21% | 91.53% | +2.32 pp | 11.50% | 16.81% | +5.31 pp |
| Hindi + mid-filler | 169 | 86.01% | 88.46% | +2.45 pp | 16.90% | 25.35% | +8.45 pp |
| Hindi + end-filler | 85 | n/a | n/a | n/a | 5.88% | 7.06% | +1.18 pp |

`n/a` means slice contains only one ground-truth class; binary F1 is not meaningful there.

## Latency scope

Classifier latency excludes transcript generation. Cached text is appropriate for this offline ablation; live deployment must add autoregressive ASR latency and therefore loses the tiny audio-only model's streaming-speed advantage.

Batched ASR cache run: 12,407 rows; 2 empty transcripts; 32.25 ms/clip amortized generation time on cuda (not single-request latency).

| End-to-end batch-1 path | Mean | p50 | p95 |
| --- | ---: | ---: | ---: |
| Audio-only | 8.71 ms | 8.66 ms | 9.92 ms |
| Audio+text | 240.15 ms | 232.83 ms | 389.58 ms |

Warm live M1 is 27.6x slower across 20 held-out clips. Model loading is excluded; feature extraction, ASR generation, tokenization, and classification are included.
