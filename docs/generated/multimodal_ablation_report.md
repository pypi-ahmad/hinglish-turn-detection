<!-- markdownlint-disable MD013 MD060 -->

# Turn Detection Ablation Report

All completed runs use matched data partitions and **3 epochs**. Checkpoint selection uses validation F1; comparison rows currently contain: test.

## Comparison

| Experiment | Status | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | FP32 MB | CPU ms | GPU ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | completed | 87.61% | 86.62% | 89.15% | 87.87% | 93.64% | 13.96% | 8,000,386 | 30.52 | 32.49 | 6.70 |
| M1_audio_plus_text | completed | 87.71% | 85.80% | 90.57% | 88.12% | 93.94% | 15.19% | 11,336,130 | 43.24 | 29.46 | 5.99 |

## Pause and hard-case slices

| Experiment | Internal-pause F1 | Internal-pause FCR | Hindi filler+pause F1 | Hard-Hinglish F1 | Hard-Hinglish FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 87.78% | 12.39% | 84.72% | 82.99% | 10.28% |
| M1_audio_plus_text | 88.36% | 13.91% | 87.50% | 86.42% | 15.89% |

## Per-experiment insights

### E1_no_augmentation

Attention pooling on original, unaugmented audio. Evaluation F1 87.87%; accuracy 87.61%; false-complete 13.96%.

### M1_audio_plus_text

Whisper audio embedding plus cached Whisper transcript embedding. Evaluation F1 88.12%; accuracy 87.71%; false-complete 15.19%. Versus E1_no_augmentation: F1 +0.25 points; false-complete +1.24 points.

## Summary

- Highest evaluation F1: **M1_audio_plus_text** (88.12%).
- Lowest false-complete rate among models with at least 75% recall: **E1_no_augmentation** (13.96%).
- Single-seed, short-budget ablations: small differences are directional, not statistical proof.
