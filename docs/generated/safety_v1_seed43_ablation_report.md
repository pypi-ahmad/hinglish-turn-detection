<!-- markdownlint-disable MD013 MD060 -->

# Turn Detection Ablation Report

All completed runs use matched data partitions and **6 epochs**. Checkpoint selection uses maximum validation F1 subject to FCR ≤10% and recall ≥85%; comparison rows contain validation results.

## Comparison

| Experiment | Status | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | FP32 MB | CPU ms | GPU ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | completed | 87.61% | 89.44% | 85.04% | 87.19% | 94.82% | 9.87% | 8,000,386 | 30.52 | 29.43 | 5.42 |
| E4_last_pool | completed | 90.04% | 90.50% | 89.29% | 89.89% | 96.48% | 9.21% | 7,901,569 | 30.14 | 28.86 | 5.13 |
| E8_partial_finetune | completed | 88.61% | 89.66% | 87.05% | 88.34% | 95.15% | 9.87% | 8,000,386 | 30.52 | 28.81 | 5.39 |

## Pause and hard-case slices

| Experiment | Internal-pause F1 | Internal-pause FCR | Hindi filler+pause F1 | Hard-Hinglish F1 | Hard-Hinglish FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 86.85% | 8.01% | 77.42% | 68.57% | 17.07% |
| E4_last_pool | 90.07% | 6.73% | 86.67% | 78.79% | 9.76% |
| E8_partial_finetune | 87.10% | 8.33% | 83.87% | 81.25% | 7.32% |

## Per-experiment insights

### E1_no_augmentation

Attention pooling on original, unaugmented audio. Evaluation F1 87.19%; accuracy 87.61%; false-complete 9.87%.

### E4_last_pool

Last-frame pooling with same augmented data. Evaluation F1 89.89%; accuracy 90.04%; false-complete 9.21%.

### E8_partial_finetune

Freeze first two Whisper encoder layers. Evaluation F1 88.34%; accuracy 88.61%; false-complete 9.87%. Partial tuning retains nearly all full-tuning F1 with substantially fewer trainable parameters.

## Summary

- Highest evaluation F1: **E4_last_pool** (89.89%).
- Lowest false-complete rate among models with at least 75% recall: **E4_last_pool** (9.21%).
- Single-seed, short-budget ablations: small differences are directional, not statistical proof.
