<!-- markdownlint-disable MD013 MD060 -->

# Turn detection ablation report

All completed runs use the same data partitions and train for **6 epochs**. The checkpoint is selected by maximum validation F1 subject to FCR ≤10% and recall ≥85%. The comparison table reports validation results.

## Comparison

| Experiment | Status | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | FP32 MB | CPU ms | GPU ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | completed | 88.05% | 89.72% | 85.71% | 87.67% | 94.33% | 9.65% | 8,000,386 | 30.52 | 29.73 | 5.71 |
| E4_last_pool | completed | 90.04% | 91.06% | 88.62% | 89.82% | 95.53% | 8.55% | 7,901,569 | 30.14 | 24.71 | 4.81 |
| E8_partial_finetune | completed | 89.16% | 90.89% | 86.83% | 88.81% | 95.83% | 8.55% | 8,000,386 | 30.52 | 26.15 | 5.24 |

## Pause and hard-case slices

| Experiment | Internal-pause F1 | Internal-pause FCR | Hindi filler+pause F1 | Hard-Hinglish F1 | Hard-Hinglish FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 87.05% | 8.01% | 78.57% | 73.33% | 7.32% |
| E4_last_pool | 89.29% | 6.73% | 76.92% | 68.97% | 7.32% |
| E8_partial_finetune | 87.57% | 7.37% | 78.57% | 75.86% | 4.88% |

## Per-experiment insights

### E1_no_augmentation

Attention pooling on original, unaugmented audio. Evaluation F1 87.67%; accuracy 88.05%; false-complete 9.65%.

### E4_last_pool

Last-frame pooling with same augmented data. Evaluation F1 89.82%; accuracy 90.04%; false-complete 8.55%.

### E8_partial_finetune

This run freezes the first two Whisper encoder layers. Evaluation F1 is 88.81%, accuracy is 89.16%, and the false-complete rate is 8.55%. Partial tuning keeps nearly all the F1 of full tuning with far fewer trainable parameters.

## Summary

- Highest evaluation F1: **E4_last_pool** (89.82%).
- Lowest false-complete rate among models with at least 75% recall: **E4_last_pool** (8.55%).
- This report covers one seed, so small differences suggest a direction rather than prove one setup is better.
