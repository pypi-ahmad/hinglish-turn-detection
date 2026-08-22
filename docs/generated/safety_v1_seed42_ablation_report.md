<!-- markdownlint-disable MD013 MD060 -->

# Turn detection ablation report

All completed runs use the same data partitions and train for **6 epochs**. The checkpoint is selected by maximum validation F1 subject to FCR ≤10% and recall ≥85%. The comparison table reports validation results.

## Comparison

| Experiment | Status | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | FP32 MB | CPU ms | GPU ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | completed | 88.83% | 89.70% | 87.50% | 88.59% | 95.08% | 9.87% | 8,000,386 | 30.52 | 29.18 | 5.76 |
| E4_last_pool | completed | 88.38% | 89.61% | 86.61% | 88.08% | 95.64% | 9.87% | 7,901,569 | 30.14 | 26.70 | 4.88 |
| E8_partial_finetune | completed | 88.50% | 89.63% | 86.83% | 88.21% | 94.89% | 9.87% | 8,000,386 | 30.52 | 27.88 | 5.83 |

## Pause and hard-case slices

| Experiment | Internal-pause F1 | Internal-pause FCR | Hindi filler+pause F1 | Hard-Hinglish F1 | Hard-Hinglish FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 88.26% | 8.01% | 68.97% | 60.61% | 17.07% |
| E4_last_pool | 87.66% | 8.01% | 80.00% | 75.00% | 9.76% |
| E8_partial_finetune | 88.77% | 7.37% | 76.92% | 68.97% | 7.32% |

## Per-experiment insights

### E1_no_augmentation

Attention pooling on original, unaugmented audio. Evaluation F1 88.59%; accuracy 88.83%; false-complete 9.87%.

### E4_last_pool

Last-frame pooling with same augmented data. Evaluation F1 88.08%; accuracy 88.38%; false-complete 9.87%.

### E8_partial_finetune

This run freezes the first two Whisper encoder layers. Evaluation F1 is 88.21%, accuracy is 88.50%, and the false-complete rate is 9.87%. Partial tuning keeps nearly all the F1 of full tuning with far fewer trainable parameters.

## Summary

- Highest evaluation F1: **E1_no_augmentation** (88.59%).
- Lowest false-complete rate among models with at least 75% recall: **E1_no_augmentation** (9.87%).
- This report covers one seed, so small differences suggest a direction rather than prove one setup is better.
