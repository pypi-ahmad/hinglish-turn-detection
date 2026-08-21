<!-- markdownlint-disable MD013 MD060 -->

# Turn detection ablation report

All completed runs use matched data partitions and **3 epochs**. Checkpoint selection uses validation F1; comparison rows currently contain: test.

## Comparison

| Experiment | Status | Accuracy | Precision | Recall | F1 | AUC | False-complete | Params | FP32 MB | CPU ms | GPU ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | completed | 87.61% | 86.62% | 89.15% | 87.87% | 93.64% | 13.96% | 8,000,386 | 30.52 | 32.49 | 6.70 |
| E2_augmented | completed | 87.75% | 85.95% | 90.45% | 88.14% | 93.79% | 14.99% | 8,000,386 | 30.52 | 29.07 | 6.27 |
| E3_mean_pool | completed | 86.05% | 83.76% | 89.68% | 86.62% | 92.98% | 17.62% | 7,901,569 | 30.14 | 28.10 | 5.24 |
| E4_last_pool | completed | 87.32% | 85.58% | 89.96% | 87.72% | 93.86% | 15.36% | 7,901,569 | 30.14 | 28.38 | 5.27 |
| E5_short_pauses | completed | 87.65% | 86.09% | 90.00% | 88.00% | 93.58% | 14.74% | 8,000,386 | 30.52 | 28.79 | 5.62 |
| E6_long_pauses | completed | 87.26% | 84.86% | 90.90% | 87.78% | 93.35% | 16.43% | 8,000,386 | 30.52 | 29.63 | 5.68 |
| E7_frozen_encoder | completed | 71.53% | 69.10% | 78.59% | 73.54% | 77.51% | 35.61% | 8,000,386 | 30.52 | 29.87 | 6.25 |
| E8_partial_finetune | completed | 87.46% | 85.08% | 91.06% | 87.97% | 93.32% | 16.18% | 8,000,386 | 30.52 | 32.07 | 6.97 |
| E11_no_silence | completed | 87.12% | 86.14% | 88.66% | 87.38% | 93.53% | 14.45% | 8,000,386 | 30.52 | 25.12 | 4.91 |

## Pause and hard-case slices

| Experiment | Internal-pause F1 | Internal-pause FCR | Hindi filler+pause F1 | Hard-Hinglish F1 | Hard-Hinglish FCR |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1_no_augmentation | 87.78% | 12.39% | 84.72% | 82.99% | 10.28% |
| E2_augmented | 88.55% | 12.83% | 87.01% | 84.81% | 14.95% |
| E3_mean_pool | 86.91% | 15.44% | 85.16% | 83.54% | 15.89% |
| E4_last_pool | 87.62% | 13.66% | 89.74% | 88.05% | 13.08% |
| E5_short_pauses | 87.84% | 12.83% | 86.27% | 84.08% | 14.95% |
| E6_long_pauses | 88.28% | 14.74% | 87.18% | 85.00% | 15.89% |
| E7_frozen_encoder | 73.64% | 32.78% | 66.67% | 64.43% | 24.30% |
| E8_partial_finetune | 88.18% | 14.49% | 88.89% | 87.18% | 12.15% |
| E11_no_silence | 87.82% | 13.02% | 81.88% | 79.74% | 15.89% |

## Per-experiment insights

### E1_no_augmentation

Attention pooling on original, unaugmented audio. Evaluation F1 87.87%; accuracy 87.61%; false-complete 13.96%.

### E2_augmented

Attention pooling with full Hinglish/pause augmentation. Evaluation F1 88.14%; accuracy 87.75%; false-complete 14.99%. Versus E1_no_augmentation: F1 +0.28 points; false-complete +1.03 points.

### E3_mean_pool

Mean pooling with same augmented data. Evaluation F1 86.62%; accuracy 86.05%; false-complete 17.62%. Versus E2_augmented: F1 -1.52 points; false-complete +2.63 points.

### E4_last_pool

Last-frame pooling with same augmented data. Evaluation F1 87.72%; accuracy 87.32%; false-complete 15.36%. Versus E2_augmented: F1 -0.42 points; false-complete +0.37 points.

### E5_short_pauses

Silence augmentation restricted to 50-250 ms. Evaluation F1 88.00%; accuracy 87.65%; false-complete 14.74%. Versus E2_augmented: F1 -0.14 points; false-complete -0.25 points.

### E6_long_pauses

Silence augmentation restricted to 600-1500 ms. Evaluation F1 87.78%; accuracy 87.26%; false-complete 16.43%. Versus E2_augmented: F1 -0.36 points; false-complete +1.44 points.

### E7_frozen_encoder

Freeze all four Whisper encoder layers. Evaluation F1 73.54%; accuracy 71.53%; false-complete 35.61%. Versus E2_augmented: F1 -14.60 points; false-complete +20.63 points. Recall is only 78.59%; its low false-complete rate is caused by over-predicting incomplete, not by a useful operating point. This equal-LR ablation does not test a frozen-head-specific learning rate or longer schedule.

### E8_partial_finetune

Freeze first two Whisper encoder layers. Evaluation F1 87.97%; accuracy 87.46%; false-complete 16.18%. Versus E2_augmented: F1 -0.17 points; false-complete +1.19 points. Partial tuning retains nearly all full-tuning F1 with substantially fewer trainable parameters.

### E11_no_silence

Full augmentation except silence insertion. Evaluation F1 87.38%; accuracy 87.12%; false-complete 14.45%. Versus E2_augmented: F1 -0.76 points; false-complete -0.54 points.

## Summary

- Highest evaluation F1: **E2_augmented** (88.14%).
- Lowest false-complete rate among models with at least 75% recall: **E1_no_augmentation** (13.96%).
- Single-seed, short-budget ablations: small differences are directional, not statistical proof.
