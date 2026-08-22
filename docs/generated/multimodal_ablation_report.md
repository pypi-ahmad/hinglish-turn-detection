<!-- markdownlint-disable MD013 MD060 -->

# Turn detection ablation report

All completed runs use the same data partitions and train for **3 epochs**. Validation F1 selects the checkpoint. The comparison table reports test results.

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

This run uses attention pooling with the original, unaugmented audio. Evaluation F1 is 87.87%, accuracy is 87.61%, and the false-complete rate is 13.96%.

### M1_audio_plus_text

This run combines a Whisper audio embedding with a cached Whisper transcript embedding. Evaluation F1 is 88.12%, accuracy is 87.71%, and the false-complete rate is 15.19%. Compared with E1_no_augmentation, F1 rises by 0.25 points and the false-complete rate rises by 1.24 points.

## Summary

- Highest evaluation F1: **M1_audio_plus_text** (88.12%).
- Lowest false-complete rate among models with at least 75% recall: **E1_no_augmentation** (13.96%).
- These short-budget ablations use one seed, so small differences suggest a direction rather than prove one model is better.
