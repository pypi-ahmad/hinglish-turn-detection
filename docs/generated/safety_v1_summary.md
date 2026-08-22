# Safety-calibrated finalist

The study compares E1, E4, and E8 across nine matched six-epoch runs using seeds 42, 43, and 44. Threshold selection uses validation data only and maximizes F1 subject to a false-complete rate (FCR) ≤ 10% and recall ≥ 85%. All nine checkpoints met both constraints on validation data.

| Architecture | Median validation F1 | Median validation FCR |
| --- | ---: | ---: |
| E1: attention, original data | 87.67% | 9.87% |
| E4: last-frame pooling, augmented data | **89.82%** | **9.21%** |
| E8: attention, partial fine-tuning | 88.34% | 9.87% |

E4 had the best median result. The median-performing E4 run, seed 44, became `checkpoints/safety_finalist/best.pt` with a threshold of 0.5777403.

## Final held-out check

The held-out test set was evaluated once after selection. It was not used to choose the architecture, seed, epoch, or threshold.

| Operating point | Accuracy | Precision | Recall | F1 | AUC | FCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated 0.5777403 | 86.69% | 89.55% | 83.26% | 86.29% | 94.47% | **9.84%** |
| Fixed 0.5 control | 87.10% | 88.74% | 85.17% | 86.92% | 94.47% | 10.95% |

Calibration met the interruption-safety target on held-out data and reduced FCR by 1.11 percentage points compared with the same checkpoint at 0.5. Recall fell by 1.91 points and missed the desired 85% held-out target by 1.74 points. This is a safety trade-off rather than an overall quality improvement. The test results are informational; changing the threshold after seeing them would leak test information.
