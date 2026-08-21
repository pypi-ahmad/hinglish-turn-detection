# Safety-calibrated finalist

Nine matched six-epoch runs compared E1, E4, and E8 across seeds 42, 43, and 44. Thresholds were selected only on validation data, maximizing F1 subject to false-complete rate (FCR) ≤ 10% and recall ≥ 85%. All nine selected checkpoints met both validation constraints.

| Architecture | Median validation F1 | Median validation FCR |
| --- | ---: | ---: |
| E1: attention, original data | 87.67% | 9.87% |
| E4: last-frame pooling, augmented data | **89.82%** | **9.21%** |
| E8: attention, partial fine-tuning | 88.34% | 9.87% |

E4 won. Median-performing E4 seed 44 became `checkpoints/safety_finalist/best.pt`; its checkpoint threshold is 0.5777403.

## Final held-out check

Held-out test was evaluated once after selection and was not used to choose architecture, seed, epoch, or threshold.

| Operating point | Accuracy | Precision | Recall | F1 | AUC | FCR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibrated 0.5777403 | 86.69% | 89.55% | 83.26% | 86.29% | 94.47% | **9.84%** |
| Fixed 0.5 control | 87.10% | 88.74% | 85.17% | 86.92% | 94.47% | 10.95% |

Calibration achieved the interruption-safety target on held-out data, cutting FCR by 1.11 percentage points versus the same checkpoint at 0.5. Cost: recall fell 1.91 points and missed the desired 85% held-out recall by 1.74 points. This is a real precision/safety trade-off, not a universal quality gain. Test results remain informational; changing the threshold after seeing them would leak test information.
