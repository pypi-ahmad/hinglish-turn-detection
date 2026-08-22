<!-- markdownlint-disable MD013 MD022 MD032 MD058 -->

# Data exploration: pipecat-ai/smart-turn-data-v3.2-train

_This report uses a local subset of **7517 training rows** and **4890 test rows**, downloaded with `scripts/prepare_data.py`. The reason for using a bounded subset instead of the full 270,946-row, 41 GB dataset is documented in `docs/01_data_preparation_approach.md`._

## 1. Sample counts
- Train subset: 7517 rows
- Test subset: 4890 rows

## 2. Class distribution (train subset)
- Complete: 3733 (49.7%)
- Incomplete: 3784 (50.3%)

## 3. Duration distribution (seconds, train subset)
- mean=7.84, median=7.20, p10=3.56, p90=13.12, min=0.36, max=32.60

## 4. Original sample rates before resampling to 16 kHz
| orig_sample_rate | n |
| --- | --- |
| 16000 | 7517 |

## 5. Languages present in this subset
| language | n |
| --- | --- |
| eng | 3646 |
| hin | 721 |
| ara | 150 |
| spa | 150 |
| rus | 150 |
| ben | 150 |
| pol | 150 |
| kor | 150 |
| ita | 150 |
| fra | 150 |
| ukr | 150 |
| ind | 150 |
| deu | 150 |
| fin | 150 |
| por | 150 |
| tur | 150 |
| nld | 150 |
| nor | 150 |
| dan | 150 |
| zho | 150 |
| mar | 150 |
| vie | 150 |
| jpn | 150 |

## 6. Silence / pause presence (train subset)
The pause proxy uses 20 ms RMS frames with a 10 ms hop. A frame is low energy when RMS is below 0.01 (-40 dBFS), and a pause must last at least 100 ms. This is not voice-activity ground truth because gain and background noise affect the threshold.
- Mean low-energy frame share: 17.1%
- Median low-energy frame share: 17.1%
- Clips with any >=100 ms pause: 69.9%
- Clips with an internal >=100 ms pause: 63.9%
- Clips with a trailing >=100 ms pause: 45.4%

## 7. Filler / synthetic-vs-human signal
- Rows with midfiller=True: 2445
- Rows with endfiller=True: 1545
- Synthetic (TTS-generated) rows: 5060 (67.3%)
- `spoken_text` is null for every inspected row. The data has no transcript or
  explicit `code-switched`/`hinglish` label. The data preparation report explains
  how the project works within this limitation.

## 8. Dataset class sanity check
`src.dataset.TurnDetectionDataset.__getitem__` returns a dictionary containing a raw waveform and label. Feature extraction happens later in `collate_fn`, which keeps the dataset class reusable. This is one example from the training subset:
- id: 0002e1f1-d00e-4063-816a-c42932faccb2
- waveform shape: (116480,), dtype: float32
- label: 1 (complete)
- language: hin

## 9. Sample audio files for manual inspection
`scripts/prepare_data.py` saves a small set of labeled Hindi and English clips to `data/samples/`, along with a before-and-after augmentation preview. Metadata for each sample WAV in the root of that directory is stored in `data/samples/labels.csv`.
