<!-- markdownlint-disable MD013 MD022 MD032 MD058 -->

# Data exploration: pipecat-ai/smart-turn-data-v3.2-train

_Generated from a local subset of **7517 rows** (train) + **4890 rows** (test), pulled via `scripts/prepare_data.py`. See docs/01_data_preparation_approach.md for why we work from a bounded subset rather than the full 270,946-row / 41GB dataset._

## 1. Sample counts
- Train subset: 7517 rows
- Test subset: 4890 rows

## 2. Class distribution (train subset)
- Complete: 3733 (49.7%)
- Incomplete: 3784 (50.3%)

## 3. Duration distribution (seconds, train subset)
- mean=7.84, median=7.20, p10=3.56, p90=13.12, min=0.36, max=32.60

## 4. Original sample rates seen (before our resample-to-16kHz step)
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
Low-energy proxy: 20 ms RMS frames, 10 ms hop, RMS below 0.01 (-40 dBFS), with a pause requiring at least 100 ms. This is not voice-activity ground truth; gain and noise affect the threshold.
- Mean low-energy frame share: 17.1%
- Median low-energy frame share: 17.1%
- Clips with any >=100 ms pause: 69.9%
- Clips with an internal >=100 ms pause: 63.9%
- Clips with a trailing >=100 ms pause: 45.4%

## 7. Filler / synthetic-vs-human signal
- Rows with midfiller=True: 2445
- Rows with endfiller=True: 1545
- Synthetic (TTS-generated) rows: 5060 (67.3%)
- `spoken_text` is null for every inspected row. There is no ready-made transcript
  or explicit `code-switched`/`hinglish` label. The data preparation report
  explains how the project handles this gap.

## 8. Dataset class sanity check
`src.dataset.TurnDetectionDataset.__getitem__` returns a dict with a raw waveform + label (feature extraction happens later, in `collate_fn`, so this class stays reusable). One real example from the train subset:
- id: 0002e1f1-d00e-4063-816a-c42932faccb2
- waveform shape: (116480,), dtype: float32
- label: 1 (complete)
- language: hin

## 9. Sample audio files for manual inspection
Saved to `data/samples/` by `scripts/prepare_data.py` (a handful of labeled Hindi/English clips + an augmentation before/after preview). Ground-truth metadata for every root sample WAV is in `data/samples/labels.csv`.
