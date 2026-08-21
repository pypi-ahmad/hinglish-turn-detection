import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import soundfile as sf
import torch

from configs import config as cfg
from src.dataset import (
    AugmentConfig,
    TurnDetectionDataset,
    _ensure_annotation_provenance,
    build_hard_negative_indices,
    collate_fn,
    load_noise_bank,
    normalize_audio_waveform,
    stratified_split,
)


class _FeatureExtractor:
    def __call__(self, waveforms, **kwargs):
        return {"input_features": torch.zeros(len(waveforms), 80, 800)}


class _Tokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs["max_length"] == 64
        return {
            "input_ids": torch.tensor([[1, 2], [3, 0]]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]]),
        }


class DatasetRegressionTests(unittest.TestCase):
    def test_required_hinglish_fillers_are_configured(self):
        required = {"um", "uh", "matlab", "actually", "tho", "yaar", "bas", "wait", "ek second", "haan"}

        self.assertTrue(required.issubset(set(cfg.HINGLISH_FILLERS)))

    def test_zero_test_fraction_does_not_drop_rows(self):
        metadata = pl.DataFrame(
            {
                "id": [str(i) for i in range(10)],
                "language": ["hin"] * 10,
                "endpoint_bool": [True] * 10,
                "source_dataset": ["source"] * 10,
            }
        )

        train, validation, test = stratified_split(metadata, val_frac=0.2, test_frac=0.0)

        self.assertEqual((train.height, validation.height, test.height), (8, 2, 0))
        self.assertEqual(train.height + validation.height + test.height, metadata.height)

    def test_split_rejects_fractions_that_exhaust_training_data(self):
        metadata = pl.DataFrame(
            {"id": ["1"], "language": ["hin"], "endpoint_bool": [True], "source_dataset": ["source"]}
        )

        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            stratified_split(metadata, val_frac=0.6, test_frac=0.4)

    def test_filler_injection_preserves_label_and_uses_mid_position(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "clip.wav", np.ones(1600, dtype=np.float32), cfg.SAMPLE_RATE)
            metadata = pl.DataFrame(
                {
                    "path": ["clip.wav"],
                    "endpoint_bool": [True],
                    "id": ["sample"],
                    "language": ["hin"],
                    "midfiller": [False],
                    "endfiller": [False],
                    "synthetic": [False],
                }
            )
            dataset = TurnDetectionDataset(
                metadata,
                root=root,
                augment=AugmentConfig(
                    p_filler=1.0,
                    p_silence=0.0,
                    p_speed=0.0,
                    p_pitch=0.0,
                    p_noise=0.0,
                    p_volume=0.0,
                ),
                filler_bank=object(),
            )

            with patch("src.dataset.inject_filler", return_value=np.ones(800, dtype=np.float32)) as inject:
                item = dataset[0]

            self.assertEqual(item["label"], 1)
            self.assertTrue(item["midfiller"])
            self.assertFalse(item["endfiller"])
            self.assertEqual(item["augment_types"], ("mid_filler",))
            self.assertAlmostEqual(item["original_duration_s"], 0.1)
            self.assertAlmostEqual(item["duration_s"], 0.05)
            inject.assert_called_once()
            self.assertEqual(inject.call_args.kwargs["position"], "mid")

    def test_filler_injection_is_not_correlated_with_one_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "clip.wav", np.ones(1600, dtype=np.float32), cfg.SAMPLE_RATE)
            metadata = pl.DataFrame(
                {
                    "path": ["clip.wav", "clip.wav"],
                    "endpoint_bool": [True, False],
                    "id": ["complete", "incomplete"],
                    "language": ["hin", "hin"],
                    "midfiller": [False, False],
                    "endfiller": [False, False],
                    "synthetic": [False, False],
                }
            )
            dataset = TurnDetectionDataset(
                metadata,
                root=root,
                augment=AugmentConfig(
                    p_filler=1.0,
                    p_silence=0.0,
                    p_speed=0.0,
                    p_pitch=0.0,
                    p_noise=0.0,
                    p_volume=0.0,
                ),
                filler_bank=object(),
            )

            with patch("src.dataset.inject_filler", side_effect=lambda wav, *_args, **_kwargs: wav):
                items = [dataset[0], dataset[1]]

            self.assertEqual([item["label"] for item in items], [1, 0])
            self.assertEqual([item["augment_types"] for item in items], [("mid_filler",), ("mid_filler",)])

    def test_audio_normalization_accepts_both_stereo_layouts(self):
        channels_first = np.vstack([np.ones(16), np.zeros(16)])
        samples_first = channels_first.T

        first = normalize_audio_waveform(channels_first, cfg.SAMPLE_RATE)
        second = normalize_audio_waveform(samples_first, cfg.SAMPLE_RATE)

        np.testing.assert_allclose(first, np.full(16, 0.5, dtype=np.float32))
        np.testing.assert_allclose(second, first)

    def test_audio_normalization_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            normalize_audio_waveform(np.array([0.0, np.nan]), cfg.SAMPLE_RATE)

    def test_legacy_metadata_marks_annotation_provenance_unknown(self):
        legacy = pl.DataFrame({"midfiller": [False], "endfiller": [False]})

        upgraded = _ensure_annotation_provenance(legacy)

        self.assertIsNone(upgraded["midfiller_annotation_known"][0])
        self.assertIsNone(upgraded["endfiller_annotation_known"][0])

    def test_long_audio_is_cropped_before_pause_augmentation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "clip.wav", np.ones(cfg.MAX_REAL_SAMPLES + cfg.SAMPLE_RATE), cfg.SAMPLE_RATE)
            metadata = pl.DataFrame(
                {
                    "path": ["clip.wav"],
                    "endpoint_bool": [False],
                    "id": ["sample"],
                    "language": ["hin"],
                    "midfiller": [False],
                    "endfiller": [False],
                    "synthetic": [False],
                    "duration_s": [9.0],
                }
            )
            dataset = TurnDetectionDataset(
                metadata,
                root=root,
                augment=AugmentConfig(
                    p_filler=0.0,
                    p_silence=1.0,
                    p_speed=0.0,
                    p_pitch=0.0,
                    p_noise=0.0,
                    p_volume=0.0,
                ),
            )

            with patch("src.dataset.random.choice", return_value="mid"), patch(
                "src.dataset.insert_silence", side_effect=lambda wav, *_args, **_kwargs: wav
            ) as insert:
                item = dataset[0]

            self.assertEqual(len(insert.call_args.args[0]), cfg.MAX_REAL_SAMPLES)
            self.assertEqual(item["augment_types"], ("mid_silence",))
            self.assertEqual(item["duration_s"], cfg.MAX_DURATION_S)

    def test_incomplete_rows_receive_higher_pause_probability(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / "clip.wav", np.ones(1600, dtype=np.float32), cfg.SAMPLE_RATE)
            metadata = pl.DataFrame(
                {
                    "path": ["clip.wav", "clip.wav"],
                    "endpoint_bool": [True, False],
                    "id": ["complete", "incomplete"],
                    "language": ["hin", "hin"],
                    "midfiller": [False, False],
                    "endfiller": [False, False],
                    "synthetic": [False, False],
                }
            )
            dataset = TurnDetectionDataset(
                metadata,
                root=root,
                augment=AugmentConfig(
                    p_filler=0.0,
                    p_silence=0.3,
                    p_speed=0.0,
                    p_pitch=0.0,
                    p_noise=0.0,
                    p_volume=0.0,
                ),
            )

            with patch("src.dataset.random.random", return_value=0.4), patch(
                "src.dataset.random.choice", return_value="mid"
            ), patch("src.dataset.insert_silence", side_effect=lambda wav, *_args, **_kwargs: wav) as insert:
                complete = dataset[0]
                incomplete = dataset[1]

            self.assertFalse(complete["is_augmented"])
            self.assertEqual(incomplete["augment_types"], ("mid_silence",))
            self.assertEqual(insert.call_args.args[2:4], (100, 800))

    def test_hard_indices_use_row_positions_even_with_duplicate_ids(self):
        metadata = pl.DataFrame(
            {
                "id": ["duplicate", "duplicate", "other", "plain"],
                "endpoint_bool": [True, False, False, True],
                "duration_s": [1.0, 5.0, 5.0, 3.0],
                "midfiller": [False, True, False, False],
                "endfiller": [False, False, True, False],
            }
        )

        indices = build_hard_negative_indices(metadata)

        self.assertEqual(indices.tolist(), [0, 1, 2])

    def test_real_noise_files_override_synthetic_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            noise_dir = Path(directory)
            sf.write(noise_dir / "office.wav", np.ones(800, dtype=np.float32) * 0.1, cfg.SAMPLE_RATE)

            clips = load_noise_bank(noise_dir)

            self.assertEqual(len(clips), 1)
            self.assertEqual(clips[0].dtype, np.float32)
            self.assertEqual(len(clips[0]), 800)

    def test_invalid_augmentation_probability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "p_noise must be between 0 and 1"):
            AugmentConfig(p_noise=1.1)

    def test_collate_keeps_one_encoder_frame_for_short_audio(self):
        batch = [
            {"waveform": np.ones(1, dtype=np.float32), "label": 0},
            {"waveform": np.ones(321, dtype=np.float32), "label": 1},
        ]

        result = collate_fn(batch, _FeatureExtractor())

        self.assertEqual(result["attention_mask"].sum(dim=1).tolist(), [1.0, 2.0])
        self.assertTrue(torch.all(result["attention_mask"][:, -1] == 1))

    def test_collate_rejects_empty_audio(self):
        with self.assertRaisesRegex(ValueError, "empty waveform"):
            collate_fn([{"waveform": np.array([], dtype=np.float32), "label": 0}], _FeatureExtractor())

    def test_collate_tokenizes_cached_transcripts_for_multimodal_model(self):
        batch = [
            {"waveform": np.ones(100, dtype=np.float32), "label": 0, "transcript": "haan"},
            {"waveform": np.ones(100, dtype=np.float32), "label": 1, "transcript": "done"},
        ]

        result = collate_fn(batch, _FeatureExtractor(), tokenizer=_Tokenizer())

        self.assertEqual(result["input_ids"].tolist(), [[1, 2], [3, 0]])
        self.assertEqual(result["text_attention_mask"].tolist(), [[1, 1], [1, 0]])


if __name__ == "__main__":
    unittest.main()
