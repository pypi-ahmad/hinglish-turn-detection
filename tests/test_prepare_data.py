import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

from scripts.prepare_data import preview_augmentations
from src.dataset import measure_pause_features


class PreparationRegressionTests(unittest.TestCase):
    def test_zero_preview_skips_tts_and_removes_stale_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            (out_dir / "stale.wav").write_bytes(b"stale")

            with patch("scripts.prepare_data.FillerBank") as filler_bank:
                result = preview_augmentations(pl.DataFrame(), 0, out_dir)

            self.assertEqual(result, {"n_examples": 0, "examples": []})
            self.assertFalse((out_dir / "stale.wav").exists())
            filler_bank.assert_not_called()

    def test_pause_measurement_distinguishes_internal_and_trailing_silence(self):
        sr = 1_000
        speech = np.ones(200, dtype=np.float32) * 0.2
        silence = np.zeros(120, dtype=np.float32)

        internal = measure_pause_features(np.concatenate([speech, silence, speech]), sr)
        trailing = measure_pause_features(np.concatenate([speech, silence]), sr)

        self.assertTrue(internal["any_pause"])
        self.assertTrue(internal["internal_pause"])
        self.assertFalse(internal["trailing_pause"])
        self.assertTrue(trailing["trailing_pause"])


if __name__ == "__main__":
    unittest.main()
