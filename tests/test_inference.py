import threading
import unittest

import numpy as np
import torch

from src.inference import TurnDetector, _load_and_resample, _validate_threshold


class InferenceInputTests(unittest.TestCase):
    def test_int16_pcm_is_scaled_to_float_audio(self):
        waveform = _load_and_resample(np.array([-32768, 0, 32767], dtype=np.int16))

        self.assertEqual(waveform.dtype, np.float32)
        self.assertAlmostEqual(float(waveform[0]), -1.0)
        self.assertLessEqual(float(waveform[-1]), 1.0)

    def test_empty_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            _load_and_resample(np.array([], dtype=np.float32))

    def test_non_finite_audio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinite"):
            _load_and_resample(np.array([0.0, np.nan], dtype=np.float32))

    def test_invalid_sample_rate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _load_and_resample((np.ones(100, dtype=np.float32), 0))

    def test_missing_audio_path_is_rejected_before_decode(self):
        with self.assertRaisesRegex(FileNotFoundError, "audio file not found"):
            _load_and_resample("does-not-exist.wav")

    def test_threshold_must_be_finite_probability(self):
        self.assertEqual(_validate_threshold(0.5), 0.5)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _validate_threshold(invalid)
        with self.assertRaises(TypeError):
            _validate_threshold(True)

    def test_prepare_audio_keeps_last_eight_seconds(self):
        waveform = np.arange(9 * 16_000, dtype=np.float32)

        prepared = TurnDetector.prepare_audio(waveform)

        np.testing.assert_array_equal(prepared, waveform[-8 * 16_000 :])

    def test_predict_uses_checkpoint_threshold_unless_overridden(self):
        detector = TurnDetector.__new__(TurnDetector)
        detector.decision_threshold = 0.7
        detector.multimodal = False
        detector._inference_lock = threading.Lock()
        detector._synchronize = lambda: None
        detector._to_batch = lambda audios: {}
        detector._forward = lambda batch: torch.tensor([0.4054651])  # sigmoid = 0.6

        default_result = detector.predict(np.ones(10, dtype=np.float32))
        override_result = detector.predict(np.ones(10, dtype=np.float32), threshold=0.5)

        self.assertEqual(default_result["decision"], "incomplete")
        self.assertEqual(default_result["threshold"], 0.7)
        self.assertEqual(override_result["decision"], "complete")
        self.assertEqual(override_result["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
