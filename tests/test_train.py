import unittest

import numpy as np

from src.train import _class_balanced_sample_weights


class SamplerWeightTests(unittest.TestCase):
    def test_hard_boost_preserves_equal_class_mass(self):
        labels = np.array([0, 0, 0, 1, 1], dtype=bool)
        weights = _class_balanced_sample_weights(labels, np.array([0, 3]), hard_weight=3.0)

        self.assertAlmostEqual(float(weights[~labels].sum()), float(weights[labels].sum()))
        self.assertAlmostEqual(float(weights[0] / weights[1]), 3.0)
        self.assertAlmostEqual(float(weights[3] / weights[4]), 3.0)

    def test_both_classes_are_required(self):
        with self.assertRaisesRegex(ValueError, "both endpoint classes"):
            _class_balanced_sample_weights(np.array([0, 0]), np.array([], dtype=int), hard_weight=3.0)


if __name__ == "__main__":
    unittest.main()
