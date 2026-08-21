import unittest

import numpy as np

from src.evaluate import classification_metrics, select_operating_threshold


class ThresholdCalibrationTests(unittest.TestCase):
    def test_selects_best_f1_that_meets_safety_constraints(self):
        labels = np.array([0] * 10 + [1] * 10)
        probs = np.array([0.05, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.60]
                         + [0.40, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90])

        calibration = select_operating_threshold(
            labels, probs, max_false_complete_rate=0.10, min_recall=0.85
        )

        self.assertTrue(calibration["feasible"])
        self.assertLessEqual(calibration["metrics"]["false_complete_rate"], 0.10)
        self.assertGreaterEqual(calibration["metrics"]["recall"], 0.85)
        self.assertEqual(calibration["threshold"], 0.40)

    def test_marks_conflicting_constraints_infeasible(self):
        labels = np.array([0, 0, 1, 1])
        probs = np.array([0.8, 0.9, 0.7, 0.85])

        calibration = select_operating_threshold(
            labels, probs, max_false_complete_rate=0.0, min_recall=1.0
        )

        self.assertFalse(calibration["feasible"])
        self.assertEqual(calibration["selection_reason"], "constraints_infeasible_lowest_fcr_at_recall_floor")
        self.assertEqual(calibration["metrics"]["recall"], 1.0)

    def test_metrics_include_confusion_counts(self):
        metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.2, 0.8, 0.4, 0.9]), 0.5)

        self.assertEqual(
            {key: metrics[key] for key in ("true_negative", "false_positive", "false_negative", "true_positive")},
            {"true_negative": 1, "false_positive": 1, "false_negative": 1, "true_positive": 1},
        )


if __name__ == "__main__":
    unittest.main()
