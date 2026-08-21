import unittest

from scripts.select_safety_finalist import select_finalist


def _result(experiment, seed, f1, fcr, feasible=True):
    return {
        "status": "completed",
        "evaluation_split": "validation",
        "experiment_id": experiment,
        "seed": seed,
        "val_metrics": {"f1": f1, "false_complete_rate": fcr},
        "threshold_calibration": {"feasible": feasible},
        "decision_threshold": 0.6,
        "checkpoint_path": f"{experiment}-{seed}.pt",
    }


class SafetySelectionTests(unittest.TestCase):
    def test_selects_architecture_median_then_median_seed(self):
        results = [
            _result("E1", 42, 0.80, 0.08),
            _result("E1", 43, 0.90, 0.09),
            _result("E1", 44, 0.85, 0.07),
            _result("E8", 42, 0.82, 0.06),
            _result("E8", 43, 0.83, 0.06),
            _result("E8", 44, 0.84, 0.06),
        ]

        selection = select_finalist(results)

        self.assertEqual(selection["winning_experiment_id"], "E1")
        self.assertEqual(selection["deployed_seed"], 44)

    def test_rejects_architecture_with_one_infeasible_seed(self):
        results = [
            _result("safe", 1, 0.80, 0.08),
            _result("safe", 2, 0.81, 0.08),
            _result("unsafe", 1, 0.90, 0.05),
            _result("unsafe", 2, 0.91, 0.05, feasible=False),
        ]

        self.assertEqual(select_finalist(results)["winning_experiment_id"], "safe")


if __name__ == "__main__":
    unittest.main()
