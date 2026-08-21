import unittest
from pathlib import Path

from app import PROJECT_ROOT, _format_prediction, checkpoint_label, resolve_checkpoint


class GradioAppTests(unittest.TestCase):
    def test_relative_checkpoint_resolves_from_project_root(self):
        resolved = resolve_checkpoint("checkpoints/model.pt")

        self.assertEqual(resolved, (PROJECT_ROOT / "checkpoints/model.pt").resolve())

    def test_incomplete_confidence_is_one_minus_complete_probability(self):
        headline, details = _format_prediction(
            {"prob_complete": 0.2, "decision": "incomplete", "latency_ms": 12.3}, "cpu"
        )

        self.assertIn("still speaking", headline)
        self.assertIn("Complete probability: 20.0%", details)
        self.assertIn("Decision confidence: 80.0%", details)

    def test_default_checkpoint_is_stable_canonical_path(self):
        resolved = resolve_checkpoint()

        self.assertIsNotNone(resolved)
        self.assertEqual(
            Path(resolved), PROJECT_ROOT / "checkpoints/baseline_attention_augmented/best.pt"
        )

    def test_external_checkpoint_label_hides_parent_path(self):
        label = checkpoint_label(Path("C:/Users/person/.cache/huggingface/best.pt"))

        self.assertEqual(label, "Checkpoint: `best.pt`")


if __name__ == "__main__":
    unittest.main()
