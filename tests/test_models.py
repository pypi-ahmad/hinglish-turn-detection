import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from src.models import MultimodalTurnDetectionModel


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(d_model=4, max_source_positions=500)
        self.embed_positions = nn.Embedding(500, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(4))

    def forward(self, input_features):
        batch = input_features.shape[0]
        return SimpleNamespace(last_hidden_state=torch.ones(batch, 400, 4))


class MultimodalModelTests(unittest.TestCase):
    def _build_model(self):
        whisper = SimpleNamespace(encoder=_Encoder())
        with patch("src.models.WhisperModel.from_pretrained", return_value=whisper):
            return MultimodalTurnDetectionModel(
                text_vocab_size=10, text_embedding_dim=3, text_pad_token_id=0
            )

    def test_fuses_audio_and_masked_text_embeddings(self):
        model = self._build_model()

        logits = model(
            torch.zeros(2, 80, 800),
            torch.ones(2, 400),
            input_ids=torch.tensor([[1, 2], [3, 0]]),
            text_attention_mask=torch.tensor([[1, 1], [1, 0]]),
        )

        self.assertEqual(logits.shape, (2,))

    def test_requires_transcript_tensors(self):
        model = self._build_model()

        with self.assertRaisesRegex(ValueError, "requires input_ids"):
            model(torch.zeros(1, 80, 800), torch.ones(1, 400))


if __name__ == "__main__":
    unittest.main()
