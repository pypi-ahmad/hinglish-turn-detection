"""
Turn-detection model architecture.

WHAT THIS MODEL DOES
---------------------
Given up to 8 seconds of 16kHz audio (the *end* of whatever the user just said),
predict whether their conversational turn is COMPLETE (they're done talking and
the AI should respond) or INCOMPLETE (they're just pausing / trailing off into a
filler word like "matlab..." or "umm..." and are about to keep going).

This is a binary classification problem over raw audio -- no transcript needed.
The model has to pick up on prosody (falling vs. rising/flat intonation, pace),
filler words, and mid-sentence pause patterns, which is exactly why we work on
the audio waveform (via log-mel spectrogram) rather than on text.

WHY THIS EXACT ARCHITECTURE
----------------------------
Pipecat AI open-sourced a production turn-detection model for the same task,
"smart-turn-v3", and this dataset (pipecat-ai/smart-turn-data-v3.2-train) IS
their own training data. We confirmed their architecture directly from their
`train.py` on GitHub:

    base encoder  = openai/whisper-tiny, ENCODER ONLY (no decoder)   ~7.6M params
    pooling       = learned attention pooling over encoder frames
    head          = small MLP -> single logit -> sigmoid = P(turn complete)
    total size    = ~8M params (matches their published model card)

We deliberately copy this architecture instead of inventing our own, for two
reasons:
  1. It's already proven at ~99% accuracy on English and 87-97% across 14
     languages in production, so it's a strong, validated starting point.
  2. Pipecat only publish ONNX exports of v3 on HuggingFace (no .safetensors /
     PyTorch checkpoint), so there is no ready-made checkpoint to load and
     fine-tune directly with `from_pretrained`. To fine-tune at all, we have to
     rebuild the architecture in PyTorch ourselves and initialize it from the
     same base they used (openai/whisper-tiny), then train it -- which is what
     `train.py` in this project does, but weighted/augmented towards Hinglish.

WHY "ENCODER ONLY" (no Whisper decoder)
-----------------------------------------
Whisper is a full speech-to-text model: an audio ENCODER (turns a mel
spectrogram into a sequence of hidden vectors) feeding a text DECODER
(autoregressively generates transcript tokens). We only need "what does this
audio sound like", not "what words were said", so we keep just the encoder and
discard the decoder entirely. This is also why whisper-tiny's decoder (~30M
params) doesn't count towards our ~8M parameter budget -- it's never loaded
into the trainable model, just briefly instantiated and then dropped.

WHY ATTENTION POOLING (instead of e.g. mean-pooling or taking the last frame)
-------------------------------------------------------------------------------
The encoder outputs one 384-dim vector per ~20ms audio frame (up to 800 frames
for 8 seconds of audio). We need a single vector to classify. Mean-pooling
would treat every 20ms frame as equally important, which is wrong: the frames
right around a pause / filler word / trailing intonation at the *end* of the
clip carry almost all the signal about whether the turn is over. A small
learned attention layer lets the model decide, per-example, which frames to
weight most heavily -- e.g. it can learn to focus on the last ~1 second where
a "matlab..." filler and a falling silence would show up.

WHY THIS SPECIFIC MLP HEAD SHAPE (Linear->LayerNorm->GELU->Dropout->Linear->GELU->Linear)
-------------------------------------------------------------------------------------------
This is copied verbatim from Pipecat's own head so that our from-scratch
reimplementation has the same capacity/inductive bias as the production
model we're trying to match or beat. LayerNorm after the first projection
keeps activation scale stable regardless of how "loud" the pooled encoder
vector is (audio energy varies a lot between recordings); GELU is a smooth
nonlinearity that tends to train a bit better than ReLU for small MLPs;
Dropout(0.1) is a light regularizer given we're fine-tuning on a fairly small,
partly-synthetic (TTS-generated) dataset where overfitting is a real risk.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import WhisperModel

# Default base checkpoint we fine-tune from. "openai/whisper-tiny" is the
# smallest official Whisper checkpoint (~39M params full seq2seq model; we
# only keep its ~7.6M-param encoder). This is also exactly what Pipecat's own
# train.py uses as `CONFIG["base_model_name"]`.
BASE_MODEL_NAME = "openai/whisper-tiny"


class AttentionPool(nn.Module):
    """Learned attention pooling over a sequence of per-frame encoder vectors.

    Turns (batch, seq_len, hidden_size) into (batch, hidden_size) by computing
    a scalar "how important is this frame" score for every timestep (via a
    tiny 2-layer MLP: Linear -> Tanh -> Linear), softmax-normalizing those
    scores across the sequence, and taking the weighted sum of frame vectors.
    This is the standard "additive attention pooling" trick used a lot in
    speech/NLP classification heads (e.g. it's how many sentence-embedding
    and speaker-verification models pool variable-length sequences).
    """

    def __init__(self, hidden_size: int, attn_dim: int = 256) -> None:
        """Create additive attention scorer for encoder frame pooling."""
        super().__init__()
        # Two linear layers with a Tanh in between: this is intentionally a
        # *small* network (attn_dim=256, matching Pipecat's own head) since
        # its only job is to produce one importance score per frame, not to
        # do heavy representation learning -- that's the encoder's job.
        self.attn = nn.Sequential(
            nn.Linear(hidden_size, attn_dim),
            nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pool encoder frames using learned masked additive attention.

        Args:
            hidden_states: (batch, seq_len, hidden_size) -- one vector per
                audio frame, as produced by the Whisper encoder.
            attention_mask: optional (batch, seq_len) mask with 1 for real
                frames and 0 for padding frames (we left-pad short clips, see
                data.py), so padding never gets attention weight.
        Returns:
            (batch, hidden_size) pooled representation of the whole clip.
        """
        # Raw importance score per frame, one scalar per timestep.
        scores = self.attn(hidden_states).squeeze(-1)  # (batch, seq_len)

        if attention_mask is not None:
            # Push padded frames' scores to -inf so softmax gives them ~0
            # weight, regardless of what the (arbitrary, learned) attention
            # network happens to output for meaningless padding input.
            scores = scores.masked_fill(attention_mask == 0, float("-inf"))

        # Normalize scores into a probability distribution over frames, then
        # take the weighted average of frame vectors using those weights.
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)
        return (hidden_states * weights).sum(dim=1)  # (batch, hidden_size)


class MeanPool(nn.Module):
    """Masked mean-pooling over frames: every real (non-padding) frame gets
    equal weight. This is the simplest possible pooling and serves as the
    control condition in the pooling-strategy ablation (see
    docs/02_experiment_plan.md) against learned `AttentionPool`. The
    hypothesis we're testing with that ablation: does letting the model learn
    *which* frames matter (attention) actually beat treating them all
    equally, specifically on pause/filler-heavy Hinglish clips where the
    diagnostic signal is concentrated near the end rather than spread evenly?
    """

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Average real encoder frames while excluding padded positions."""
        if attention_mask is None:
            return hidden_states.mean(dim=1)
        mask = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
        summed = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1.0)
        return summed / count


class LastFramePool(nn.Module):
    """Take only the final frame's hidden vector.

    Because our data pipeline LEFT-pads short clips (real audio always ends
    at the last position of the fixed-length input -- see dataset.py's
    `collate_fn`), "the last frame" is always genuine audio, never padding,
    regardless of how long the original clip was. This is the cheapest
    possible pooling (no learned parameters, O(1) instead of O(seq_len)) and
    is the third arm of the pooling ablation: does the model even need
    anything more than "look at the very last instant of audio", or does
    context from earlier in the clip (which mean/attention pooling both see)
    actually help decide completeness?
    """

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return final encoder frame, which aligns with utterance ending."""
        return hidden_states[:, -1, :]


_POOLING_REGISTRY = {
    "attention": AttentionPool,
    "mean": MeanPool,
    "last": LastFramePool,
}


class TurnDetectionModel(nn.Module):
    """Whisper-tiny encoder + a configurable pooling layer + MLP head ->
    P(turn complete).

    Forward pass expects pre-computed log-mel spectrogram features (produced
    by `transformers.WhisperFeatureExtractor`, NOT raw waveform samples) --
    see dataset.py's `collate_fn` for the exact preprocessing that has to
    happen before audio reaches this model.
    """

    def __init__(
        self,
        base_model_name: str = BASE_MODEL_NAME,
        freeze_encoder_layers: int = 0,
        pooling: str = "attention",
    ) -> None:
        """
        Args:
            base_model_name: HF hub id of the Whisper checkpoint to start from.
                Defaults to whisper-tiny to match Pipecat's own smart-turn-v3.
            freeze_encoder_layers: if > 0, freeze the *first* N transformer
                layers of the encoder (leaving later layers + the head
                trainable). Whisper-tiny has 4 encoder layers total. Freezing
                early layers can help when fine-tuning on a smaller dataset
                than the original pretraining data, since early layers tend
                to learn generic low-level acoustic features that transfer
                fine as-is, while later layers specialize. Default 0 = fully
                fine-tune everything, which is fine given our dataset is still
                tens of thousands of examples.
            pooling: one of "attention" (learned, default -- matches
                Pipecat's own smart-turn-v3), "mean" (unweighted average over
                real frames), or "last" (just the final frame). Exists so the
                pooling-strategy ablation in docs/02_experiment_plan.md can
                swap this one line without touching anything else.
        """
        super().__init__()

        # We load the FULL WhisperModel (encoder + decoder) via
        # `from_pretrained` because that's the only public entry point that
        # correctly restores the pretrained encoder weights from the
        # checkpoint. We then immediately throw the decoder away -- we never
        # call it, and `del` lets Python's garbage collector free its memory
        # right away rather than it sitting around unused for the rest of
        # training.
        whisper = WhisperModel.from_pretrained(base_model_name)
        self.encoder = whisper.encoder
        del whisper

        hidden_size = self.encoder.config.d_model  # 384 for whisper-tiny

        # Official Smart Turn v3.2 uses an 8-second feature window (800 mel
        # frames -> 400 encoder positions), not Whisper ASR's native 30-second
        # window. Keep Whisper's pretrained first 400 positions instead of
        # randomly reinitializing them: our bounded 6k-row training subset is
        # far smaller than Pipecat's 270k-row run and needs that prior signal.
        position_weights = self.encoder.embed_positions.weight[:400].detach().clone()
        self.encoder.config.max_source_positions = 400
        self.encoder.embed_positions = nn.Embedding.from_pretrained(position_weights, freeze=False)

        n_encoder_layers = len(self.encoder.layers)
        if not 0 <= freeze_encoder_layers <= n_encoder_layers:
            raise ValueError(f"freeze_encoder_layers must be between 0 and {n_encoder_layers}")
        if freeze_encoder_layers == n_encoder_layers:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        elif freeze_encoder_layers > 0:
            for layer in self.encoder.layers[:freeze_encoder_layers]:
                for p in layer.parameters():
                    p.requires_grad_(False)

        if pooling not in _POOLING_REGISTRY:
            raise ValueError(f"pooling must be one of {list(_POOLING_REGISTRY)}, got {pooling!r}")
        self.pooling_name = pooling
        self.pool = _POOLING_REGISTRY[pooling](hidden_size) if pooling == "attention" else _POOLING_REGISTRY[pooling]()

        # The classification head. Shape/order copied from Pipecat's own
        # smart-turn-v3 training code so our from-scratch reimplementation
        # has the same capacity as the model we're benchmarking against.
        # Output is a single raw logit; callers apply sigmoid to get a
        # probability, or use BCEWithLogitsLoss (which applies sigmoid
        # internally, more numerically stable) during training.
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),   # stabilizes scale of the pooled vector before the MLP
            nn.GELU(),
            nn.Dropout(0.1),     # light regularization against overfitting the fine-tuning set
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        for module in [*self.classifier, *self.pool.modules()]:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def encode_audio(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return one pooled Whisper encoder vector per audio clip."""
        encoder_out = self.encoder(input_features).last_hidden_state
        return self.pool(encoder_out, attention_mask)

    def forward(self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            input_features: (batch, 80, n_frames) log-mel spectrogram, as
                produced by WhisperFeatureExtractor. For our 8-second max
                clip length at 16kHz with Whisper's default hop length of
                160 samples, n_frames = 8*16000/160 = 800.
            attention_mask: optional (batch, n_frames) 1/0 mask marking real
                vs. padding frames (see dataset.py's left-padding scheme).

        Returns:
            (batch,) tensor of raw logits. sigmoid(logits) = P(turn complete).
        """
        # WhisperEncoder ignores attention_mask internally (it always runs
        # full self-attention over the whole spectrogram, including padding),
        # so we pass the mask ONLY to our own pooling layer, which is what
        # actually needs to know which frames are real vs. padding.
        pooled = self.encode_audio(input_features, attention_mask)
        logits = self.classifier(pooled).squeeze(-1)
        return logits

    @property
    def num_parameters(self) -> int:
        """Total trainable + frozen parameter count, for sanity-checking against
        the ~8M-parameter figure Pipecat report for smart-turn-v3."""
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        """Parameter count updated by the optimizer for the chosen freeze policy."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MultimodalTurnDetectionModel(TurnDetectionModel):
    """Whisper audio embedding plus compact token-averaged transcript embedding."""

    def __init__(
        self,
        base_model_name: str = BASE_MODEL_NAME,
        freeze_encoder_layers: int = 0,
        pooling: str = "attention",
        text_vocab_size: int = 51_865,
        text_embedding_dim: int = 64,
        text_pad_token_id: int = 50_257,
    ) -> None:
        super().__init__(base_model_name, freeze_encoder_layers, pooling)
        if text_vocab_size <= 0 or text_embedding_dim <= 0:
            raise ValueError("text_vocab_size and text_embedding_dim must be positive")
        if not 0 <= text_pad_token_id < text_vocab_size:
            raise ValueError("text_pad_token_id must be inside text vocabulary")

        audio_dim = self.encoder.config.d_model
        self.text_embedding = nn.Embedding(
            text_vocab_size, text_embedding_dim, padding_idx=text_pad_token_id
        )
        self.text_pad_token_id = text_pad_token_id
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim + text_embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.1)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        text_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse pooled audio and transcript token embeddings into one logit."""
        if input_ids is None or text_attention_mask is None:
            raise ValueError("multimodal model requires input_ids and text_attention_mask")
        audio_embedding = self.encode_audio(input_features, attention_mask)
        token_embeddings = self.text_embedding(input_ids)
        text_mask = text_attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
        text_embedding = (token_embeddings * text_mask).sum(dim=1) / text_mask.sum(dim=1).clamp(min=1.0)
        fused = torch.cat((audio_embedding, text_embedding), dim=-1)
        return self.classifier(fused).squeeze(-1)


def build_model(
    base_model_name: str = BASE_MODEL_NAME,
    freeze_encoder_layers: int = 0,
    pooling: str = "attention",
    multimodal: bool = False,
    text_vocab_size: int = 51_865,
    text_embedding_dim: int = 64,
    text_pad_token_id: int = 50_257,
) -> TurnDetectionModel:
    """Convenience constructor -- see TurnDetectionModel.__init__ for args."""
    if multimodal:
        return MultimodalTurnDetectionModel(
            base_model_name,
            freeze_encoder_layers,
            pooling,
            text_vocab_size,
            text_embedding_dim,
            text_pad_token_id,
        )
    return TurnDetectionModel(base_model_name, freeze_encoder_layers, pooling)


if __name__ == "__main__":
    # Quick smoke test: build the model in each pooling mode, check its
    # parameter count stays well under the 15M budget (and roughly matches
    # Pipecat's reported ~8M for the "attention" mode, since that's an exact
    # architectural match to smart-turn-v3), and run one dummy batch through
    # it to make sure shapes line up end-to-end before we wire up real data.
    # Smart Turn resizes Whisper to 800 mel frames / 400 hidden positions.
    dummy = torch.randn(2, 80, 800)
    dummy_mask = torch.ones(2, 400)
    for pooling in ("attention", "mean", "last"):
        m = build_model(pooling=pooling)
        out = m(dummy, dummy_mask)
        print(f"pooling={pooling:10s} params={m.num_parameters:,} logits_shape={tuple(out.shape)}")
        assert m.num_parameters < 15_000_000, "over the 15M parameter budget!"
