"""
Evaluation utilities: classification metrics, latency/size benchmarking, and
slice-based error analysis (specifically on Hinglish fillers + pauses, since
that's the whole point of this project -- an aggregate accuracy number alone
would hide exactly the failure modes we care most about).

This module is deliberately import-only for its core functions
(`evaluate_model`, `measure_latency`, `error_analysis`) so both `train.py`
(runs a quick eval each epoch) and `scripts/run_experiments.py` (runs a full
eval per ablation) share the exact same metric computation -- we never want
two slightly-different accuracy calculations floating around the codebase.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import WhisperFeatureExtractor


@torch.no_grad()
def evaluate_model(model: torch.nn.Module, dataloader: DataLoader, device: str) -> dict[str, Any]:
    """Run `model` over every batch in `dataloader` and compute the metrics
    we track for every experiment (see docs/02_experiment_plan.md):
    accuracy, precision, recall, F1, AUC, and "false complete rate".

    WHY "FALSE COMPLETE RATE" GETS ITS OWN METRIC
    -----------------------------------------------
    In a real voice assistant, the two error directions are NOT equally
    costly: predicting "complete" when the user was actually still talking
    (a false positive on the "complete" class) means the assistant *cuts the
    user off* -- a jarring, visible failure. Predicting "incomplete" when
    they were actually done just means a slightly longer pause before the
    assistant responds -- annoying, but much less disruptive. Plain accuracy
    or F1 can look fine while this specific, worse error type is happening a
    lot, so we report it separately: false_complete_rate =
    P(predict complete | truth is incomplete) = FP / (FP + TN) w.r.t. the
    "complete" class.

    Returns a dict of the model's raw predictions/probabilities alongside
    the computed metrics, so callers (e.g. `error_analysis`) can slice
    correctness by metadata (language, midfiller, etc.) without a second
    forward pass.
    """
    model.eval()
    all_labels, all_probs, all_meta = [], [], []

    for batch in dataloader:
        input_features = batch["input_features"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        model_kwargs = {}
        if "input_ids" in batch:
            model_kwargs = {
                "input_ids": batch["input_ids"].to(device),
                "text_attention_mask": batch["text_attention_mask"].to(device),
            }
        logits = model(input_features, attention_mask, **model_kwargs)
        probs = torch.sigmoid(logits).float().cpu().numpy()

        all_probs.append(probs)
        all_labels.append(batch["labels"].numpy())
        all_meta.extend(batch["meta"])

    if not all_labels:
        raise ValueError("cannot evaluate an empty dataloader")

    labels = np.concatenate(all_labels)
    probs = np.concatenate(all_probs)
    preds = (probs >= 0.5).astype(int)

    # False-complete-rate: among truly-incomplete examples (label==0), what
    # fraction did we wrongly call complete (pred==1)?
    incomplete_mask = labels == 0
    false_complete_rate = (
        float(preds[incomplete_mask].mean()) if incomplete_mask.any() else float("nan")
    )

    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auc": roc_auc_score(labels, probs) if len(set(labels.tolist())) > 1 else float("nan"),
        "false_complete_rate": false_complete_rate,
        "n_examples": len(labels),
    }
    return {"metrics": metrics, "labels": labels, "probs": probs, "preds": preds, "meta": all_meta}


def measure_latency(
    model: torch.nn.Module,
    device: str,
    feature_extractor: WhisperFeatureExtractor,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict[str, Any]:
    """Benchmark single-example inference latency on `device`, plus report
    model size -- both are explicit metrics the brief asks us to track per
    experiment (alongside accuracy/F1/etc.), since the whole point of this
    task is a SMALL, FAST model, not just an accurate one.

    We benchmark with batch size 1 (not our training batch size) because
    that's what actually matters in production: a live voice assistant
    decides turn-completion for one utterance at a time, in the critical
    path of response latency -- there's no batching to amortize cost over.
    """
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from configs import config as cfg

    model.eval().to(device)
    dummy_wave = [np.zeros(cfg.WHISPER_WINDOW_SAMPLES, dtype=np.float32)]
    features = feature_extractor(
        dummy_wave,
        sampling_rate=cfg.SAMPLE_RATE,
        return_tensors="pt",
        padding="max_length",
        max_length=cfg.WHISPER_WINDOW_SAMPLES,
        truncation=True,
        do_normalize=True,
    )
    input_features = features["input_features"].to(device)
    mask = torch.ones(1, cfg.WHISPER_ENCODER_HIDDEN_LEN, device=device)
    model_kwargs = {}
    if hasattr(model, "text_embedding"):
        model_kwargs = {
            "input_ids": torch.zeros((1, 8), dtype=torch.long, device=device),
            "text_attention_mask": torch.ones((1, 8), dtype=torch.long, device=device),
        }

    with torch.no_grad():
        for _ in range(n_warmup):
            model(input_features, mask, **model_kwargs)
        if device == "cuda":
            torch.cuda.synchronize()

        times_ms = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(input_features, mask, **model_kwargs)
            if device == "cuda":
                torch.cuda.synchronize()
            times_ms.append((time.perf_counter() - t0) * 1000)

    times_ms = np.array(times_ms)
    num_params = sum(p.numel() for p in model.parameters())
    # fp32 on-disk/in-memory size; a deployed int8-quantized ONNX export (see
    # export_onnx.py) would be ~4x smaller, matching Pipecat's own
    # fp32-vs-int8 size comparison for smart-turn-v3.
    size_mb = num_params * 4 / 1e6

    return {
        "device": device,
        "latency_ms_mean": float(times_ms.mean()),
        "latency_ms_p50": float(np.percentile(times_ms, 50)),
        "latency_ms_p95": float(np.percentile(times_ms, 95)),
        "num_parameters": num_params,
        "model_size_mb_fp32": size_mb,
    }


def error_analysis(result: dict[str, Any], top_k_errors: int = 15) -> dict[str, Any]:
    """Break down correctness by the metadata slices we actually care about
    for this task -- Hindi-tagged Hinglish proxy rows (language == "hin"), mid-sentence fillers
    (midfiller), trailing fillers (endfiller), and synthetic vs. human audio
    -- plus surface the most confidently-wrong examples for qualitative
    inspection. This is what turns a single accuracy number into the kind of
    error analysis the brief asks for ("especially on Hinglish fillers and
    mid-sentence pauses").
    """
    labels, preds, probs, meta = result["labels"], result["preds"], result["probs"], result["meta"]
    correct = preds == labels

    def _slice_metrics(mask: np.ndarray) -> dict | None:
        if mask.sum() == 0:
            return None
        slice_labels = labels[mask]
        slice_preds = preds[mask]
        slice_probs = probs[mask]
        return {
            "n": int(mask.sum()),
            "n_complete": int((slice_labels == 1).sum()),
            "n_incomplete": int((slice_labels == 0).sum()),
            "accuracy": float(correct[mask].mean()),
            "precision": precision_score(slice_labels, slice_preds, zero_division=0),
            "recall": recall_score(slice_labels, slice_preds, zero_division=0),
            "f1": f1_score(slice_labels, slice_preds, zero_division=0),
            "auc": (
                roc_auc_score(slice_labels, slice_probs)
                if len(set(slice_labels.tolist())) > 1
                else float("nan")
            ),
            "false_complete_rate": (
                float(preds[mask & (labels == 0)].mean()) if (mask & (labels == 0)).any() else float("nan")
            ),
        }

    is_hin = np.array([m["language"] == "hin" for m in meta])
    is_midfiller = np.array([bool(m["midfiller"]) for m in meta])
    is_endfiller = np.array([bool(m["endfiller"]) for m in meta])
    is_synthetic = np.array([bool(m["synthetic"]) for m in meta])
    is_augmented = np.array([bool(m.get("is_augmented", False)) for m in meta])
    has_internal_pause = np.array([bool(m.get("internal_pause", False)) for m in meta])
    has_trailing_pause = np.array([bool(m.get("trailing_pause", False)) for m in meta])
    duration_s = np.array([float(m.get("duration_s", 0.0)) for m in meta])
    is_hard_negative = ((labels == 1) & (duration_s <= 1.5)) | (
        (labels == 0) & (duration_s >= 4.0) & (is_midfiller | is_endfiller)
    )
    is_hard_hinglish_proxy = is_hin & (is_hard_negative | ((is_midfiller | is_endfiller) & has_internal_pause))

    slices = {
        "overall": _slice_metrics(np.ones_like(labels, dtype=bool)),
        "hindi": _slice_metrics(is_hin),
        "midfiller": _slice_metrics(is_midfiller),
        "endfiller": _slice_metrics(is_endfiller),
        "synthetic_audio": _slice_metrics(is_synthetic),
        "human_audio": _slice_metrics(~is_synthetic),
        "our_augmented_examples": _slice_metrics(is_augmented),
        "hindi_midfiller": _slice_metrics(is_hin & is_midfiller),
        "hindi_endfiller": _slice_metrics(is_hin & is_endfiller),
        "internal_pause": _slice_metrics(has_internal_pause),
        "trailing_pause": _slice_metrics(has_trailing_pause),
        "hindi_internal_pause": _slice_metrics(is_hin & has_internal_pause),
        "hindi_filler_pause": _slice_metrics(is_hin & (is_midfiller | is_endfiller) & has_internal_pause),
        "hard_hinglish_proxy": _slice_metrics(is_hard_hinglish_proxy),
        "hard_negatives": _slice_metrics(is_hard_negative),
    }

    # Most confidently wrong: sort wrong predictions by how far probs was
    # from the decision boundary in the WRONG direction -- these are the
    # cases most worth reading transcripts/listening to, since "barely
    # wrong" errors are much less informative than "confidently wrong" ones.
    wrong_idx = np.where(~correct)[0]
    confidence_of_wrong_answer = np.where(labels[wrong_idx] == 1, 1 - probs[wrong_idx], probs[wrong_idx])
    worst_idx = wrong_idx[np.argsort(-confidence_of_wrong_answer)][:top_k_errors]
    worst_examples = [
        {
            "id": meta[i]["id"],
            "language": meta[i]["language"],
            "true_label": int(labels[i]),
            "predicted_prob_complete": float(probs[i]),
            "midfiller": bool(meta[i]["midfiller"]),
            "endfiller": bool(meta[i]["endfiller"]),
            "synthetic": bool(meta[i]["synthetic"]),
        }
        for i in worst_idx
    ]

    return {"slices": slices, "worst_examples": worst_examples}


def evaluate_checkpoint(checkpoint_path: str | Path, metadata_path: str | Path) -> dict[str, Any]:
    """Evaluate one saved checkpoint on a clean metadata split."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import polars as pl
    from torch.utils.data import DataLoader
    from transformers import WhisperFeatureExtractor, WhisperTokenizer

    from configs import config as cfg
    from src.dataset import AugmentConfig, TurnDetectionDataset, collate_fn
    from src.models import build_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    multimodal = bool(config["model"].get("multimodal", False))
    model = build_model(
        base_model_name=config["model"]["base_model_name"],
        freeze_encoder_layers=config["model"]["freeze_encoder_layers"],
        pooling=config["model"]["pooling"],
        multimodal=multimodal,
        text_vocab_size=config["model"].get("text_vocab_size", 51_865),
        text_embedding_dim=config["model"].get("text_embedding_dim", 64),
        text_pad_token_id=config["model"].get("text_pad_token_id", 50_257),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config["model"]["base_model_name"], chunk_length=cfg.MAX_DURATION_S
    )
    tokenizer = (
        WhisperTokenizer.from_pretrained(config["model"]["base_model_name"])
        if multimodal
        else None
    )
    metadata = pl.read_parquet(metadata_path)
    if multimodal and ("transcript" not in metadata.columns or metadata["transcript"].null_count() > 0):
        raise ValueError("multimodal evaluation requires a non-null transcript column")
    dataset = TurnDetectionDataset(
        metadata,
        augment=AugmentConfig(enabled=False),
        include_pause_features=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["eval_batch_size"],
        shuffle=False,
        collate_fn=partial(collate_fn, feature_extractor=feature_extractor, tokenizer=tokenizer),
    )
    result = evaluate_model(model, dataloader, device)
    return {
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "metrics": result["metrics"],
        "error_analysis": error_analysis(result),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    report = evaluate_checkpoint(args.checkpoint, args.metadata)
    rendered = json.dumps(report, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(rendered)
        print(f"[evaluate] wrote {args.output}")
    print(rendered)
