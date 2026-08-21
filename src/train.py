"""
Training script: fine-tune TurnDetectionModel (see models.py) on the
Hinglish-focused turn-detection data (see dataset.py).

USAGE
-----
    python src/train.py --config configs/baseline.yaml \\
        --train-meta data/subset/train_split.parquet \\
        --val-meta data/subset/val_split.parquet

Each run is one entry in the ablation matrix (docs/02_experiment_plan.md):
point `--config` at a different YAML to get a
different pooling strategy, freeze setting, or augmentation mix, and results
land in `experiments/<experiment_name>/` -- see scripts/run_experiments.py
for the harness that runs several configs back to back.

WHY A CUSTOM LOOP INSTEAD OF `transformers.Trainer`
--------------------------------------------------------
`Trainer` is built around HF `Dataset`/`datasets.Dataset` objects and
`PreTrainedModel` subclasses with a specific forward signature/output
convention. Our `TurnDetectionModel` is a small hand-built nn.Module (not a
`PreTrainedModel`) and our data pipeline already does its own
augmentation/collation (see dataset.py). Wrapping that in `Trainer` would
mean fighting its assumptions for a training loop that's, in the end, only
~40 lines -- not worth the extra abstraction layer for a model this small.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import yaml
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    get_cosine_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs import config as cfg
from src.dataset import (
    AugmentConfig,
    FillerBank,
    TurnDetectionDataset,
    build_hard_negative_indices,
    collate_fn,
)
from src.evaluate import (
    classification_metrics,
    evaluate_model,
    measure_latency,
    select_operating_threshold,
)
from src.models import build_model


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML experiment config and require a mapping root."""
    with open(path, encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    return config


def _class_balanced_sample_weights(labels: np.ndarray, hard_indices: np.ndarray, hard_weight: float) -> np.ndarray:
    """Boost hard rows while keeping total sampler mass equal by class."""
    if hard_weight <= 0:
        raise ValueError("hard_negative_oversample_weight must be positive")
    labels = np.asarray(labels, dtype=bool)
    if labels.size == 0 or np.unique(labels).size != 2:
        raise ValueError("training metadata must contain both endpoint classes")

    weights = np.ones(labels.size, dtype=np.float64)
    weights[hard_indices] = hard_weight
    for label in (False, True):
        class_mask = labels == label
        weights[class_mask] /= weights[class_mask].sum()
    return weights


def build_dataloaders(
    config: dict[str, Any],
    train_meta_path: str | Path,
    val_meta_path: str | Path,
    feature_extractor: WhisperFeatureExtractor,
    tokenizer: WhisperTokenizer | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Build balanced training loader and clean validation loader."""
    train_meta = pl.read_parquet(train_meta_path)
    val_meta = pl.read_parquet(val_meta_path)
    if tokenizer is not None:
        for split_name, metadata in (("train", train_meta), ("validation", val_meta)):
            if "transcript" not in metadata.columns or metadata["transcript"].null_count() > 0:
                raise ValueError(f"{split_name} metadata requires a non-null transcript column")

    filler_bank = None
    if config["data"]["use_augmentation"] and config["data"]["augment_config"].get("p_filler", 0) > 0:
        filler_bank = FillerBank(cache_dir=cfg.DATA_DIR / "fillers")
        print("[train] synthesizing/checking Hinglish filler-word TTS cache (one-time, cached to disk)...")
        filler_bank.build()

    augment_cfg = AugmentConfig(enabled=config["data"]["use_augmentation"], **config["data"]["augment_config"])
    # Validation is ALWAYS evaluated on clean, unaugmented audio -- augmenting
    # the validation set would let us "validate" against a moving target and
    # make epoch-to-epoch comparisons meaningless.
    train_ds = TurnDetectionDataset(train_meta, augment=augment_cfg, filler_bank=filler_bank)
    val_ds = TurnDetectionDataset(val_meta, augment=AugmentConfig(enabled=False))

    hard_idx = np.array([], dtype=np.int64)
    if config["data"]["use_hard_negatives"]:
        hard_idx = build_hard_negative_indices(train_meta)
        print(f"[train] {len(hard_idx)}/{len(train_ds)} rows flagged as hard negatives (oversample weight="
              f"{config['data']['hard_negative_oversample_weight']})")
    sample_weights = _class_balanced_sample_weights(
        train_meta["endpoint_bool"].to_numpy(), hard_idx, config["data"]["hard_negative_oversample_weight"]
    )
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)

    collate = partial(collate_fn, feature_extractor=feature_extractor, tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds, batch_size=config["training"]["batch_size"], sampler=sampler, collate_fn=collate, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["training"]["eval_batch_size"], shuffle=False, collate_fn=collate, num_workers=0
    )
    return train_loader, val_loader


def train(
    config: dict[str, Any], train_meta_path: str | Path, val_meta_path: str | Path
) -> dict[str, Any]:
    """Train one configured experiment and return metrics for best checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        config["model"]["base_model_name"], chunk_length=cfg.MAX_DURATION_S
    )
    multimodal = bool(config["model"].get("multimodal", False))
    if multimodal and config["data"]["use_augmentation"]:
        raise ValueError("multimodal training requires unaugmented audio matching cached transcripts")
    tokenizer = (
        WhisperTokenizer.from_pretrained(config["model"]["base_model_name"])
        if multimodal
        else None
    )
    if tokenizer is not None:
        config["model"]["text_vocab_size"] = len(tokenizer)
        config["model"]["text_pad_token_id"] = tokenizer.pad_token_id
    train_loader, val_loader = build_dataloaders(
        config, train_meta_path, val_meta_path, feature_extractor, tokenizer
    )

    model = build_model(
        base_model_name=config["model"]["base_model_name"],
        freeze_encoder_layers=config["model"]["freeze_encoder_layers"],
        pooling=config["model"]["pooling"],
        multimodal=multimodal,
        text_vocab_size=config["model"].get("text_vocab_size", 51_865),
        text_embedding_dim=config["model"].get("text_embedding_dim", 64),
        text_pad_token_id=config["model"].get("text_pad_token_id", 50_257),
    ).to(device)

    max_parameters = 15_000_000
    if model.num_parameters >= max_parameters:
        raise ValueError(
            f"model has {model.num_parameters:,} parameters; requirement is under {max_parameters:,}"
        )

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / config["training"]["grad_accum_steps"]))
    total_steps = steps_per_epoch * config["training"]["num_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config["training"]["warmup_ratio"]),
        num_training_steps=total_steps,
    )
    loss_fn = BCEWithLogitsLoss()

    use_amp = config["training"]["mixed_precision"] == "fp16" and device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    exp_dir = Path("experiments") / config["experiment_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    ckpt_dir = Path(config["checkpoint"]["dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_selection_key: tuple[float, ...] | None = None
    grad_accum = config["training"]["grad_accum_steps"]
    calibration_config = config.get("evaluation", {}).get("threshold_calibration", {})
    calibration_enabled = bool(calibration_config.get("enabled", False))
    max_fcr = float(calibration_config.get("max_false_complete_rate", 0.10))
    min_recall = float(calibration_config.get("min_recall", 0.85))

    print(
        f"[train] device={device} amp={use_amp} model_params={model.num_parameters:,} "
        f"trainable_params={model.num_trainable_parameters:,} "
        f"train_batches={len(train_loader)} val_batches={len(val_loader)}"
    )

    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        epoch_t0 = time.time()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{config['training']['num_epochs']}", unit="batch")
        for step, batch in enumerate(pbar):
            input_features = batch["input_features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            model_kwargs = {}
            if multimodal:
                model_kwargs = {
                    "input_ids": batch["input_ids"].to(device),
                    "text_attention_mask": batch["text_attention_mask"].to(device),
                }

            # The final accumulation group may contain fewer micro-batches.
            # Divide by its actual size so that update has the same scale as
            # every full group.
            group_start = (step // grad_accum) * grad_accum
            accumulation_size = min(grad_accum, len(train_loader) - group_start)
            with torch.amp.autocast(device, enabled=use_amp):
                logits = model(input_features, attention_mask, **model_kwargs)
                unscaled_loss = loss_fn(logits, labels)
                loss = unscaled_loss / accumulation_size

            scaler.scale(loss).backward()
            running_loss += unscaled_loss.item()
            pbar.set_postfix(loss=f"{unscaled_loss.item():.4f}")

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                # Unscale before clipping so the max-norm is measured on the
                # true gradient scale, not the fp16 loss-scaled one.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["grad_clip_norm"])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        val_result = evaluate_model(model, val_loader, device)
        val_metrics = val_result["metrics"]
        calibration = (
            select_operating_threshold(
                val_result["labels"],
                val_result["probs"],
                max_false_complete_rate=max_fcr,
                min_recall=min_recall,
            )
            if calibration_enabled
            else {
                "threshold": 0.5,
                "feasible": True,
                "selection_reason": "fixed_threshold",
                "constraints": None,
                "metrics": val_metrics,
            }
        )
        calibrated_metrics = calibration["metrics"]
        epoch_summary = {
            "epoch": epoch + 1,
            "train_loss": running_loss / len(train_loader),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"calibrated_val_{k}": v for k, v in calibrated_metrics.items()},
            "threshold_feasible": calibration["feasible"],
            "elapsed_s": time.time() - epoch_t0,
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(epoch_summary)
        print(
            f"[epoch {epoch + 1}/{config['training']['num_epochs']}] "
            f"loss={epoch_summary['train_loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_false_complete_rate={val_metrics['false_complete_rate']:.4f} "
            f"threshold={calibration['threshold']:.4f} calibrated_fcr="
            f"{calibrated_metrics['false_complete_rate']:.4f} feasible={calibration['feasible']} "
            f"({epoch_summary['elapsed_s']:.0f}s)"
        )

        selection_key = (
            float(calibration["feasible"]),
            float(calibrated_metrics["f1"]),
            -float(calibrated_metrics["false_complete_rate"]),
            float(calibration["threshold"]),
        )
        if best_selection_key is None or selection_key > best_selection_key:
            best_selection_key = selection_key
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                    "val_metrics_fixed_0_5": val_metrics,
                    "calibrated_val_metrics": calibrated_metrics,
                    "decision_threshold": calibration["threshold"],
                    "threshold_calibration": calibration,
                    "num_parameters": model.num_parameters,
                    "num_trainable_parameters": model.num_trainable_parameters,
                },
                ckpt_dir / "best.pt",
            )
            print(
                f"[train] new best calibrated_val_f1={calibrated_metrics['f1']:.4f} "
                f"threshold={calibration['threshold']:.4f} -> saved {ckpt_dir / 'best.pt'}"
            )

    (exp_dir / "history.json").write_text(json.dumps(history, indent=2))

    # Final report: reload the BEST checkpoint (not necessarily the last
    # epoch's weights) for the latency benchmark and the metrics we actually
    # report for this experiment, since "best by val F1" is our selection
    # criterion, including validation safety constraints when enabled.
    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])
    final_result = evaluate_model(model, val_loader, device, best_ckpt.get("decision_threshold", 0.5))
    final_val_metrics = final_result["metrics"]
    final_fixed_metrics = classification_metrics(final_result["labels"], final_result["probs"], 0.5)
    saved_calibrated = best_ckpt.get("calibrated_val_metrics", best_ckpt["val_metrics"])
    if not math.isclose(final_val_metrics["f1"], saved_calibrated["f1"], abs_tol=1e-9):
        raise RuntimeError("reloaded best checkpoint calibrated validation F1 does not match saved metric")
    print(
        f"[final validation] accuracy={final_val_metrics['accuracy']:.4f} "
        f"f1={final_val_metrics['f1']:.4f} auc={final_val_metrics['auc']:.4f}"
    )
    latency_gpu = measure_latency(model, "cuda", feature_extractor) if torch.cuda.is_available() else None
    latency_cpu = measure_latency(model, "cpu", feature_extractor)

    final_report = {
        "experiment_name": config["experiment_name"],
        "config": config,
        "best_val_metrics": final_val_metrics,
        "best_val_metrics_fixed_0_5": final_fixed_metrics,
        "decision_threshold": best_ckpt.get("decision_threshold", 0.5),
        "threshold_calibration": best_ckpt.get("threshold_calibration"),
        "latency_gpu": latency_gpu,
        "latency_cpu": latency_cpu,
        "history": history,
    }
    (exp_dir / "metrics.json").write_text(json.dumps(final_report, indent=2, default=str))
    print(f"[train] wrote {exp_dir / 'metrics.json'}")
    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--train-meta", default=str(cfg.SUBSET_DIR / "train_split.parquet"))
    parser.add_argument("--val-meta", default=str(cfg.SUBSET_DIR / "val_split.parquet"))
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, args.train_meta, args.val_meta)
