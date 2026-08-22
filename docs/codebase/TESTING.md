<!-- markdownlint-disable MD013 -->

# Testing and verification

The current suite checks dataset splits, label-preserving augmentation, pause
policy, hard-case indices, padding masks, audio validation, multimodal tensor
shapes, sampler balance, experiment manifests, report rendering, checkpoint
resolution, and transcript-cache mismatch.

```powershell
uv run python -m unittest discover -s tests -v
uvx ruff check src scripts tests app.py
uv lock --check
uv pip check
```

Prepared-data and training smoke tests are separate because they need local data
and may download model assets:

```powershell
uv run python scripts/prepare_data.py --train-scan-budget 2000 --test-scan-budget 500
uv run python src/train.py --config configs/smoke.yaml
```

The suite does not run the full training loop or load the canonical checkpoint
for a real prediction. It also does not exercise `evaluate_checkpoint` end to
end or call the Gradio endpoint. The repository has no coverage gate, type
checker, notebook execution gate, or CI workflow. These are coverage gaps, not
known failures.

Training reloads best checkpoint and verifies calibrated validation F1. Experiment
manifests hash their metadata. Multimodal training rejects augmented audio paired
with cached text. Test evaluation is opt-in.
