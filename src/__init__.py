"""Hinglish turn-detection training, evaluation, and inference package.

Read order for a newcomer: dataset.py (data pipeline + augmentation) ->
models.py (architecture) -> train.py (training loop) -> evaluate.py
(metrics/error analysis) -> inference.py (production-facing wrapper used by
../app.py).
"""
