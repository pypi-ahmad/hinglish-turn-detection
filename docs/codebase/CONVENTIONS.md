<!-- markdownlint-disable MD013 -->

# Development conventions

Reusable behavior belongs in `src/`; scripts should only coordinate it and
write artifacts. Physical constants and dataset IDs belong in
`configs/config.py`. Experiment choices belong in YAML.

Augmentation must preserve the publisher endpoint label. Filler or silence
presence is not a replacement label. Training may use augmentation, but
validation and test data stay clean. When calibration is enabled, checkpoints
maximize validation F1 under configured FCR/recall constraints. False-complete
rate is reported because interruption is the expensive error.

Public functions use type hints and docstrings. Paths use `pathlib.Path`.
Boundary validation raises explicit exceptions rather than assertions. Model
forward methods return raw logits; callers apply sigmoid and choose a threshold.

Tests use `unittest`, temporary directories, and mocks for expensive external
calls. Run them with:

```powershell
uv run python -m unittest discover -s tests -v
```

Documentation keeps publisher-wide facts separate from local-subset
measurements and proxy slices. Generated reports go under `docs/generated/`;
the numbered documents contain the edited interpretation.

## Change checklist

1. Find the owning module and its tests.
2. State the observable contract.
3. Make the smallest complete edit.
4. Add a regression test when behavior changes.
5. Run focused tests, the full suite, Ruff, and link checks.
6. Update machine artifacts before changing claims based on them.
