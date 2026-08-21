<!-- markdownlint-disable MD013 -->

# Repository structure

```text
.
├── app.py              Gradio entry point
├── configs/            Shared constants and experiment YAML files
├── data/samples/       Small labeled demo clips
├── docs/               Reports, tutorial, and codebase notes
├── experiments/        Selected metrics and manifests
├── notebooks/          Exploration and experiment notebooks
├── scripts/            Data and experiment commands
├── src/                Reusable model pipeline
└── tests/              Regression tests
```

`src/dataset.py` owns acquisition, audio normalization, augmentation, splits,
and collation. `src/models.py` owns model architecture. Training and evaluation
live in `src/train.py` and `src/evaluate.py`; `src/inference.py` is the public
prediction API.

Scripts compose those modules and write artifacts. They are meant to run from
the repository root. They add the root to `sys.path`, so this codebase behaves
like a checkout rather than an installed library with console commands.

`configs/config.py` contains values shared by all experiments. YAML files hold
choices that vary, such as pooling, augmentation, and layer freezing. Importing
the Python config creates local data, checkpoint, cache, and ONNX directories.

Labels use `0` for incomplete and `1` for complete. Prepared metadata uses
`endpoint_bool`; the best checkpoint is saved as `<checkpoint.dir>/best.pt`.

## Evidence

- `README.md`
- `configs/config.py` and `configs/baseline.yaml`
- `.gitignore` and `.gitattributes`
- `src/`, `scripts/`, and `tests/`
