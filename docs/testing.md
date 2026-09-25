# Testing

Install development dependencies and use the same commands as CI:

```shell
python -m pip install ".[dev]"
python -m ruff check run_suite2p_in_depth.py src paper tests
python -m ruff format --check run_suite2p_in_depth.py src paper tests
python -m mypy
python -m pytest --cov-fail-under=75
python -m build
python -m twine check dist/*
python -m pip_audit --skip-editable
```

Tests use deterministic synthetic data, mocked Suite2p boundaries, and Matplotlib Agg. They must
not require CUDA, a GUI, laboratory files, home-directory writes, or network access. Unit tests
cover pure numerical behavior and validation; integration tests build temporary Suite2p-like
trees; paper smoke tests assert exact panel names and non-empty images. Branch-aware package and
paper coverage must remain at least 75 percent.

CI runs quality/build/audit on Ubuntu, the full test suite on Ubuntu and Windows with Python 3.12,
synthetic workflow integrations with pinned official Suite2p `v1.1.0`, a source-checkout CLI
smoke test with only `requirements.txt` installed, and a clean-wheel CLI smoke test.
Each job first installs `torch` and `torchvision` from PyTorch's CPU wheel index. Pip caching is
disabled to avoid retaining installation archives and exhausting the hosted runner's disk space.
Documentation checks validate local links, CLI command names, packaged examples, and maintainer
files.

CUDA and representative scientific equivalence require a manual de-identified dataset comparison;
hosted CI is deliberately CPU-only.
