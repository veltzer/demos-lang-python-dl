# TOFIX

Findings from a code scan on 2026-10-04.

## Medium

- `data/MNIST/raw/train-images-idx3-ubyte:1` - the whole downloaded MNIST dataset (8 files, ~66 MB, both `.gz` and decompressed copies) is committed; it is a download artifact of `src/02_cnns/05_mnist_train.py:15-16`, which uses `root="data"` relative to the current directory, so running the tests from the repo root drops it into the tree. `git rm --cached` the files and point the dataset `root` at a cache outside the repo (e.g. `~/.cache/torchvision`), updating `src/02_cnns/05_mnist_train.md:10` to match.
- `notes/dev_environment.md:47` - tells the reader to `pip install -r requirements.txt`, which does not exist (dependencies are in `pyproject.toml`, installed with `uv sync`), and refers to "the venv note", which is not in `notes/`; fix both references.

## Low

- `notes/dev_environment.md:27-28` - the package list includes `tensorflow` (no Python 3.14 wheel, deliberately not a dependency - see `tests/conftest.py:7-12`) and lists `numpy` twice; align it with `pyproject.toml:54-63`.
- `pyproject.toml:77` - `mypy_path = "src:python:scripts"` names `python/` and `scripts/`, which do not exist in this repo; reduce to `"src"`.
- `tests/conftest.py:20` - network tests are only enabled when `-m` is exactly `"network"`; any other marker expression that selects them (e.g. `-m "network and not slow"`) still skips them. Check whether `"network"` appears in the expression, or use a dedicated `--network` option.
