"""Repo-level pytest configuration: register the 'network' marker and skip by default."""

import os

import pytest

# Keras 3 picks its backend at import time and defaults to tensorflow, which
# publishes no cp314 wheel -- so on the ubuntu-26.04 runner (Python 3.14) the
# default backend is not installable. Keras is multi-backend, so point it at
# torch, which does ship cp314. Nothing here imports keras, and pytest imports
# this module before collecting the test modules that do, so this lands early.
os.environ.setdefault("KERAS_BACKEND", "torch")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: requires network access")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("-m") != "network":
        skip_net = pytest.mark.skip(reason="needs -m network")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_net)
