"""Repo-level pytest configuration: register the 'network' marker and skip by default."""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "network: requires network access")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("-m") != "network":
        skip_net = pytest.mark.skip(reason="needs -m network")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_net)
