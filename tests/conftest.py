"""Run every test on default settings, never on the developer's own configuration.

``Settings`` reads ``.env`` and the environment, which on a developer's machine hold their
provider keys, Langfuse project, API key and LLM cache. Tests that picked those up sent traces
and scores to that Langfuse project, and an ``API_AUTH_KEY`` turned API tests into 401s.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from adaptiverag.config import Settings, get_settings


def pytest_configure(config: pytest.Config) -> None:
    """Drop the developer's settings before any test module is imported.

    It runs this early because importing ``adaptiverag.api.main`` already builds an app
    from ``get_settings()``.
    """
    Settings.model_config["env_file"] = None
    setting_names = set(Settings.model_fields)
    for name in [name for name in os.environ if name.lower() in setting_names]:
        del os.environ[name]


@pytest.fixture(autouse=True)
def _fresh_default_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Give each test its own ``get_settings()`` and its own LLM cache file.

    The default cache path is relative to the working directory, so from the repository it is
    the developer's own cache. Tests that pass ``llm_cache_path`` still get theirs.
    """
    monkeypatch.setenv("LLM_CACHE_PATH", str(tmp_path / "llm_cache.sqlite"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
