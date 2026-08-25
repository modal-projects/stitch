"""Import smoke test for the pool app module.

The app module resolves config fields and router helpers at import time, so it
must only reference symbols that exist in the same commit. Importing it here
(with the required env vars set) makes that class of defect fail loudly.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def test_app_module_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPERIMENT_CONFIG", "glm5_2_fp8")
    monkeypatch.setenv("RUN_ID", "import-check")
    sys.modules.pop("cookbook.inference_only.app", None)
    importlib.import_module("cookbook.inference_only.app")
