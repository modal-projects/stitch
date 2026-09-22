from __future__ import annotations

import pytest

from cookbook.common import smoke


class _Pool:
    def __init__(self, app_name: str, cls_name: str) -> None:
        pass

    def gateway_url(self) -> str:
        return "https://pool"


def test_unclaimed_pool_cannot_satisfy_nonzero_version(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "ModalFlashPool", _Pool)
    monkeypatch.setattr(smoke, "_get_json", lambda *args, **kwargs: {"run_id": None})

    with pytest.raises(TimeoutError, match="pool is unclaimed"):
        smoke.smoke_flash_pool(
            app_name="app",
            cls_name="Server",
            model_name="model",
            weight_version=10,
            timeout_seconds=0,
        )


def test_unclaimed_pool_can_serve_base(monkeypatch) -> None:
    monkeypatch.setattr(smoke, "ModalFlashPool", _Pool)
    monkeypatch.setattr(smoke, "_get_json", lambda *args, **kwargs: {"run_id": None})
    monkeypatch.setattr(
        smoke,
        "_post_json",
        lambda *args, **kwargs: {
            "choices": [{"message": {"content": "OK"}}],
        },
    )

    smoke.smoke_flash_pool(
        app_name="app",
        cls_name="Server",
        model_name="model",
        weight_version=0,
        timeout_seconds=0,
    )
