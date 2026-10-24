import asyncio

import pytest

from cookbook.common.hooks import gated_rollout_request_hook


def test_rollout_hook_adds_auth_without_losing_request_policy(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "wk-test")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "ws-test")
    request = {
        "payload": {"messages": []},
        "headers": {"Modal-Session-ID": "session", "Other": "value"},
    }
    asyncio.run(
        gated_rollout_request_hook(
            {"rollout_request_weight_version_mode": "none"}, None, request
        )
    )
    assert request["headers"] == {
        "Modal-Session-ID": "session",
        "Other": "value",
        "Modal-Key": "wk-test",
        "Modal-Secret": "ws-test",
    }
    assert request["payload"]["messages"] == []
    assert request["payload"]["weight_version"] == {
        "min_version": None,
        "exact_version": None,
    }


def test_rollout_hook_rejects_missing_credentials(monkeypatch):
    monkeypatch.delenv("MODAL_PROXY_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_PROXY_TOKEN_SECRET", raising=False)
    with pytest.raises(ValueError, match="Set both"):
        asyncio.run(
            gated_rollout_request_hook(
                {"rollout_request_weight_version_mode": "none"}, None, {"payload": {}}
            )
        )
