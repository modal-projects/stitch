"""SGLangEngine harness: version stamping (pure dict mutation, no HTTP).

The /pull_weights + /update_weights_from_disk control paths hit a real engine, so they
are validated e2e; the request/response stamping is the provable-without-sglang part."""

from __future__ import annotations

import asyncio
import os

from stitch.engines.sglang import SGLangEngine
from stitch.types import VersionRef


def test_stamp_request_namespaces_by_version() -> None:
    engine = SGLangEngine("http://engine", "/ckpt")
    req: dict = {"text": "hi"}
    engine.stamp_request(req, VersionRef("r1", 7))
    assert req["extra_key"] == "wv7;r1/"  # version + run namespace, no user key
    listed: dict = {"extra_key": ["a", "b"]}
    engine.stamp_request(listed, VersionRef(None, 3))
    assert listed["extra_key"] == ["wv3;a", "wv3;b"]  # run-less, per-element


def test_stamp_response_generate_vs_openai() -> None:
    engine = SGLangEngine("http://engine", "/ckpt")
    gen: dict = {"text": "x", "meta_info": {}}
    engine.stamp_response(gen, VersionRef("r1", 4), VersionRef("r1", 5))
    assert gen["meta_info"] == {"weight_version": "4", "weight_version_start": 4, "weight_version_end": 5}
    openai: dict = {"choices": []}
    engine.stamp_response(openai, VersionRef("r1", 4), VersionRef("r1", 4))
    assert openai["weight_version_start"] == 4 and openai["weight_version_end"] == 4
    assert "meta_info" not in openai and "weight_version" not in openai


def _commit_payload(*, weight_names=None, partial_reload=None) -> dict:
    """Build the /update_weights_from_disk payload commit() would POST, without HTTP."""
    engine = SGLangEngine("http://engine", "/ckpt")
    captured: dict = {}

    async def fake_post(path, payload, *, timeout=None, action=None):
        captured["path"], captured["payload"] = path, payload

    engine._post = fake_post  # type: ignore[method-assign]
    prior = os.environ.get("STITCH_PARTIAL_RELOAD")
    if partial_reload is not None:
        os.environ["STITCH_PARTIAL_RELOAD"] = partial_reload
    try:
        asyncio.run(engine.commit(VersionRef("r1", 5), weight_names=weight_names))
    finally:
        if partial_reload is not None:
            if prior is None:
                del os.environ["STITCH_PARTIAL_RELOAD"]
            else:
                os.environ["STITCH_PARTIAL_RELOAD"] = prior
    assert captured["path"] == "/update_weights_from_disk"
    return captured["payload"]


def test_commit_names_touched_tensors_for_partial_reload() -> None:
    payload = _commit_payload(weight_names=["a", "b"])
    assert payload["weight_names"] == ["a", "b"]  # O(delta): the fork reloads only these
    assert payload["weight_version"] == "5"


def test_commit_full_reload_when_no_names() -> None:
    assert "weight_names" not in _commit_payload(weight_names=None)  # full reload names nothing


def test_commit_kill_switch_forces_full_reload() -> None:
    # STITCH_PARTIAL_RELOAD=0 drops the names even when the reconciler supplies them.
    assert "weight_names" not in _commit_payload(weight_names=["a", "b"], partial_reload="0")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"sglang engine harness: {len(tests)} PASS")
