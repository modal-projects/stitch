"""SGLangEngine harness: version stamping (pure dict mutation, no HTTP).

The /pull_weights + /update_weights_from_disk control paths hit a real engine, so they
are validated e2e; the request/response stamping is the provable-without-sglang part."""

from __future__ import annotations

import asyncio

from stitch.engines.sglang import SGLangEngine
from stitch.types import VersionKind, VersionManifest, VersionRef


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


def _capture_commit(*, mode="disk", weight_names=None) -> tuple[str, dict]:
    engine = SGLangEngine("http://engine", "/ckpt", weight_update_mode=mode)
    captured: dict = {}

    async def fake_post(path, payload, *, timeout=None, action=None):
        captured["path"], captured["payload"] = path, payload

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.commit(VersionRef("r1", 5), weight_names=weight_names))
    return captured["path"], captured["payload"]


def test_commit_ignores_touched_names_for_dense_reload() -> None:
    path, payload = _capture_commit(weight_names=["a", "b"])
    assert path == "/update_weights_from_disk"
    assert "weight_names" not in payload
    assert payload["weight_version"] == "5"


def test_commit_full_reload_when_no_names() -> None:
    _, payload = _capture_commit(weight_names=None)
    assert "weight_names" not in payload


def test_host_runtime_stage_and_commit_use_prepared_endpoints() -> None:
    engine = SGLangEngine("http://engine", "/ckpt", weight_update_mode="host_runtime")
    calls: list[tuple[str, dict]] = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        calls.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    manifest = VersionManifest(
        ref=VersionRef("r1", 5),
        kind=VersionKind.DELTA,
        files=[],
    )
    asyncio.run(engine.stage(manifest, "/bulletin/r1/weight_v000005"))
    asyncio.run(engine.commit(manifest.ref, flush_cache=True, weight_names=["a"]))
    assert calls[0] == (
        "/pull_weights",
        {
            "local_checkpoint_dir": "/ckpt",
            "source_dir": "/bulletin/r1",
            "target_version": 5,
            "prepare": "runtime",
        },
    )
    assert calls[1] == (
        "/update_weights_from_prepared",
        {"weight_version": "5", "flush_cache": True},
    )


def test_host_runtime_reset_recaptures_base_image() -> None:
    engine = SGLangEngine("http://engine", "/ckpt", weight_update_mode="host_runtime")
    calls: list[tuple[str, dict]] = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        calls.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.reset())
    assert calls[-1][0] == "/update_weights_from_disk"
    assert calls[-1][1]["weight_version"] == "0"
    assert calls[-1][1]["refresh_host_runtime"] is True


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"sglang engine harness: {len(tests)} PASS")
