"""SGLang engine request construction and version-stamping tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from stitch.engines.base import EngineHealthStatus
from stitch.engines.sglang import SGLangEngine
from stitch.types import VersionKind, VersionManifest, VersionRef


def _manifest(kind: VersionKind) -> VersionManifest:
    return VersionManifest(VersionRef("r1", 5), kind, ["weights"])


def test_stamp_request_namespaces_by_version() -> None:
    engine = SGLangEngine("http://engine", "/base", "/ckpt")
    req: dict = {"text": "hi"}
    engine.stamp_request(req, VersionRef("r1", 7))
    assert req["extra_key"] == "wv7;r1/"  # version + run namespace, no user key
    listed: dict = {"extra_key": ["a", "b"]}
    engine.stamp_request(listed, VersionRef(None, 3))
    assert listed["extra_key"] == ["wv3;a", "wv3;b"]  # run-less, per-element


def test_delta_update_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="delta_update_mode"):
        SGLangEngine(
            "http://engine",
            "/base",
            "/ckpt",
            delta_update_mode="memory",  # type: ignore[arg-type]
        )


def test_disk_mode_requires_local_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires local_checkpoint_dir"):
        SGLangEngine("http://engine", "/base", None)


def test_cpu_mode_does_not_require_local_checkpoint() -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        None,
        delta_update_mode="cpu",
    )
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.initialize_update_destination())
    assert requests == [
        (
            "/stage_weight_update",
            {
                "base_checkpoint_dir": "/base",
                "base_version": 0,
                "target_version": 0,
                "destination": "cpu",
            },
        )
    ]


def test_cpu_mode_reset_requires_a_fresh_replica() -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        None,
        delta_update_mode="cpu",
    )
    with pytest.raises(RuntimeError, match="fresh rollout replica"):
        asyncio.run(engine.reset())


def test_disk_mode_reset_stages_and_loads_base() -> None:
    engine = SGLangEngine("http://engine", "/base", "/ckpt")
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.reset())
    assert requests == [
        (
            "/stage_weight_update",
            {
                "base_checkpoint_dir": "/base",
                "base_version": 0,
                "target_version": 0,
                "destination": "disk",
                "local_checkpoint_dir": "/ckpt",
            },
        ),
        (
            "/update_weights_from_disk",
            {
                "model_path": "/ckpt",
                "load_format": "auto",
                "weight_version": "0",
                "flush_cache": False,
            },
        ),
    ]


def test_stamp_response_generate_vs_openai() -> None:
    engine = SGLangEngine("http://engine", "/base", "/ckpt")
    gen: dict = {"text": "x", "meta_info": {}}
    engine.stamp_response(gen, VersionRef("r1", 4), VersionRef("r1", 5))
    assert gen["meta_info"] == {
        "weight_version": "4",
        "weight_version_start": 4,
        "weight_version_end": 5,
    }
    openai: dict = {"choices": []}
    engine.stamp_response(openai, VersionRef("r1", 4), VersionRef("r1", 4))
    assert openai["weight_version_start"] == 4 and openai["weight_version_end"] == 4
    assert "meta_info" not in openai and "weight_version" not in openai


def _commit_request(
    *,
    kind: VersionKind,
    delta_update_mode: str = "disk",
    disk_load_format: str = "auto",
) -> tuple[str, dict]:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        "/ckpt",
        delta_update_mode=delta_update_mode,
        disk_load_format=disk_load_format,
    )
    captured: dict = {}

    async def fake_post(path, payload, *, timeout=None, action=None):
        captured["path"], captured["payload"] = path, payload

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.commit(_manifest(kind)))
    return captured["path"], captured["payload"]


def test_disk_delta_commit_uses_checkpoint_loader() -> None:
    path, payload = _commit_request(
        kind=VersionKind.DELTA,
        disk_load_format="fastsafetensors",
    )
    assert path == "/update_weights_from_disk"
    assert payload == {
        "model_path": "/ckpt",
        "load_format": "fastsafetensors",
        "weight_version": "5",
        "flush_cache": False,
    }


def test_cpu_delta_commit_uses_host_image() -> None:
    path, payload = _commit_request(
        kind=VersionKind.DELTA,
        delta_update_mode="cpu",
    )
    assert path == "/update_weights_from_cpu"
    assert payload == {"target_version": 5, "flush_cache": False}


def test_full_checkpoint_is_never_loaded_from_cpu() -> None:
    with pytest.raises(ValueError, match="delta manifests only"):
        _commit_request(
            kind=VersionKind.FULL,
            delta_update_mode="cpu",
        )


def test_cpu_mode_stages_deltas_in_cpu() -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        "/ckpt",
        delta_update_mode="cpu",
    )
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.stage(_manifest(VersionKind.DELTA), "/source/weight_v000005"))
    assert requests == [
        (
            "/stage_weight_update",
            {
                "base_checkpoint_dir": "/base",
                "base_version": 0,
                "checkpoint_source_dir": "/source",
                "target_version": 5,
                "destination": "cpu",
            },
        )
    ]


def test_cpu_mode_rejects_full_checkpoint_staging() -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        "/ckpt",
        delta_update_mode="cpu",
    )
    with pytest.raises(ValueError, match="delta manifests only"):
        asyncio.run(
            engine.stage(
                _manifest(VersionKind.FULL),
                "/source/weight_v000005",
            )
        )


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_initialize_update_destination(mode: str) -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        "/ckpt",
        delta_update_mode=mode,
    )
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.initialize_update_destination())
    expected = {
        "base_checkpoint_dir": "/base",
        "base_version": 0,
        "target_version": 0,
        "destination": mode,
    }
    if mode == "disk":
        expected["local_checkpoint_dir"] = "/ckpt"
    assert requests == [("/stage_weight_update", expected)]


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_resumed_destination_preserves_boot_version(mode: str) -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base-v119",
        "/ckpt",
        delta_update_mode=mode,
    )
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.initialize_update_destination(119))
    expected = {
        "base_checkpoint_dir": "/base-v119",
        "base_version": 119,
        "target_version": 119,
        "destination": mode,
    }
    if mode == "disk":
        expected["local_checkpoint_dir"] = "/ckpt"
    assert requests == [("/stage_weight_update", expected)]

    requests.clear()
    resumed_delta = VersionManifest(
        VersionRef("r1", 120), VersionKind.DELTA, ["weights"]
    )
    asyncio.run(engine.stage(resumed_delta, "/source/weight_v000120"))
    staged = {
        "base_checkpoint_dir": "/base-v119",
        "base_version": 119,
        "checkpoint_source_dir": "/source",
        "target_version": 120,
        "destination": mode,
    }
    if mode == "disk":
        staged["local_checkpoint_dir"] = "/ckpt"
    assert requests == [("/stage_weight_update", staged)]


def test_staging_and_commit_have_independent_timeouts() -> None:
    engine = SGLangEngine(
        "http://engine",
        "/base",
        "/ckpt",
        weight_staging_timeout=3600.0,
        weight_update_timeout=600.0,
    )
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, timeout))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.stage(_manifest(VersionKind.DELTA), "/source/weight_v000005"))
    asyncio.run(engine.commit(_manifest(VersionKind.DELTA)))
    assert requests == [
        ("/stage_weight_update", 3600.0),
        ("/update_weights_from_disk", 600.0),
    ]


class _HealthClient:
    def __init__(self, outcome) -> None:
        self.outcome = outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        pass

    async def get(self, url: str) -> httpx.Response:
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return httpx.Response(self.outcome, request=httpx.Request("GET", url))


@pytest.mark.parametrize(
    "outcome,expected",
    [
        (200, EngineHealthStatus.HEALTHY),
        (503, EngineHealthStatus.UNRESPONSIVE),
        (
            httpx.ReadTimeout("busy", request=httpx.Request("GET", "http://engine")),
            EngineHealthStatus.UNRESPONSIVE,
        ),
        (
            httpx.ConnectError(
                "connection refused",
                request=httpx.Request("GET", "http://engine"),
            ),
            EngineHealthStatus.UNREACHABLE,
        ),
    ],
)
def test_health_check_classifies_engine_failures(
    monkeypatch, outcome, expected
) -> None:
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **_kwargs: _HealthClient(outcome),
    )
    engine = SGLangEngine("http://engine", "/base", "/ckpt")
    assert asyncio.run(engine.check_health()).status is expected


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"sglang engine harness: {len(tests)} PASS")
