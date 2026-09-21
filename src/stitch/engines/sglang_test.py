"""SGLang engine request construction and version-stamping tests."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from stitch.engines.base import EngineHealthStatus
from stitch.engines.sglang import SGLangEngine
from stitch.types import VersionKind, VersionManifest, VersionRef


def _manifest(kind: VersionKind = VersionKind.DELTA) -> VersionManifest:
    return VersionManifest(VersionRef("r1", 5), kind, ["weights"])


def _engine(mode: str = "disk") -> SGLangEngine:
    return SGLangEngine(
        "http://engine",
        "/ckpt" if mode == "disk" else None,
        delta_update_mode=mode,  # type: ignore[arg-type]
    )


def test_stamp_request_namespaces_by_version() -> None:
    engine = _engine()
    req: dict = {"text": "hi"}
    engine.stamp_request(req, VersionRef("r1", 7))
    assert req["extra_key"] == "wv7;r1/"
    listed: dict = {"extra_key": ["a", "b"]}
    engine.stamp_request(listed, VersionRef(None, 3))
    assert listed["extra_key"] == ["wv3;a", "wv3;b"]


def test_delta_update_mode_is_validated() -> None:
    with pytest.raises(ValueError, match="delta_update_mode"):
        _engine("memory")


def test_disk_mode_requires_local_checkpoint() -> None:
    with pytest.raises(ValueError, match="requires local_checkpoint_dir"):
        SGLangEngine("http://engine")


def test_cpu_mode_does_not_require_local_checkpoint() -> None:
    _engine("cpu")


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_initialize_verifies_sglang_startup_contract(mode: str) -> None:
    engine = _engine(mode)
    requests: list[str] = []

    async def fake_get(path, *, ok=(200,)):
        del ok
        requests.append(path)
        if path == "/server_info":
            return {
                "weight_update_staging": mode,
                "weight_update_local_checkpoint_dir": (
                    "/ckpt" if mode == "disk" else None
                ),
            }
        return {"weight_version": "119"}

    engine._get_json = fake_get  # type: ignore[method-assign]
    asyncio.run(engine.initialize_update_destination(119))
    assert requests == ["/server_info", "/model_info"]


@pytest.mark.parametrize(
    ("server_info", "model_info", "message"),
    [
        (
            {
                "weight_update_staging": "disk",
                "weight_update_local_checkpoint_dir": None,
            },
            {"weight_version": "3"},
            "staging mode",
        ),
        (
            {
                "weight_update_staging": "cpu",
                "weight_update_local_checkpoint_dir": None,
            },
            {"weight_version": "2"},
            "startup weight version",
        ),
    ],
)
def test_initialize_rejects_mismatched_sglang_contract(
    server_info: dict, model_info: dict, message: str
) -> None:
    engine = _engine("cpu")

    async def fake_get(path, *, ok=(200,)):
        del ok
        return server_info if path == "/server_info" else model_info

    engine._get_json = fake_get  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match=message):
        asyncio.run(engine.initialize_update_destination(3))


def test_initialize_rejects_mismatched_checkpoint_directory() -> None:
    engine = _engine("disk")

    async def fake_get(path, *, ok=(200,)):
        del ok
        if path == "/server_info":
            return {
                "weight_update_staging": "disk",
                "weight_update_local_checkpoint_dir": "/other",
            }
        return {"weight_version": "0"}

    engine._get_json = fake_get  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="checkpoint directory"):
        asyncio.run(engine.initialize_update_destination())


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_stage_prepares_one_target(mode: str) -> None:
    engine = _engine(mode)
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload, timeout, action))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.stage(_manifest(), "/source/weight_v000005"))
    assert requests == [
        (
            "/prepare_weight_update",
            {"checkpoint_source_dir": "/source", "target_version": 5},
            3600.0,
            "weight preparation",
        )
    ]


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_commit_publishes_the_prepared_target(mode: str) -> None:
    engine = _engine(mode)
    requests = []

    async def fake_post(path, payload, *, timeout=None, action=None):
        requests.append((path, payload, timeout, action))

    engine._post = fake_post  # type: ignore[method-assign]
    asyncio.run(engine.commit(_manifest(), flush_cache=False))
    assert requests == [
        (
            "/commit_weight_update",
            {
                "target_version": 5,
                "abort_all_requests": False,
                "torch_empty_cache": False,
                "flush_cache": False,
            },
            600.0,
            "weight commit",
        )
    ]


def test_cpu_mode_rejects_full_checkpoint() -> None:
    engine = _engine("cpu")
    with pytest.raises(ValueError, match="delta manifests only"):
        asyncio.run(engine.stage(_manifest(VersionKind.FULL), "/source/weight_v000005"))


@pytest.mark.parametrize("mode", ["disk", "cpu"])
def test_reset_requires_a_fresh_replica(mode: str) -> None:
    with pytest.raises(RuntimeError, match="fresh rollout replica"):
        asyncio.run(_engine(mode).reset())


def test_stamp_response_generate_vs_openai() -> None:
    engine = _engine()
    gen: dict = {"text": "x", "meta_info": {}}
    engine.stamp_response(gen, VersionRef("r1", 4), VersionRef("r1", 5))
    assert gen["meta_info"] == {
        "weight_version": "4",
        "weight_version_start": 4,
        "weight_version_end": 5,
    }
    openai: dict = {"choices": [{"meta_info": {}}]}
    engine.stamp_response(openai, VersionRef("r1", 4), VersionRef("r1", 4))
    assert openai["weight_version_start"] == 4
    assert openai["weight_version_end"] == 4
    assert openai["choices"][0]["meta_info"] == {
        "weight_version": "4",
        "weight_version_start": 4,
        "weight_version_end": 4,
    }
    assert "meta_info" not in openai and "weight_version" not in openai


class _HealthClient:
    def __init__(self, outcome) -> None:
        self.outcome = outcome
        self.urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        pass

    async def get(self, url: str) -> httpx.Response:
        self.urls.append(url)
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
    client = _HealthClient(outcome)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_kwargs: client)
    engine = _engine()
    assert asyncio.run(engine.check_health()).status is expected
    assert client.urls == ["http://engine/health"]
