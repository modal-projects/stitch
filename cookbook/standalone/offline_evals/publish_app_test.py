"""Publisher: index generation, pointer-keyed no-op."""

import os

os.environ.setdefault("EXPERIMENT_CONFIG", "glm5_2_fp8")
os.environ.setdefault("RUN_ID", "publish-app-test")

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cookbook.standalone.offline_evals import publish_app
from stitch.types import VersionRef


def _write_tiny_safetensors(
    path: Path, tensors: tuple[str, ...] = ("w1", "w2")
) -> None:
    """Write a real (tiny) safetensors container: 8-byte LE header length, JSON
    header, then zero-filled tensor data."""
    header: dict[str, dict] = {}
    offset = 0
    for name in tensors:
        nbytes = 4  # one F32 scalar per tensor
        header[name] = {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + b"\x00" * offset)


class _FakeStore:
    """Records call order so the test can prove refresh precedes read_pointer."""

    def __init__(self, pointer: VersionRef | None) -> None:
        self.pointer = pointer
        self.calls: list[str] = []

    def refresh(self) -> None:
        self.calls.append("refresh")

    def read_pointer(self) -> VersionRef | None:
        self.calls.append("read_pointer")
        return self.pointer


# ── Invalid-target cleanup ───────────────────────────────────────────────────


def _make_source(path: Path, shards: tuple[str, ...] = ("model.safetensors",)) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        _write_tiny_safetensors(path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}})
    )
    (path / "config.json").write_text("{}")
    return path


def test_prepare_version_dir(tmp_path: Path) -> None:
    # fresh copy
    source = _make_source(tmp_path / "source")
    target = tmp_path / "weight_v000001"
    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert (target / "model.safetensors.index.json").exists()

    # missing index recopied (a dir with shards but no index was recycled mid-copy)
    target = tmp_path / "weight_v000002"
    target.mkdir()
    _write_tiny_safetensors(target / "model.safetensors")
    (target / "stale-marker").write_text("partial")  # proof the dir was replaced
    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert not (target / "stale-marker").exists()

    # incomplete shard set recopied (index present but a source shard missing)
    source2 = _make_source(
        tmp_path / "source2",
        shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    )
    target = tmp_path / "weight_v000003"
    target.mkdir()
    _write_tiny_safetensors(target / "model-00001-of-00002.safetensors")
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}})
    )
    assert publish_app._prepare_version_dir(target, source2) == "copied"
    assert (target / "model-00002-of-00002.safetensors").exists()

    # valid target kept
    target = _make_source(tmp_path / "weight_v000004")
    (target / "extra-note").write_text("keep me")
    assert publish_app._prepare_version_dir(target, source) == "kept"
    assert (target / "extra-note").exists()


class _FakeSpawn:
    """Stands in for the materialize_version Modal function's .spawn()."""

    def __init__(self, job_id: str = "fc-test-1") -> None:
        self.job_id = job_id
        self.calls: list[dict] = []

    def spawn(self, **kwargs: object) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(object_id=self.job_id)


class _FakeFunctionCall:
    """Stands in for modal.FunctionCall.get(timeout=0) across job states."""

    def __init__(self, state: str, result: dict | None = None) -> None:
        self.state = state
        self.result = result or {}

    def get(self, timeout: float = 0) -> dict:
        if self.state == "pending":
            raise TimeoutError(
                "still running"
            )  # builtins, as modal 1.5.4 actually raises
        if self.state == "failure":
            raise RuntimeError("remote boom")
        return self.result


def _patch_job_poll(
    monkeypatch: pytest.MonkeyPatch, calls: dict[str, _FakeFunctionCall]
) -> None:
    monkeypatch.setattr(
        publish_app, "_function_call_from_id", lambda job_id: calls[job_id]
    )


_server_seq = 0


def _make_server(
    tmp_path: Path,
    pointer: VersionRef | None = None,
    *,
    label: str | None = None,
) -> publish_app.PublisherServer:
    global _server_seq
    if label is None:
        _server_seq += 1
        label = f"run-{_server_seq}"
    run_dir = tmp_path / label
    (run_dir / "updates").mkdir(parents=True, exist_ok=True)
    return publish_app.PublisherServer(
        store=_FakeStore(pointer),
        run_dir=run_dir,
        port=0,
    )


def _client(server: publish_app.PublisherServer) -> TestClient:
    return TestClient(server.build_app())


def _publish(client: TestClient, version: int, source: str, **extra: object):
    return client.post(
        "/publish",
        json={
            "run_id": publish_app.RUN_ID,
            "version": version,
            "source": source,
            **extra,
        },
    )


def _assert_spawned(fake: _FakeSpawn, **expected: object) -> None:
    assert fake.calls == [{"run_id": publish_app.RUN_ID, **expected}]


def test_publisher_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """publish -> job -> status lifecycle: status variants, pointer-keyed no-op,
    spawn/409, and job states."""

    # /status variants: latest_version is the store pointer, staged_versions the
    # weight_v* dirs on disk.
    for pointer, dirs, expected in (
        (
            VersionRef("run", 3),
            ["weight_v000000", "weight_v000003", "not-a-version"],
            {"latest_version": 3, "staged_versions": [0, 3]},
        ),
        (
            VersionRef("run", 0),
            ["weight_v000000", "weight_v000001"],
            {"latest_version": 0, "staged_versions": [0, 1]},
        ),
        (None, [], {"latest_version": None, "staged_versions": []}),
    ):
        server = _make_server(tmp_path, pointer=pointer)
        for name in dirs:
            (server.updates_dir / name).mkdir()
        resp = _client(server).get("/status")
        assert resp.status_code == 200
        assert resp.json() == expected, "latest is the pointer, not a partial dir"

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("materialize job must not spawn on a no-op")

    monkeypatch.setattr(publish_app, "materialize_version", _fail)
    client = _client(_make_server(tmp_path, pointer=VersionRef("run", 5)))
    resp = _publish(client, 3, "/nonexistent")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no-op"
    assert resp.json()["version"] == 3

    # The no-op decision is keyed on the store pointer (refresh precedes read).
    store = _FakeStore(VersionRef("run", 5))
    # older than or equal to the pointer is a no-op; newer publishes
    for version, expected in ((3, True), (5, True), (6, False)):
        assert publish_app._is_already_published(store, version) is expected
    assert store.calls == ["refresh", "read_pointer"] * 3
    assert publish_app._is_already_published(_FakeStore(None), 1) is False, (
        "with no pointer nothing is a no-op"
    )

    client = _client(_make_server(tmp_path))
    resp = _publish(client, 1, "/nonexistent")
    assert resp.status_code == 400
    assert "source directory not found" in resp.json()["detail"]

    source = _make_source(tmp_path / "source")
    fake_spawn = _FakeSpawn(job_id="fc-pub-1")
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path)
    client = _client(server)
    resp = _publish(client, 1, str(source))
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
    assert resp.json()["job_id"] == "fc-pub-1"
    _assert_spawned(fake_spawn, version=1, source=str(source), publish=True)

    monkeypatch.setattr(publish_app, "materialize_version", _FakeSpawn())
    calls = {"fc-test-1": _FakeFunctionCall("pending")}
    _patch_job_poll(monkeypatch, calls)
    client = _client(_make_server(tmp_path))
    resp = _publish(client, 1, str(source))
    assert resp.status_code == 202
    resp = _publish(client, 2, str(source))
    assert resp.status_code == 409
    assert "still running" in resp.json()["detail"]
    calls["fc-test-1"].state = "success"
    resp = _publish(client, 2, str(source))
    assert resp.status_code == 202

    _patch_job_poll(
        monkeypatch,
        {
            "fc-pending": _FakeFunctionCall("pending"),
            "fc-success": _FakeFunctionCall("success", {"version": 1, "path": "/x"}),
            "fc-failure": _FakeFunctionCall("failure"),
        },
    )
    client = _client(_make_server(tmp_path))
    assert client.get("/job/fc-pending").json()["status"] == "pending"
    ok = client.get("/job/fc-success").json()
    assert ok["status"] == "success"
    assert ok["result"] == {"version": 1, "path": "/x"}
    bad = client.get("/job/fc-failure").json()
    assert bad["status"] == "failure"
    assert "remote boom" in bad["error"]
