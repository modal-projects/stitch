"""Publisher: index generation, pointer-keyed no-op."""

import os

os.environ.setdefault("EXPERIMENT_CONFIG", "qwen3_0p6b_poc")
os.environ.setdefault("RUN_ID", "publish-app-test")

import json
import struct
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from modal.exception import ConnectionError as ModalConnectionError
from modal.exception import TimeoutError as ModalTimeoutError

from cookbook.inference_only import publish_app
from stitch.types import VersionRef


def _write_tiny_safetensors(path: Path, tensors: tuple[str, ...] = ("w1", "w2")) -> None:
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


def test_index_generation(tmp_path: Path) -> None:
    """Index is generated from real shards, an existing index gets its version
    rewritten, and missing/corrupt inputs raise loudly."""
    _write_tiny_safetensors(tmp_path / "model.safetensors")

    publish_app._ensure_safetensors_index(tmp_path, 7)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["version"] == 7
    assert index["weight_map"] == {"w1": "model.safetensors", "w2": "model.safetensors"}

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"version": 1},
                "weight_map": {"w1": "model.safetensors", "w2": "model.safetensors"},
            }
        )
    )

    publish_app._ensure_safetensors_index(tmp_path, 3)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["version"] == 3, "existing index gets the version rewritten"
    assert set(index["weight_map"]) == {"w1", "w2"}

    with pytest.raises(FileNotFoundError, match="no safetensors files"):
        publish_app._ensure_safetensors_index(tmp_path / "empty", 1)

    (tmp_path / "model.safetensors.index.json").write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        publish_app._ensure_safetensors_index(tmp_path, 1)


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


def test_prepare_version_dir_copies_fresh(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "source")
    target = tmp_path / "weight_v000001"

    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert (target / "model.safetensors").exists()
    assert (target / "model.safetensors.index.json").exists()


def test_prepare_version_dir_removes_partial_missing_index(tmp_path: Path) -> None:
    """A dir with shards but no index (recycled mid-copy) is removed and recopied."""
    source = _make_source(tmp_path / "source")
    target = tmp_path / "weight_v000001"
    target.mkdir()
    _write_tiny_safetensors(target / "model.safetensors")
    (target / "stale-marker").write_text("partial")  # proof the dir was replaced

    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert (target / "model.safetensors.index.json").exists()
    assert not (target / "stale-marker").exists()


def test_prepare_version_dir_removes_incomplete_shard_set(tmp_path: Path) -> None:
    """Index present but a source shard missing -> invalid -> recopied."""
    source = _make_source(
        tmp_path / "source", shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
    )
    target = tmp_path / "weight_v000001"
    target.mkdir()
    _write_tiny_safetensors(target / "model-00001-of-00002.safetensors")
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}})
    )

    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert (target / "model-00002-of-00002.safetensors").exists()


def test_prepare_version_dir_keeps_valid_target(tmp_path: Path) -> None:
    source = _make_source(tmp_path / "source")
    target = _make_source(tmp_path / "weight_v000001")
    (target / "extra-note").write_text("keep me")

    assert publish_app._prepare_version_dir(target, source) == "kept"
    assert (target / "extra-note").exists()


# ── Job spawn / polling fakes ────────────────────────────────────────────────


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
            raise TimeoutError("still running")  # builtins, as modal 1.5.4 actually raises
        if self.state == "failure":
            raise RuntimeError("remote boom")
        if self.state == "query-error":
            raise ModalConnectionError("transient poll error")
        if self.state == "modal-timeout":
            raise ModalTimeoutError("still running")
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
    volume_reloader: Callable[[], None] | None = None,
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
        # Unit tests must never invoke the real run_volume.reload() (needs Modal auth).
        volume_reloader=volume_reloader or (lambda: None),
    )


def _make_publish_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    _write_tiny_safetensors(source / "model.safetensors")
    return source


# ── Server-path volume reload ────────────────────────────────────────────────


def _make_recording_server(
    tmp_path: Path,
    *,
    label: str,
    staged_on_reload: tuple[int, ...] = (),
) -> tuple[publish_app.PublisherServer, _FakeStore]:
    """Server whose reloader records order and exposes dirs staged elsewhere."""
    run_dir = tmp_path / label
    (run_dir / "updates").mkdir(parents=True, exist_ok=True)
    store = _FakeStore(None)

    def reloader() -> None:
        store.calls.append("reload")
        for version in staged_on_reload:
            (run_dir / "updates" / f"weight_v{version:06d}").mkdir(exist_ok=True)

    server = publish_app.PublisherServer(
        store=store,
        run_dir=run_dir,
        port=0,
        volume_reloader=reloader,
    )
    return server, store


def test_publisher_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """publish -> job -> status lifecycle: status variants, pointer-keyed no-op,
    spawn/409, job states, volume-reload ordering, and server start/stop."""
    from fastapi.testclient import TestClient

    server = _make_server(tmp_path, pointer=VersionRef("run", 3))
    (server.updates_dir / "weight_v000000").mkdir()
    (server.updates_dir / "weight_v000003").mkdir()
    (server.updates_dir / "not-a-version").mkdir()
    client = TestClient(server.build_app())
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json() == {"latest_version": 3, "staged_versions": [0, 3]}

    server = _make_server(tmp_path, pointer=VersionRef("run", 0))
    (server.updates_dir / "weight_v000000").mkdir()
    (server.updates_dir / "weight_v000001").mkdir()
    client = TestClient(server.build_app())
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json() == {"latest_version": 0, "staged_versions": [0, 1]}, (
        "latest is the pointer, not a partial dir"
    )

    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.get("/status")
    assert resp.json() == {"latest_version": None, "staged_versions": []}

    def _fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("materialize job must not spawn on a no-op")

    monkeypatch.setattr(publish_app, "materialize_version", _fail)
    client = TestClient(_make_server(tmp_path, pointer=VersionRef("run", 5)).build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 3, "source": "/nonexistent"},
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "no-op"
    assert resp.json()["version"] == 3

    # The no-op decision is keyed on the store pointer (refresh precedes read).
    store = _FakeStore(VersionRef("run", 5))
    assert publish_app._is_already_published(store, 3) is True, (
        "older than the pointer is a no-op"
    )
    assert publish_app._is_already_published(store, 5) is True, (
        "equal to the pointer is a no-op"
    )
    assert publish_app._is_already_published(store, 6) is False, (
        "newer than the pointer publishes"
    )
    assert store.calls == ["refresh", "read_pointer"] * 3
    assert publish_app._is_already_published(_FakeStore(None), 1) is False, (
        "with no pointer nothing is a no-op"
    )

    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": "/nonexistent"},
    )
    assert resp.status_code == 400
    assert "source directory not found" in resp.json()["detail"]

    source = tmp_path / "source"
    source.mkdir()
    _write_tiny_safetensors(source / "model.safetensors")
    fake_spawn = _FakeSpawn(job_id="fc-pub-1")
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path)
    client = TestClient(server.build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": str(source)},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "accepted"
    assert resp.json()["job_id"] == "fc-pub-1"
    assert fake_spawn.calls == [
        {
            "run_id": publish_app.RUN_ID,
            "version": 1,
            "source": str(source),
            "publish": True,
        }
    ]

    monkeypatch.setattr(publish_app, "materialize_version", _FakeSpawn())
    calls = {"fc-test-1": _FakeFunctionCall("pending")}
    _patch_job_poll(monkeypatch, calls)
    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": str(source)},
    )
    assert resp.status_code == 202
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 2, "source": str(source)},
    )
    assert resp.status_code == 409
    assert "still running" in resp.json()["detail"]
    calls["fc-test-1"].state = "success"
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 2, "source": str(source)},
    )
    assert resp.status_code == 202

    # Server paths reload the volume before reading state (a sibling container
    # may have staged dirs since this container last looked).
    monkeypatch.setattr(publish_app, "materialize_version", _FakeSpawn("fc-reload-pub"))
    server, store = _make_recording_server(tmp_path, label="reload-publish")
    client = TestClient(server.build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": str(source)},
    )
    assert resp.status_code == 202
    assert store.calls == ["reload", "refresh", "refresh", "read_pointer"], (
        "/publish reloads, then the no-op check refreshes and reads the pointer"
    )

    server, store = _make_recording_server(
        tmp_path, label="reload-status", staged_on_reload=(7,)
    )
    client = TestClient(server.build_app())
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["staged_versions"] == [7], (
        "/status reload exposes a dir staged by another container"
    )
    assert store.calls == ["reload", "refresh", "read_pointer"], (
        "/status reloads before reading the pointer"
    )

    # The default reloader reloads only for the modal-volume backend.
    events: list[str] = []
    monkeypatch.setattr(
        publish_app,
        "run_volume",
        SimpleNamespace(reload=lambda: events.append("reload")),
    )
    monkeypatch.setattr(
        publish_app,
        "STORE_DEPLOYMENT",
        SimpleNamespace(backend=publish_app.storage.MODAL_VOLUME),
    )
    publish_app._default_volume_reloader()
    assert events == ["reload"], "modal-volume backend reloads by default"
    monkeypatch.setattr(
        publish_app,
        "STORE_DEPLOYMENT",
        SimpleNamespace(backend=publish_app.storage.S3),
    )
    publish_app._default_volume_reloader()
    assert events == ["reload"], "non-volume backends skip the reload by default"

    _patch_job_poll(
        monkeypatch,
        {
            "fc-pending": _FakeFunctionCall("pending"),
            "fc-success": _FakeFunctionCall("success", {"version": 1, "path": "/x"}),
            "fc-failure": _FakeFunctionCall("failure"),
        },
    )
    client = TestClient(_make_server(tmp_path).build_app())
    assert client.get("/job/fc-pending").json()["status"] == "pending"
    ok = client.get("/job/fc-success").json()
    assert ok["status"] == "success"
    assert ok["result"] == {"version": 1, "path": "/x"}
    bad = client.get("/job/fc-failure").json()
    assert bad["status"] == "failure"
    assert "remote boom" in bad["error"]

    # /job reads no volume state, so it must not invoke the reloader.
    server, store = _make_recording_server(tmp_path, label="reload-job")
    _patch_job_poll(monkeypatch, {"fc-job": _FakeFunctionCall("pending")})
    client = TestClient(server.build_app())
    resp = client.get("/job/fc-job")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    assert store.calls == [], "/job never reloads the volume"

    # start() binds uvicorn on the configured port; stop() exits the server.
    import sys
    import types

    captured: dict[str, object] = {}

    class _FakeServer:
        def __init__(self, config: object) -> None:
            captured["config"] = config
            self.should_exit = False

        def run(self) -> None:
            captured["ran"] = True

    class _FakeConfig:
        def __init__(self, app: object, **kwargs: object) -> None:
            captured["app"] = app
            captured["kwargs"] = kwargs

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.Config = _FakeConfig  # type: ignore[attr-defined]
    fake_uvicorn.Server = _FakeServer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)

    server = _make_server(tmp_path)
    server.port = 8123
    server.start()
    server.server_thread.join(timeout=5)

    assert captured["ran"] is True
    assert captured["kwargs"] == {
        "host": "0.0.0.0",
        "port": 8123,
        "timeout_keep_alive": 300,
    }
    assert isinstance(captured["app"], publish_app.FastAPI)
    assert server.server_thread.daemon is True

    server.stop()
    assert server.uvicorn_server.should_exit is True


def test_publisher_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foreign run_id rejected; truncated shards recopied; volume reload/commit;
    job poll errors are unknown."""
    from fastapi.testclient import TestClient

    real_materialize = publish_app.materialize_version

    with pytest.raises(ValueError, match="refusing to build a store"):
        publish_app._build_store("someone-elses-run")

    source = _make_publish_source(tmp_path)
    fake_spawn = _FakeSpawn()
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path)
    (server.updates_dir / "weight_v000001").mkdir()
    client = TestClient(server.build_app())
    for path, body in (
        ("/publish", {"run_id": "someone-elses-run", "version": 2, "source": str(source)}),
    ):
        resp = client.post(path, json=body)
        assert resp.status_code == 400, path
        assert "does not match" in resp.json()["detail"], path
    assert fake_spawn.calls == []

    source_dir = _make_source(tmp_path / "src-trunc")
    target = tmp_path / "weight_v000001"
    target.mkdir()
    source_shard = source_dir / "model.safetensors"
    (target / "model.safetensors").write_bytes(source_shard.read_bytes()[:-4])
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {}})
    )
    assert publish_app._target_is_valid(target, source_dir) is False, (
        "truncated shard with the right name is not a valid target"
    )
    assert publish_app._prepare_version_dir(target, source_dir) == "copied"
    assert (target / "model.safetensors").stat().st_size == source_shard.stat().st_size

    events: list[str] = []
    monkeypatch.setattr(
        publish_app,
        "run_volume",
        SimpleNamespace(
            reload=lambda: events.append("reload"),
            commit=lambda: events.append("commit"),
        ),
    )
    monkeypatch.setattr(publish_app, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(
        publish_app,
        "STORE_DEPLOYMENT",
        SimpleNamespace(backend=publish_app.storage.MODAL_VOLUME),
    )
    source_dir = _make_source(tmp_path / "src-mat")
    monkeypatch.setattr(publish_app, "materialize_version", real_materialize)
    result = publish_app.materialize_version.local(
        run_id=publish_app.RUN_ID, version=1, source=str(source_dir), publish=False
    )
    assert result["published"] is False
    assert events == ["reload", "commit"], "staging path reloads then commits"

    events.clear()
    monkeypatch.setattr(
        publish_app,
        "publish_version",
        SimpleNamespace(local=lambda **kwargs: events.append("publish")),
    )
    result = publish_app.materialize_version.local(
        run_id=publish_app.RUN_ID, version=2, source=str(source_dir), publish=True
    )
    assert result["published"] is True
    assert events == ["reload", "publish"], "publish path reloads without staging commit"

    calls = {"fc-modal-to": _FakeFunctionCall("modal-timeout")}
    _patch_job_poll(monkeypatch, calls)
    state = publish_app._job_state("fc-modal-to")
    assert state["status"] == "pending"

    def _boom(job_id: str) -> object:
        raise RuntimeError("modal lookup down")

    monkeypatch.setattr(publish_app, "_function_call_from_id", _boom)
    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.get("/job/fc-gone")
    assert resp.status_code == 200
    assert resp.json()["status"] == "unknown", "lookup error is unknown, not 500"

    calls = {"fc-flaky": _FakeFunctionCall("query-error")}
    _patch_job_poll(monkeypatch, calls)
    state = publish_app._job_state("fc-flaky")
    assert state["status"] == "unknown"
    assert "transient poll error" in state["error"]

    source = _make_publish_source(tmp_path)
    monkeypatch.setattr(publish_app, "materialize_version", _FakeSpawn())
    calls = {"fc-test-1": _FakeFunctionCall("pending")}
    _patch_job_poll(monkeypatch, calls)
    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": str(source)},
    )
    assert resp.status_code == 202
    calls["fc-test-1"].state = "query-error"
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 2, "source": str(source)},
    )
    assert resp.status_code == 409, "unknown poll keeps the single-writer guard"
    assert "still running" in resp.json()["detail"]
