"""Publisher: index generation, pointer-keyed no-op, fleet-mixed gate."""

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
    (target / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    assert publish_app._prepare_version_dir(target, source2) == "copied"
    assert (target / "model-00002-of-00002.safetensors").exists()

    # valid target kept
    target = _make_source(tmp_path / "weight_v000004")
    (target / "extra-note").write_text("keep me")
    assert publish_app._prepare_version_dir(target, source) == "kept"
    assert (target / "extra-note").exists()

    # truncated shard with the right name is not a valid target -> recopied
    target = tmp_path / "weight_v000005"
    target.mkdir()
    source_shard = source / "model.safetensors"
    (target / "model.safetensors").write_bytes(source_shard.read_bytes()[:-4])
    (target / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": {}}))
    assert publish_app._target_is_valid(target, source) is False
    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert (target / "model.safetensors").stat().st_size == source_shard.stat().st_size


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


async def _empty_server_infos() -> list[dict]:
    return []


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


# ── cpu-mode FULL guard ──────────────────────────────────────────────────────


def _make_delta_dir(
    path: Path,
    *,
    version: int = 2,
    base_version: int = 1,
    shards: tuple[str, ...] = ("model-00001-of-00002.safetensors",),
    index_shards: tuple[str, ...] = ("model-00001-of-00002.safetensors",),
) -> Path:
    """A staged DELTA dir: index metadata carries delta_encoding; weight_map lists
    only the delta's own shards (a subset of any full checkpoint's shard set)."""
    path.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        _write_tiny_safetensors(path / shard)
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {
                    "version": version,
                    "base_version": base_version,
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "xxh3-128",
                },
                "weight_map": {f"w{i}": shard for i, shard in enumerate(index_shards)},
            }
        )
    )
    return path


def _patch_materialize_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, events: list[str]
) -> list[dict]:
    """Point materialize_version at a tmp RUN_DIR backed by a recording volume;
    returns the list that publish_version.local kwargs are appended to."""
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
    published: list[dict] = []
    monkeypatch.setattr(
        publish_app,
        "publish_version",
        SimpleNamespace(local=lambda **kwargs: published.append(kwargs)),
    )
    return published


def test_cpu_mode_full_publish_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cpu update mode is delta-only: the /publish handler rejects a FULL source
    pre-spawn (400, no job spawned) but accepts a delta source, and the spawned
    materialize job re-checks post-index, pre-pointer (RuntimeError, no publish,
    no staging commit) so the two checks cannot drift."""
    from fastapi.testclient import TestClient

    real_materialize = publish_app.materialize_version

    # cpu mode rejects FULL sources pre-spawn but accepts a delta source.
    monkeypatch.setattr(publish_app.exp, "SGLANG_DELTA_UPDATE_MODE", "cpu")
    fake_spawn = _FakeSpawn("fc-cpu")
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    client = TestClient(_make_server(tmp_path).build_app())

    # Missing index is treated as FULL (the job would generate a non-delta index).
    source_no_index = _make_publish_source(tmp_path)
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": str(source_no_index)},
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "cpu update mode is delta-only" in detail
    assert "FULL version 1" in detail
    assert "model.safetensors.index.json" in detail
    assert fake_spawn.calls == []

    # An existing index without delta_encoding is also FULL.
    source_full_index = _make_source(tmp_path / "source-full-index")
    resp = client.post(
        "/publish",
        json={
            "run_id": publish_app.RUN_ID,
            "version": 2,
            "source": str(source_full_index),
        },
    )
    assert resp.status_code == 400
    assert "FULL version 2" in resp.json()["detail"]
    assert fake_spawn.calls == []

    # A delta source (delta_encoding in metadata) passes the guard and spawns.
    source_delta = _make_delta_dir(tmp_path / "source-delta", version=3, base_version=2)
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 3, "source": str(source_delta)},
    )
    assert resp.status_code == 202
    assert fake_spawn.calls == [
        {
            "run_id": publish_app.RUN_ID,
            "version": 3,
            "source": str(source_delta),
            "publish": True,
        }
    ]

    # The spawned job re-checks after index stamping and before publish_version.
    monkeypatch.setattr(publish_app, "materialize_version", real_materialize)
    events: list[str] = []
    published = _patch_materialize_env(monkeypatch, tmp_path, events)

    full_source = _make_publish_source(tmp_path)
    with pytest.raises(RuntimeError, match="cpu update mode is delta-only"):
        publish_app.materialize_version.local(
            run_id=publish_app.RUN_ID,
            version=1,
            source=str(full_source),
            publish=True,
        )
    assert published == []
    assert events == ["reload"], "no staging commit should run on the rejected publish"

    delta_source = _make_delta_dir(tmp_path / "delta-source", version=2, base_version=1)
    result = publish_app.materialize_version.local(
        run_id=publish_app.RUN_ID,
        version=2,
        source=str(delta_source),
        publish=True,
    )
    assert result["published"] is True
    assert published == [
        {
            "run_id": publish_app.RUN_ID,
            "model_dir": publish_app._version_dir(tmp_path / "run", 2),
        }
    ]


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
    spawn/409, job states, volume-reload ordering, and the mixed-fleet gate."""
    from fastapi.testclient import TestClient

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
        resp = TestClient(server.build_app()).get("/status")
        assert resp.status_code == 200
        assert resp.json() == expected, "latest is the pointer, not a partial dir"

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

def test_publisher_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foreign run_id rejected before spawn; volume reload/commit ordering;
    poll timeouts and errors keep the single-writer guard."""
    from fastapi.testclient import TestClient

    real_materialize = publish_app.materialize_version

    with pytest.raises(ValueError, match="refusing to build a store"):
        publish_app._build_store("someone-elses-run")

    source = _make_publish_source(tmp_path)
    fake_spawn = _FakeSpawn()
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.post(
        "/publish",
        json={"run_id": "someone-elses-run", "version": 2, "source": str(source)},
    )
    assert resp.status_code == 400
    assert "does not match" in resp.json()["detail"]
    assert fake_spawn.calls == []

    events: list[str] = []
    published = _patch_materialize_env(monkeypatch, tmp_path, events)
    source_dir = _make_source(tmp_path / "src-mat")
    monkeypatch.setattr(publish_app, "materialize_version", real_materialize)
    result = publish_app.materialize_version.local(
        run_id=publish_app.RUN_ID, version=1, source=str(source_dir), publish=False
    )
    assert result["published"] is False
    assert events == ["reload", "commit"], "staging path reloads then commits"

    events.clear()
    result = publish_app.materialize_version.local(
        run_id=publish_app.RUN_ID, version=2, source=str(source_dir), publish=True
    )
    assert result["published"] is True
    assert events == ["reload"], "publish path reloads without staging commit"
    assert published == [
        {
            "run_id": publish_app.RUN_ID,
            "model_dir": publish_app._version_dir(tmp_path / "run", 2),
        }
    ]

    calls = {"fc-modal-to": _FakeFunctionCall("modal-timeout")}
    _patch_job_poll(monkeypatch, calls)
    assert publish_app._job_state("fc-modal-to")["status"] == "pending"

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
