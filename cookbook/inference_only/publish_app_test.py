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


# ── Delta-aware validity ─────────────────────────────────────────────────────


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


def test_target_is_valid_accepts_complete_delta(tmp_path: Path) -> None:
    """A delta's weight_map is a subset of the full source; that must still count as valid."""
    source = _make_source(
        tmp_path / "source",
        shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"),
    )
    target = _make_delta_dir(tmp_path / "weight_v000002")

    assert publish_app._target_is_valid(target, source) is True
    assert publish_app._prepare_version_dir(target, source) == "kept"


def test_target_is_valid_rejects_partial_delta(tmp_path: Path) -> None:
    """A delta missing one of its own shards is partial -> rmtree'd and recopied."""
    source = _make_source(tmp_path / "source")
    target = _make_delta_dir(
        tmp_path / "weight_v000002",
        shards=("model-00001-of-00002.safetensors",),
        index_shards=(
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ),
    )

    assert publish_app._target_is_valid(target, source) is False
    assert publish_app._prepare_version_dir(target, source) == "copied"
    assert not (target / "model-00001-of-00002.safetensors").exists()
    assert (target / "model.safetensors").exists()


# ── Synthetic-delta fabrication (real generator, tiny base) ──────────────────


def _make_fp8_style_base(path: Path, tensor_names: tuple[str, ...]) -> Path:
    """A tiny 'fp8-format' base checkpoint: one shard of zeroed F32 tensors plus
    its HF index (the generator only needs raw bytes + an index)."""
    import numpy as np
    import safetensors.numpy

    path.mkdir(parents=True, exist_ok=True)
    tensors = {name: np.zeros(2048, dtype=np.float32) for name in tensor_names}
    safetensors.numpy.save_file(tensors, str(path / "model-00001.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"version": 0},
                "weight_map": {
                    name: "model-00001.safetensors" for name in tensor_names
                },
            }
        )
    )
    return path


class _DeltaFakeStore:
    """Just enough Store for stitch.publish.publish_version."""

    def __init__(self) -> None:
        self.pointer: VersionRef | None = None
        self.published: list = []

    def read_pointer(self) -> VersionRef | None:
        return self.pointer

    def publish(self, manifest: object, files_dir: str) -> None:
        self.published.append(manifest)

    def advance_pointer(self, ref: VersionRef) -> None:
        self.pointer = ref

    def compare_and_advance_pointer(
        self, expected: VersionRef | None, ref: VersionRef
    ) -> None:
        assert self.pointer == expected
        self.pointer = ref


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "updates").mkdir(parents=True)
    return run_dir


def test_fabricate_delta_dir_produces_publishable_delta(tmp_path: Path) -> None:
    """The real generator's output passes stitch's publish validation, and the
    per-tensor xxh3-128 checksums match a manual XOR-decode of the base bytes."""
    import numpy as np
    import xxhash
    import zstandard
    from safetensors import safe_open

    base = _make_fp8_style_base(
        tmp_path / "base",
        ("model.layers.0.mlp.weight", "model.layers.1.mlp.weight", "model.norm.weight"),
    )
    run_dir = _run_dir(tmp_path)

    version_dir = publish_app.fabricate_delta_dir(
        run_dir, 0, 1, 2, anchor_dir=base, density=0.5, seed=7
    )

    assert version_dir == run_dir / "updates" / "weight_v000001"
    index = json.loads(
        (version_dir / "model.safetensors.index.json").read_text()
    )
    meta = index["metadata"]
    assert meta["version"] == 1
    assert meta["base_version"] == 0
    assert meta["delta_encoding"] == "xor"
    assert meta["compression_format"] == "zstd"
    assert meta["checksum_format"] == "xxh3-128"
    assert 1 <= len(index["weight_map"]) <= 2
    for shard in set(index["weight_map"].values()):
        assert (version_dir / shard).is_file()

    import stitch.publish

    store = _DeltaFakeStore()
    ref = stitch.publish.publish_version(
        store, None, str(version_dir), run_id="run"
    )
    assert ref == VersionRef("run", 1)
    assert store.pointer == VersionRef("run", 1)
    assert store.published[0].kind.value == "delta"

    # Decode zstd/XOR and compare xxh3-128 to shard metadata.
    base_bytes = (base / "model-00001.safetensors").read_bytes()
    base_index = json.loads((base / "model.safetensors.index.json").read_text())
    from tools.profiling._synthetic_delta import _safetensors_header

    data_start, header = _safetensors_header(base / "model-00001.safetensors")
    for shard_name in set(index["weight_map"].values()):
        with safe_open(str(version_dir / shard_name), framework="numpy") as f:
            checksums = f.metadata()
            for name in index["weight_map"]:
                if index["weight_map"][name] != shard_name:
                    continue
                begin, end = header[name]["data_offsets"]
                raw = base_bytes[data_start + begin : data_start + end]
                compressed = f.get_tensor(name).tobytes()
                delta = zstandard.ZstdDecompressor().decompressobj().decompress(compressed)
                assert len(delta) == len(raw)
                applied = np.bitwise_xor(
                    np.frombuffer(raw, dtype=np.uint8),
                    np.frombuffer(delta, dtype=np.uint8),
                )
                assert xxhash.xxh3_128(applied).hexdigest() == checksums[name]
    assert base_index["metadata"]["version"] == 0  # anchor untouched


def test_fabricate_delta_dir_chains_without_retouching_tensors(tmp_path: Path) -> None:
    """A delta on a delta-staged base excludes tensors earlier deltas touched, so
    the base checkpoint's bytes still match the chain's current bytes there."""
    base = _make_fp8_style_base(
        tmp_path / "base",
        ("model.layers.0.mlp.weight", "model.layers.1.mlp.weight", "model.norm.weight"),
    )
    run_dir = _run_dir(tmp_path)

    v1 = publish_app.fabricate_delta_dir(
        run_dir, 0, 1, 1, anchor_dir=base, density=0.5, seed=1
    )
    v2 = publish_app.fabricate_delta_dir(
        run_dir, 1, 2, 1, anchor_dir=base, density=0.5, seed=2
    )

    v1_index = json.loads((v1 / "model.safetensors.index.json").read_text())
    v2_index = json.loads((v2 / "model.safetensors.index.json").read_text())
    assert v2_index["metadata"]["version"] == 2
    assert v2_index["metadata"]["base_version"] == 1
    assert not set(v1_index["weight_map"]) & set(v2_index["weight_map"])

    again = publish_app.fabricate_delta_dir(
        run_dir, 1, 2, 1, anchor_dir=base, density=0.5, seed=2
    )
    assert again == v2


def test_fabricate_delta_dir_excludes_embed_and_lm_head(tmp_path: Path) -> None:
    """embed/lm_head-class tensors dominated target_bytes and compile scope; the
    selector must skip them and still fill num_tensors from per-layer weights."""
    base = _make_fp8_style_base(
        tmp_path / "base",
        (
            "model.embed_tokens.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "lm_head.weight",
        ),
    )
    run_dir = _run_dir(tmp_path)

    version_dir = publish_app.fabricate_delta_dir(
        run_dir, 0, 1, 2, anchor_dir=base, density=0.5, seed=3
    )

    index = json.loads((version_dir / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == {
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }


def test_delta_anchor_dir_resolution(tmp_path: Path) -> None:
    """v0 and staged-FULL bases anchor at their own dirs; a staged-DELTA base
    anchors at the config's base checkpoint (its bytes are XOR blobs, not raw)."""
    run_dir = _run_dir(tmp_path)

    assert publish_app._delta_anchor_dir(run_dir, 0) == Path(
        publish_app.exp.ROLLOUT_CHECKPOINT_PATH
    )

    _make_delta_dir(run_dir / "updates" / "weight_v000001")
    assert publish_app._delta_anchor_dir(run_dir, 1) == Path(
        publish_app.exp.ROLLOUT_CHECKPOINT_PATH
    )

    full = _make_source(run_dir / "updates" / "weight_v000002")
    assert publish_app._delta_anchor_dir(run_dir, 2) == full


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
    server_info_provider: object = _empty_server_infos,
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
        server_info_provider=server_info_provider,  # type: ignore[arg-type]
        # Unit tests must never invoke the real run_volume.reload() (needs Modal auth).
        volume_reloader=volume_reloader or (lambda: None),
    )


_MIXED_INFOS = [
    {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
    {"applied_version": 4, "sync_state": "HOLDING", "target_version": 5},
]

_UNMIXED_INFOS = [
    {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
    {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
]


def _provider_of(infos: list[dict]) -> object:
    async def provider() -> list[dict]:
        return infos

    return provider


def _make_publish_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    _write_tiny_safetensors(source / "model.safetensors")
    return source


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
        server_info_provider=_empty_server_infos,  # type: ignore[arg-type]
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

    source = _make_publish_source(tmp_path)
    fake_spawn = _FakeSpawn()
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path, server_info_provider=_provider_of(_MIXED_INFOS))
    client = TestClient(server.build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 5, "source": str(source)},
    )
    assert resp.status_code == 409
    body = resp.json()
    assert body["status"] == "retryable"
    assert body["reason"] == "fleet mixed"
    assert body["detail"]["applied_versions"] == [3, 4]
    assert body["detail"]["transitioning_replicas"] == [
        {"sync_state": "HOLDING", "target_version": 5, "applied_version": 4}
    ]
    assert fake_spawn.calls == []

    # fleet_is_mixed truth table: one applied version across the fleet is not
    # mixed; two applied versions are; one applied version with a replica
    # moving toward a newer target is; an empty/unreachable probe is not
    # (the publisher must work standalone).
    assert publish_app.fleet_is_mixed(_UNMIXED_INFOS)[0] is False
    assert publish_app.fleet_is_mixed(_MIXED_INFOS)[0] is True
    transitioning = [
        {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
        {"applied_version": 3, "sync_state": "STAGING", "target_version": 4},
    ]
    mixed, detail = publish_app.fleet_is_mixed(transitioning)
    assert mixed is True
    assert detail["applied_versions"] == [3]
    assert detail["transitioning_replicas"] == [
        {"sync_state": "STAGING", "target_version": 4, "applied_version": 3}
    ]
    assert publish_app.fleet_is_mixed([]) == (
        False,
        {"applied_versions": [], "transitioning_replicas": []},
    )

    fake_spawn = _FakeSpawn(job_id="fc-force-1")
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path, server_info_provider=_provider_of(_MIXED_INFOS))
    client = TestClient(server.build_app())
    resp = client.post(
        "/publish",
        json={
            "run_id": publish_app.RUN_ID,
            "version": 5,
            "source": str(source),
            "force": True,
        },
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "fc-force-1"

    fake_spawn = _FakeSpawn(job_id="fc-unmixed-1")
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path, server_info_provider=_provider_of(_UNMIXED_INFOS))
    client = TestClient(server.build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 4, "source": str(source)},
    )
    assert resp.status_code == 202
    assert fake_spawn.calls == [
        {
            "run_id": publish_app.RUN_ID,
            "version": 4,
            "source": str(source),
            "publish": True,
        }
    ]


def test_fabricate_walk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """fabricate -> fabricate_delta -> publish: spawn args, 400 guards, reload
    ordering, and the cpu-mode delta-only rejection (handler + job sides)."""
    from fastapi.testclient import TestClient

    real_materialize = publish_app.materialize_version

    fake_spawn = _FakeSpawn(job_id="fc-fab-1")
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    server = _make_server(tmp_path)
    from_dir = server.updates_dir / "weight_v000002"
    from_dir.mkdir()
    _write_tiny_safetensors(from_dir / "model.safetensors")
    client = TestClient(server.build_app())
    resp = client.post(
        "/fabricate",
        json={"run_id": publish_app.RUN_ID, "from_version": 2, "new_version": 4},
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "fc-fab-1"
    assert fake_spawn.calls == [
        {
            "run_id": publish_app.RUN_ID,
            "version": 4,
            "source": str(from_dir),
            "publish": False,
        }
    ]

    fake_spawn = _FakeSpawn(job_id="fc-delta-1")
    monkeypatch.setattr(publish_app, "fabricate_delta_version", fake_spawn)
    server = _make_server(tmp_path)
    (server.updates_dir / "weight_v000001").mkdir()
    client = TestClient(server.build_app())
    resp = client.post(
        "/fabricate_delta",
        json={"run_id": publish_app.RUN_ID, "base_version": 1, "new_version": 2},
    )
    assert resp.status_code == 202
    assert resp.json()["job_id"] == "fc-delta-1"
    assert resp.json()["version"] == 2
    assert resp.json()["base_version"] == 1
    assert resp.json()["path"].endswith("weight_v000002")
    assert fake_spawn.calls == [
        {
            "run_id": publish_app.RUN_ID,
            "base_version": 1,
            "new_version": 2,
            "num_tensors": 4,
        }
    ]

    client = TestClient(_make_server(tmp_path).build_app())
    resp = client.post(
        "/fabricate_delta",
        json={"run_id": publish_app.RUN_ID, "base_version": 2, "new_version": 2},
    )
    assert resp.status_code == 400, "new_version must exceed base_version"
    resp = client.post(
        "/fabricate_delta",
        json={"run_id": publish_app.RUN_ID, "base_version": 1, "new_version": 2},
    )
    assert resp.status_code == 400
    assert "not staged" in resp.json()["detail"]
    resp = client.post(
        "/fabricate_delta",
        json={
            "run_id": publish_app.RUN_ID,
            "base_version": 0,
            "new_version": 1,
            "num_tensors": 0,
        },
    )
    assert resp.status_code == 400, "num_tensors must be positive"

    # The reloader exposes a from_dir staged by another container.
    monkeypatch.setattr(publish_app, "materialize_version", _FakeSpawn("fc-reload-fab"))
    monkeypatch.setattr(publish_app, "fabricate_delta_version", _FakeSpawn("fc-reload-delta"))
    server, store = _make_recording_server(
        tmp_path, label="reload-fabricate", staged_on_reload=(2,)
    )
    client = TestClient(server.build_app())
    resp = client.post(
        "/fabricate",
        json={"run_id": publish_app.RUN_ID, "from_version": 2, "new_version": 4},
    )
    assert resp.status_code == 202
    assert store.calls == ["reload", "refresh"], (
        "/fabricate reloads before checking the staged source"
    )

    # /fabricate_delta: reload makes the staged base visible before the 400 check.
    server, store = _make_recording_server(
        tmp_path, label="reload-fabricate-delta", staged_on_reload=(1,)
    )
    client = TestClient(server.build_app())
    resp = client.post(
        "/fabricate_delta",
        json={"run_id": publish_app.RUN_ID, "base_version": 1, "new_version": 2},
    )
    assert resp.status_code == 202
    assert store.calls == ["reload", "refresh"], (
        "/fabricate_delta reloads before checking the staged base"
    )

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
    monkeypatch.setattr(
        publish_app,
        "run_volume",
        SimpleNamespace(
            reload=lambda: events.append("reload"),
            commit=lambda: events.append("commit"),
        ),
    )
    monkeypatch.setattr(publish_app, "RUN_DIR", tmp_path / "run-cpu")
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
            "model_dir": publish_app._version_dir(tmp_path / "run-cpu", 2),
        }
    ]


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

    monkeypatch.setattr(publish_app, "fabricate_delta_version", fake_spawn)
    for path, body in (
        ("/fabricate", {"run_id": "someone-elses-run", "from_version": 1, "new_version": 2}),
        ("/fabricate_delta", {"run_id": "someone-elses-run", "base_version": 1, "new_version": 2}),
    ):
        resp = client.post(path, json=body)
        assert resp.status_code == 400, path
        assert "does not match" in resp.json()["detail"], path
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
