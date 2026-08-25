"""Publisher: index generation, pointer-keyed no-op, fleet-mixed gate."""

import os

os.environ.setdefault("EXPERIMENT_CONFIG", "qwen3_0p6b_poc")
os.environ.setdefault("RUN_ID", "publish-app-test")

import asyncio
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


def test_index_generated_from_real_safetensors_file(tmp_path: Path) -> None:
    _write_tiny_safetensors(tmp_path / "model.safetensors")

    publish_app._ensure_safetensors_index(tmp_path, 7)

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["metadata"]["version"] == 7
    assert index["weight_map"] == {"w1": "model.safetensors", "w2": "model.safetensors"}


def test_index_existing_gets_version_rewritten(tmp_path: Path) -> None:
    _write_tiny_safetensors(tmp_path / "model.safetensors")
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
    assert index["metadata"]["version"] == 3
    assert set(index["weight_map"]) == {"w1", "w2"}


def test_index_missing_weights_raises_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no safetensors files"):
        publish_app._ensure_safetensors_index(tmp_path, 1)


def test_index_corrupt_existing_raises_loudly(tmp_path: Path) -> None:
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


def test_noop_keyed_on_store_pointer() -> None:
    store = _FakeStore(VersionRef("run", 5))
    assert publish_app._is_already_published(store, 3) is True  # older than pointer
    assert publish_app._is_already_published(store, 5) is True  # equal to pointer
    assert publish_app._is_already_published(store, 6) is False  # newer than pointer
    assert store.calls == ["refresh", "read_pointer"] * 3


def test_noop_with_no_pointer() -> None:
    assert publish_app._is_already_published(_FakeStore(None), 1) is False


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


def test_fleet_is_mixed() -> None:
    infos = [
        {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
        {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
    ]
    mixed, detail = publish_app.fleet_is_mixed(infos)
    assert mixed is False
    assert detail["applied_versions"] == [3]
    assert detail["transitioning_replicas"] == []

    mixed, detail = publish_app.fleet_is_mixed([])
    assert mixed is False
    assert detail["applied_versions"] == []

    infos = [
        {"applied_version": 3, "sync_state": "IDLE"},
        {"applied_version": 4, "sync_state": "IDLE"},
    ]
    mixed, detail = publish_app.fleet_is_mixed(infos)
    assert mixed is True
    assert detail["applied_versions"] == [3, 4]

    infos = [
        {"applied_version": 3, "sync_state": "Holding", "target_version": 4},
        {"applied_version": 3, "sync_state": "IDLE", "target_version": 3},
    ]
    mixed, detail = publish_app.fleet_is_mixed(infos)
    assert mixed is True
    assert detail["transitioning_replicas"] == [
        {"sync_state": "Holding", "target_version": 4, "applied_version": 3}
    ]

    infos = [
        {
            "applied_version": 3,
            "sync_state": "COMMITTING",
            "metrics": {"target_version": 5},
        }
    ]
    mixed, detail = publish_app.fleet_is_mixed(infos)
    assert mixed is True
    assert detail["transitioning_replicas"][0]["target_version"] == 5

    for info in (
        {"applied_version": 3, "sync_state": "HOLDING", "target_version": None},
        {"applied_version": 3, "sync_state": "STAGING"},
    ):
        mixed, detail = publish_app.fleet_is_mixed([info])
        assert mixed is False
        assert detail["transitioning_replicas"] == []

    for info in (
        {"applied_version": 3, "target_version": 9},
        {"applied_version": 3, "sync_state": None, "target_version": 9},
    ):
        mixed, _ = publish_app.fleet_is_mixed([info])
        assert mixed is False

    infos = [
        {"applied_version": 4, "sync_state": "COMMITTING", "target_version": 4},
        {"applied_version": 4, "sync_state": "STAGING", "target_version": 3},
    ]
    mixed, detail = publish_app.fleet_is_mixed(infos)
    assert mixed is False
    assert detail["transitioning_replicas"] == []

    infos = [{"applied_version": None, "sync_state": "staging", "target_version": 0}]
    mixed, detail = publish_app.fleet_is_mixed(infos)
    assert mixed is True
    assert detail["transitioning_replicas"] == [
        {"sync_state": "staging", "target_version": 0, "applied_version": None}
    ]


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


# ── Server-path volume reload / cpu-mode FULL guard ─────────────────────────


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


def test_server_paths_reload_volume_before_reading_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """publish/fabricate/fabricate_delta/status run the reloader before state reads."""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(publish_app, "materialize_version", _FakeSpawn("fc-reload-pub"))
    monkeypatch.setattr(publish_app, "fabricate_delta_version", _FakeSpawn("fc-reload-delta"))

    # /publish: reload -> refresh, then the no-op check refreshes and reads the pointer.
    server, store = _make_recording_server(tmp_path, label="reload-publish")
    source = _make_publish_source(tmp_path)
    client = TestClient(server.build_app())
    resp = client.post(
        "/publish",
        json={"run_id": publish_app.RUN_ID, "version": 1, "source": str(source)},
    )
    assert resp.status_code == 202
    assert store.calls == ["reload", "refresh", "refresh", "read_pointer"]

    # /fabricate: the reloader exposes a from_dir staged by another container.
    server, store = _make_recording_server(
        tmp_path, label="reload-fabricate", staged_on_reload=(2,)
    )
    client = TestClient(server.build_app())
    resp = client.post(
        "/fabricate",
        json={"run_id": publish_app.RUN_ID, "from_version": 2, "new_version": 4},
    )
    assert resp.status_code == 202
    assert store.calls == ["reload", "refresh"]

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
    assert store.calls == ["reload", "refresh"]

    # /status: reload precedes pointer read and exposes a newly staged dir.
    server, store = _make_recording_server(
        tmp_path, label="reload-status", staged_on_reload=(7,)
    )
    client = TestClient(server.build_app())
    resp = client.get("/status")
    assert resp.status_code == 200
    assert resp.json()["staged_versions"] == [7]
    assert store.calls == ["reload", "refresh", "read_pointer"]


def test_job_does_not_reload_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """/job reads no volume state, so it must not invoke the reloader."""
    from fastapi.testclient import TestClient

    server, store = _make_recording_server(tmp_path, label="reload-job")
    _patch_job_poll(monkeypatch, {"fc-job": _FakeFunctionCall("pending")})
    client = TestClient(server.build_app())
    resp = client.get("/job/fc-job")
    assert resp.status_code == 200
    assert resp.json()["status"] == "pending"
    assert store.calls == []


def test_default_volume_reloader_follows_backend_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default reloader reloads only for the modal-volume backend."""
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
    assert events == ["reload"]

    monkeypatch.setattr(
        publish_app,
        "STORE_DEPLOYMENT",
        SimpleNamespace(backend=publish_app.storage.S3),
    )
    publish_app._default_volume_reloader()
    assert events == ["reload"]


def test_publish_rejects_full_publish_in_cpu_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cpu mode rejects FULL sources pre-spawn but accepts a delta source."""
    from fastapi.testclient import TestClient

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


def test_materialize_version_rejects_full_publish_in_cpu_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spawned job re-checks after index stamping and before publish_version."""
    monkeypatch.setattr(publish_app.exp, "SGLANG_DELTA_UPDATE_MODE", "cpu")
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


def test_publisher_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Status, pointer-keyed no-op, spawn/409, fabricate, mixed-fleet gate."""
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
    assert resp.status_code == 400
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
    assert resp.status_code == 400

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
    assert fake_spawn.calls == []

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


def test_start_binds_uvicorn_on_configured_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    discovery failure is an empty probe; job poll errors are unknown."""
    from fastapi.testclient import TestClient

    real_materialize = publish_app.materialize_version

    with pytest.raises(ValueError, match="refusing to build a store"):
        publish_app._build_store("someone-elses-run")

    source = _make_publish_source(tmp_path)
    fake_spawn = _FakeSpawn()
    monkeypatch.setattr(publish_app, "materialize_version", fake_spawn)
    monkeypatch.setattr(publish_app, "fabricate_delta_version", fake_spawn)
    server = _make_server(tmp_path)
    (server.updates_dir / "weight_v000001").mkdir()
    client = TestClient(server.build_app())
    for path, body in (
        ("/publish", {"run_id": "someone-elses-run", "version": 2, "source": str(source)}),
        (
            "/fabricate",
            {"run_id": "someone-elses-run", "from_version": 1, "new_version": 2},
        ),
        (
            "/fabricate_delta",
            {"run_id": "someone-elses-run", "base_version": 1, "new_version": 2},
        ),
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

    import stitch.pools.modal_flash as modal_flash

    class _DownPool:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def discover_replicas_async(self) -> list[str]:
            raise RuntimeError("pool down")

    monkeypatch.setattr(modal_flash, "ModalFlashPool", _DownPool)
    assert asyncio.run(publish_app._default_server_info_provider()) == []

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
