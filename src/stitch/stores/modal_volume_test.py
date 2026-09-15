"""ModalVolumeStore harness, driven through the real ``publish_version`` flow.

Covers everything provable without Modal (volume_name=None → a local dir): the pointer
round-trip, the claim → delta-chain path, and the external-staging copy. The volume-backed
path is validated e2e."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from modal.types import FileEntryType

from stitch.publish import publish_version
from stitch.publisher import Publisher
from stitch.stores import modal_volume
from stitch.stores.modal_volume import ModalVolumeStore
from stitch.types import VersionKind, VersionRef


def _write_version(
    root: Path, ref: VersionRef, *, base: int | None = None, diff: str | None = None
) -> str:
    d = root / "updates" / Path(ref.identity).name
    d.mkdir(parents=True)
    meta: dict = {"version": ref.version}
    if diff:
        meta.update(
            {
                "delta_encoding": diff,
                "base_version": base,
                "compression_format": "zstd",
                "checksum_format": "xxh3-128",
            }
        )
    (d / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": meta, "weight_map": {"w": "model-00001.safetensors"}})
    )
    (d / "model-00001.safetensors").write_bytes(b"\x00")
    return str(d)


def test_publish_full_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root, run_id="r1")
        assert store.read_pointer() is None
        vdir = _write_version(root, VersionRef("r1", 1))  # framework wrote it in place
        ref = publish_version(store, None, vdir, run_id="r1")
        assert ref == VersionRef("r1", 1)
        assert store.read_pointer() == VersionRef(
            "r1", 1
        )  # pointer parses back to the ref
        man = store.read_manifest(ref)
        assert man.kind is VersionKind.FULL
        assert (Path(store.materialize(ref)) / "model.safetensors.index.json").exists()


def test_claim_then_delta_chain() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ModalVolumeStore(root, run_id="r1")
        store.claim(VersionRef("r1", 0))
        assert store.read_pointer() == VersionRef("r1", 0)  # base before any publish
        publish_version(
            store, None, _write_version(root, VersionRef("r1", 1)), run_id="r1"
        )
        publish_version(
            store,
            None,
            _write_version(root, VersionRef("r1", 2), base=1, diff="xor"),
            run_id="r1",
        )
        assert store.read_pointer() == VersionRef("r1", 2)
        man = store.read_manifest(VersionRef("r1", 2))
        assert man.kind is VersionKind.DELTA


def test_copies_when_files_dir_is_external() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, staging = Path(tmp) / "store", Path(tmp) / "staging"
        root.mkdir()
        store = ModalVolumeStore(root, run_id="r1")
        src = _write_version(
            staging, VersionRef("r1", 1)
        )  # a staging dir, not the store layout
        publish_version(store, None, src, run_id="r1")
        assert (
            root / "updates" / "weight_v000001" / "model.safetensors.index.json"
        ).exists()


def test_rejects_a_version_from_another_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store = ModalVolumeStore(tmp, run_id="r1")
        try:
            store.advance_pointer(VersionRef("r2", 1))
        except ValueError as exc:
            assert "scoped to run 'r1'" in str(exc)
        else:
            raise AssertionError("cross-run pointer must fail")


def test_runs_sharing_a_volume_have_independent_state() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        volume_root = Path(tmp)
        first_root = volume_root / "run-a"
        second_root = volume_root / "run-b"
        first = ModalVolumeStore(first_root, run_id="run-a")
        second = ModalVolumeStore(second_root, run_id="run-b")

        first.claim(VersionRef("run-a", 0))
        second.claim(VersionRef("run-b", 0))
        publish_version(
            first,
            None,
            _write_version(first_root, VersionRef("run-a", 1)),
            run_id="run-a",
        )

        assert first.read_pointer() == VersionRef("run-a", 1)
        assert second.read_pointer() == VersionRef("run-b", 0)
        assert Path(first.materialize(VersionRef("run-a", 1))).is_dir()
        assert not Path(second.materialize(VersionRef("run-b", 1))).exists()


class _CommittedVolume:
    def __init__(self, local: Path, remote: Path) -> None:
        self.local, self.remote = local, remote

    def read_file(self, path):
        yield (self.remote / path).read_bytes()

    def iterdir(self, path, *, recursive):
        assert recursive
        for file in (self.remote / path).rglob("*"):
            yield SimpleNamespace(
                path=str(file.relative_to(self.remote)),
                type=FileEntryType.FILE if file.is_file() else FileEntryType.DIRECTORY,
            )

    def commit(self):
        pointer = self.local / "latest"
        if pointer.exists():
            (self.remote / "namespace/r1/latest").write_bytes(pointer.read_bytes())

    def reload(self):
        raise RuntimeError("Cannot reload while native checkpoint files are open")


@pytest.fixture
def committed_store(tmp_path, monkeypatch):
    local = tmp_path / "mount/r1"
    remote = tmp_path / "committed"
    directory = Path(_write_version(local, VersionRef("r1", 1)))
    persisted = remote / "namespace/r1/updates" / directory.name
    persisted.mkdir(parents=True)
    for file in directory.iterdir():
        (persisted / file.name).write_bytes(file.read_bytes())
    (directory / "model-00001.safetensors").unlink()
    (remote / "namespace/r1/latest").write_text("r1/weight_v000000")
    volume = _CommittedVolume(local, remote)
    monkeypatch.setattr(modal_volume, "_volume", lambda name: volume)
    store = ModalVolumeStore(
        local, run_id="r1", volume_name="test", volume_path="namespace/r1"
    )
    return store, directory, persisted, volume


def test_publishes_committed_peer_shards_without_reloading_open_mount(committed_store):
    store, directory, _, _ = committed_store

    Publisher(store, run_id="r1").publish(str(directory))

    assert store.read_pointer() == VersionRef("r1", 1)


def test_reads_durable_pointer_when_local_mount_is_stale(committed_store):
    store, _, _, volume = committed_store
    (store.root / "latest").write_text("r1/weight_v000000")
    (volume.remote / "namespace/r1/latest").write_text("r1/weight_v000004")

    assert store.read_pointer() == VersionRef("r1", 4)


@pytest.mark.parametrize("corruption", ["missing_shard", "directory_shard", "index"])
def test_rejects_incomplete_or_different_committed_version(committed_store, corruption):
    store, directory, persisted, _ = committed_store
    shard = persisted / "model-00001.safetensors"
    if corruption == "index":
        (persisted / "model.safetensors.index.json").write_text("{}")
    else:
        shard.unlink()
        if corruption == "directory_shard":
            shard.mkdir()

    with pytest.raises(RuntimeError, match="checkpoint publication failed"):
        Publisher(store, run_id="r1").publish(str(directory))

    assert store.read_pointer() == VersionRef("r1", 0)


@pytest.mark.parametrize("name", ["../other.safetensors", "/other.safetensors", "."])
def test_api_verification_rejects_shards_outside_version(committed_store, name):
    store, directory, persisted, _ = committed_store
    index = json.dumps({"metadata": {"version": 1}, "weight_map": {"w": name}})
    for root in (directory, persisted):
        (root / "model.safetensors.index.json").write_text(index)

    with pytest.raises(RuntimeError, match="Invalid checkpoint shard path"):
        Publisher(store, run_id="r1").publish(str(directory))

    assert store.read_pointer() == VersionRef("r1", 0)


def test_rejects_pointer_change_during_committed_verification(
    committed_store, monkeypatch
):
    store, directory, _, volume = committed_store
    verify = store.verify_committed_version

    def concurrent_publish(*args):
        manifest = verify(*args)
        (volume.remote / "namespace/r1/latest").write_text("r1/weight_v000002")
        return manifest

    monkeypatch.setattr(store, "verify_committed_version", concurrent_publish)

    with pytest.raises(RuntimeError, match="latest changed while publishing"):
        Publisher(store, run_id="r1").publish(str(directory))

    assert store.read_pointer() == VersionRef("r1", 2)


def test_api_store_preserves_external_staging_copy(
    committed_store, tmp_path, monkeypatch
):
    store, _, _, volume = committed_store
    monkeypatch.setattr(volume, "reload", lambda: None)
    source = _write_version(tmp_path / "staging", VersionRef("r1", 1))

    Publisher(store, run_id="r1").publish(source)

    assert store.read_pointer() == VersionRef("r1", 1)
    assert (store.root / "updates/weight_v000001/model-00001.safetensors").is_file()


@pytest.mark.parametrize("path", ["/absolute", "../other", "run/../other"])
def test_rejects_invalid_volume_relative_root(tmp_path, path):
    with pytest.raises(ValueError, match="relative path"):
        ModalVolumeStore(tmp_path, run_id="r1", volume_name="test", volume_path=path)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"modal_volume harness: {len(tests)} PASS")
