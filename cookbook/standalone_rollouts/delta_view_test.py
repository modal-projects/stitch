from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cookbook.standalone_rollouts.delta_view import (
    INDEX_FILENAME,
    DeltaIndexError,
    DerivedDeltaConflict,
    LocalDeltaView,
    derive_delta_index,
)
from cookbook.standalone_rollouts.ledger import (
    DeltaFormats,
    IdentityLedger,
    LedgerEntry,
)


FORMATS = DeltaFormats(
    delta_encoding="xor",
    compression_format="zstd",
    checksum_format="xxh3-128",
)


def _entry(identity: str = "delta-a", *, version: int = 1) -> LedgerEntry:
    return LedgerEntry(
        version=version,
        identity=identity,
        previous_snapshot_identity="base" if version == 1 else "delta-a",
        formats=FORMATS,
    )


def _write_upload(
    root: Path,
    identity: str,
    *,
    index: object | None = None,
    shard_names: tuple[str, ...] = ("model-00001-of-00001.safetensors",),
) -> Path:
    upload = root / identity
    upload.mkdir(parents=True)
    if index is None:
        index = {
            "metadata": {"total_size": 3},
            "weight_map": {"tensor": shard_names[0]} if shard_names else {},
        }
    (upload / INDEX_FILENAME).write_text(json.dumps(index), encoding="utf-8")
    for name in shard_names:
        (upload / name).write_bytes(f"bytes:{name}".encode())
    return upload


class DeriveIndexTest(unittest.TestCase):
    def test_derives_metadata_without_touching_the_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upload = _write_upload(root, "delta-a")
            before = {path.name: path.read_bytes() for path in upload.iterdir()}

            derived = derive_delta_index(root, _entry(), committed=False)

            self.assertEqual(
                {path.name: path.read_bytes() for path in upload.iterdir()}, before
            )
            body = json.loads(derived.index_path.read_text(encoding="utf-8"))
            self.assertEqual(
                body["metadata"],
                {
                    "version": "000001",
                    "base_version": "000000",
                    "delta_encoding": "xor",
                    "compression_format": "zstd",
                    "checksum_format": "xxh3-128",
                },
            )
            self.assertEqual(
                body["weight_map"], {"tensor": "model-00001-of-00001.safetensors"}
            )
            self.assertNotIn(".stitch", {path.name for path in upload.iterdir()})

    def test_rejects_an_unvalidated_ledger_entry_before_path_access(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            unsafe = LedgerEntry(
                version=1,
                identity="../outside",
                previous_snapshot_identity="base",
                formats=FORMATS,
            )
            with self.assertRaises(DeltaIndexError):
                derive_delta_index(Path(tmp), unsafe, committed=False)

    def test_empty_weight_map_is_a_valid_no_op_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_upload(
                root,
                "delta-a",
                index={"metadata": {}, "weight_map": {}},
                shard_names=(),
            )
            derived = derive_delta_index(root, _entry(), committed=False)
            self.assertEqual(derived.shard_names, ())

    def test_rejects_malformed_index_or_unsafe_shard_names(self) -> None:
        invalid_indexes = (
            [],
            {"weight_map": {}},
            {"metadata": [], "weight_map": {}},
            {"metadata": {}, "weight_map": []},
            {"metadata": {}, "weight_map": {"tensor": 7}},
            {"metadata": {}, "weight_map": {"tensor": "../outside.safetensors"}},
            {"metadata": {}, "weight_map": {"tensor": "/absolute.safetensors"}},
            {"metadata": {}, "weight_map": {"tensor": "nested/shard.safetensors"}},
            {"metadata": {}, "weight_map": {"tensor": "nested\\shard.safetensors"}},
            {"metadata": {}, "weight_map": {"tensor": "weights.bin"}},
        )
        for index in invalid_indexes:
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_upload(root, "delta-a", index=index, shard_names=())
                with self.assertRaises(DeltaIndexError):
                    derive_delta_index(root, _entry(), committed=False)

    def test_missing_referenced_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_upload(
                root,
                "delta-a",
                index={
                    "metadata": {},
                    "weight_map": {"tensor": "missing.safetensors"},
                },
                shard_names=(),
            )
            with self.assertRaises(FileNotFoundError):
                derive_delta_index(root, _entry(), committed=False)

    def test_referenced_shard_must_be_a_regular_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            upload = _write_upload(root, "delta-a", shard_names=())
            (upload / "directory.safetensors").mkdir()
            (upload / INDEX_FILENAME).write_text(
                json.dumps(
                    {
                        "metadata": {},
                        "weight_map": {"tensor": "directory.safetensors"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(DeltaIndexError):
                derive_delta_index(root, _entry(), committed=False)

    def test_retry_recreates_missing_but_rejects_different_committed_index(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_upload(root, "delta-a")
            first = derive_delta_index(root, _entry(), committed=False)
            original = first.index_path.read_bytes()

            derive_delta_index(root, _entry(), committed=True)
            first.index_path.unlink()
            recreated = derive_delta_index(root, _entry(), committed=True)
            self.assertEqual(recreated.index_path.read_bytes(), original)

            recreated.index_path.write_text("{}", encoding="utf-8")
            with self.assertRaises(DerivedDeltaConflict):
                derive_delta_index(root, _entry(), committed=True)

            recreated.index_path.write_bytes(b"\xff")
            with self.assertRaises(DerivedDeltaConflict):
                derive_delta_index(root, _entry(), committed=True)

    def test_new_entry_overwrites_a_precommit_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_upload(root, "delta-a")
            orphan = root / ".stitch" / "deltas" / "delta-a" / INDEX_FILENAME
            orphan.parent.mkdir(parents=True)
            orphan.write_text("orphan", encoding="utf-8")
            derived = derive_delta_index(root, _entry(), committed=False)
            self.assertNotEqual(
                derived.index_path.read_text(encoding="utf-8"), "orphan"
            )


class LocalDeltaViewTest(unittest.TestCase):
    def _ledger_and_uploads(self, transport: Path) -> IdentityLedger:
        ledger = IdentityLedger.new("base")
        first = ledger.append_delta("delta-a", "base", FORMATS).entry
        second = ledger.append_delta("delta-b", "delta-a", FORMATS).entry
        _write_upload(
            transport,
            "delta-a",
            shard_names=("a.safetensors", "unreferenced.safetensors"),
        )
        _write_upload(transport, "delta-b", shard_names=("b.safetensors",))
        derive_delta_index(transport, first, committed=False)
        derive_delta_index(transport, second, committed=False)
        return ledger

    def test_builds_only_derived_index_and_referenced_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            transport, local = base / "transport", base / "local"
            transport.mkdir()
            ledger = self._ledger_and_uploads(transport)

            LocalDeltaView(local, transport).rebuild(ledger)

            self.assertFalse((local / "weight_v000000").exists())
            version = local / "weight_v000001"
            self.assertTrue((version / INDEX_FILENAME).is_file())
            self.assertFalse((version / INDEX_FILENAME).is_symlink())
            self.assertTrue((version / "a.safetensors").is_symlink())
            self.assertFalse((version / "unreferenced.safetensors").exists())
            self.assertTrue((local / "latest").is_symlink())

    def test_missing_shard_never_installs_partial_version_and_can_retry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            transport, local = base / "transport", base / "local"
            transport.mkdir()
            ledger = IdentityLedger.new("base")
            entry = ledger.append_delta("delta-a", "base", FORMATS).entry
            upload = _write_upload(transport, "delta-a")
            derive_delta_index(transport, entry, committed=False)
            shard = upload / "model-00001-of-00001.safetensors"
            contents = shard.read_bytes()
            shard.unlink()

            view = LocalDeltaView(local, transport)
            with self.assertRaises(FileNotFoundError):
                view.rebuild(ledger)
            self.assertFalse((local / "weight_v000001").exists())
            self.assertEqual(list(local.glob(".*.tmp-*")), [])

            shard.write_bytes(contents)
            view.rebuild(ledger)
            self.assertTrue((local / "weight_v000001").is_dir())

    def test_concurrent_rebuilds_leave_one_complete_immutable_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            transport, local = base / "transport", base / "local"
            transport.mkdir()
            ledger = self._ledger_and_uploads(transport)
            view = LocalDeltaView(local, transport)

            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(lambda _unused: view.rebuild(ledger), range(2)))

            version = local / "weight_v000002"
            self.assertTrue((version / INDEX_FILENAME).is_file())
            self.assertTrue((version / "b.safetensors").is_symlink())
            self.assertEqual(list(local.glob(".*.tmp-*")), [])

    def test_existing_version_must_be_an_owned_directory_not_a_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            transport, local, outside = (
                base / "transport",
                base / "local",
                base / "outside",
            )
            transport.mkdir()
            local.mkdir()
            outside.mkdir()
            ledger = self._ledger_and_uploads(transport)
            (local / "weight_v000001").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(DeltaIndexError):
                LocalDeltaView(local, transport).rebuild(ledger)


if __name__ == "__main__":
    unittest.main()
