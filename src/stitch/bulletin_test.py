from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import (
    VersionManifest,
    format_snapshot_identity,
    parse_snapshot_identity,
)


class SnapshotIdentityTest(unittest.TestCase):
    def test_format_round_trips_with_and_without_run_id(self) -> None:
        self.assertEqual(format_snapshot_identity("run-a", 5), "run-a/weight_v000005")
        self.assertEqual(format_snapshot_identity(None, 5), "weight_v000005")
        self.assertEqual(parse_snapshot_identity("run-a/weight_v000005"), ("run-a", 5))

    def test_parse_tolerates_bare_legacy_and_garbage(self) -> None:
        self.assertEqual(parse_snapshot_identity("weight_v000005"), (None, 5))  # bare
        self.assertEqual(parse_snapshot_identity("000005"), (None, 5))  # legacy flat pointer
        self.assertEqual(parse_snapshot_identity(""), (None, 0))  # missing -> not-ready
        self.assertEqual(parse_snapshot_identity("garbage"), (None, 0))  # unparseable -> not-ready


class SlimeLayoutBulletinTest(unittest.TestCase):
    """The slime/customer layout: per-run ``<run_id>/weight_v{N}/`` chains and a
    single self-identifying ``latest`` pointer ``<run_id>/weight_v{N}``."""

    def _write_version(self, base: Path, version: int, prev: int) -> Path:
        vdir = base / f"weight_v{version:06d}"
        vdir.mkdir(parents=True)
        (vdir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": f"{version:06d}",
                        "base_version": f"{prev:06d}",
                        "delta_encoding": "xor",
                        "compression_format": "zstd",
                        "checksum_format": "xxh3-128",
                    },
                    "weight_map": {"w": "model-00001-of-00001.safetensors"},
                }
            ),
            encoding="utf-8",
        )
        return vdir

    def test_run_id_layout_reads_chain(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_version(root / "run-a", 1, 0)
            self._write_version(root / "run-a", 2, 1)
            (root / "latest").write_text("run-a/weight_v000002", encoding="utf-8")

            board = FilesystemBulletinBoard(root, layout="slime")

            self.assertEqual(board.read_latest(), ("run-a", 2))
            self.assertEqual(board.version_dir("run-a", 2), root / "run-a" / "weight_v000002")
            manifest = board.read_manifest("run-a", 2)
            self.assertEqual(manifest.version, 2)
            self.assertEqual(manifest.base_version, 1)
            self.assertEqual(manifest.compression_format, "zstd")
            self.assertEqual(manifest.checksum_format, "xxh3-128")

    def test_read_latest_absent_is_none_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(FilesystemBulletinBoard(Path(tmp), layout="slime").read_latest(), (None, 0))

    def test_write_latest_run_id_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = FilesystemBulletinBoard(root, layout="slime")
            board.write_latest("run-a", 7)
            self.assertEqual((root / "latest").read_text(encoding="utf-8"), "run-a/weight_v000007")
            self.assertEqual(board.read_latest(), ("run-a", 7))

    def test_write_latest_none_run_id_writes_bare_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = FilesystemBulletinBoard(root, layout="slime")
            board.write_latest(None, 7)
            self.assertEqual((root / "latest").read_text(encoding="utf-8"), "weight_v000007")
            self.assertEqual(board.read_latest(), (None, 7))

    def test_legacy_bare_pointer_parses_runless(self) -> None:
        # A pre-run-id deployment left `latest` = "000005"; it must parse as a
        # run-less pointer (None, 5), never be misread as a phantom run.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "latest").write_text("000005", encoding="utf-8")
            self.assertEqual(FilesystemBulletinBoard(root, layout="slime").read_latest(), (None, 5))

    def test_publish_manifest_advances_run_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_version(root / "run-a", 1, 0)
            board = FilesystemBulletinBoard(root, layout="slime")
            board.publish_manifest(
                VersionManifest(version=1, base_version=0, backend="disk_delta", load_format="auto"),
                run_id="run-a",
            )
            self.assertEqual(board.read_latest(), ("run-a", 1))

    def test_unknown_layout_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FilesystemBulletinBoard("/tmp", layout="bogus")


class StitchLayoutBulletinTest(unittest.TestCase):
    """The engine-neutral protocol layout is run-less: run_id is always None."""

    def test_publish_and_read_round_trip_run_id_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = FilesystemBulletinBoard(root)  # default layout="stitch"
            manifest = VersionManifest(version=3, base_version=2, backend="fake", load_format="noop")
            board.publish_manifest(manifest)
            self.assertEqual(board.read_latest(), (None, 3))
            self.assertEqual(board.version_dir(None, 3), root / "versions" / "weight_v000003")
            self.assertEqual(board.read_manifest(None, 3).version, 3)


if __name__ == "__main__":
    unittest.main()
