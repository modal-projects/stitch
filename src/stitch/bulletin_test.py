from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import VersionManifest


class SlimeLayoutBulletinTest(unittest.TestCase):
    """The slime/customer flat layout: weight_v{N}/ dirs + a raw `latest`
    pointer + per-version model.safetensors.index.json."""

    def _write_version(self, root: Path, version: int, base: int) -> Path:
        vdir = root / f"weight_v{version:06d}"
        vdir.mkdir(parents=True)
        (vdir / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {
                        "version": f"{version:06d}",
                        "base_version": f"{base:06d}",
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

    def test_flat_layout_reads_slime_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_version(root, 1, 0)
            self._write_version(root, 2, 1)
            (root / "latest").write_text("000002", encoding="utf-8")

            board = FilesystemBulletinBoard(root, layout="slime")

            self.assertEqual(board.read_latest(), 2)
            self.assertEqual(board.version_dir(2), root / "weight_v000002")
            manifest = board.read_manifest(2)
            self.assertEqual(manifest.version, 2)
            self.assertEqual(manifest.base_version, 1)
            self.assertEqual(manifest.compression_format, "zstd")
            self.assertEqual(manifest.checksum_format, "xxh3-128")

    def test_read_latest_absent_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(FilesystemBulletinBoard(Path(tmp), layout="slime").read_latest(), 0)

    def test_write_latest_is_raw_zero_padded_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = FilesystemBulletinBoard(root, layout="slime")
            board.write_latest(7)
            self.assertEqual((root / "latest").read_text(encoding="utf-8"), "000007")
            self.assertEqual(board.read_latest(), 7)

    def test_publish_manifest_only_advances_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_version(root, 1, 0)
            board = FilesystemBulletinBoard(root, layout="slime")
            board.publish_manifest(
                VersionManifest(version=1, base_version=0, backend="disk_delta", load_format="auto")
            )
            self.assertEqual(board.read_latest(), 1)

    def test_unknown_layout_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FilesystemBulletinBoard("/tmp", layout="bogus")


if __name__ == "__main__":
    unittest.main()
