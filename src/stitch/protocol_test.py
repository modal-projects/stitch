from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import Artifact, VersionManifest, WeightVersionPolicy, read_latest


class ProtocolTest(unittest.TestCase):
    def test_manifest_round_trips_extended_and_legacy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            board = FilesystemBulletinBoard(root)
            manifest = VersionManifest(
                version=3,
                base_version=2,
                backend="sparse_delta",
                load_format="delta",
                transition_files=["rank0000_flush000000.safetensors"],
                artifacts=[
                    Artifact(
                        kind="transition",
                        path="rank0000_flush000000.safetensors",
                        checksum="sha256:abc",
                    )
                ],
                run_id="run-1",
                base_model="Qwen/Qwen3-4B",
            )

            board.publish_manifest(manifest)
            loaded = board.read_manifest(3)

            self.assertEqual(read_latest(root), 3)
            self.assertEqual(loaded.version, 3)
            self.assertEqual(loaded.base_version, 2)
            self.assertEqual(loaded.transition_artifact_paths(), ["rank0000_flush000000.safetensors"])
            self.assertEqual(loaded.artifacts[0].checksum, "sha256:abc")
            self.assertEqual(loaded.run_id, "run-1")

    def test_weight_version_policy_ignores_malformed_payload(self) -> None:
        self.assertEqual(WeightVersionPolicy.from_payload({}), WeightVersionPolicy())
        self.assertEqual(WeightVersionPolicy.from_payload({"weight_version": 7}), WeightVersionPolicy())
        self.assertEqual(
            WeightVersionPolicy.from_payload({"weight_version": {"min_required_version": "5", "exact_version": 6}}),
            WeightVersionPolicy(min_required_version=5, exact_version=6),
        )


if __name__ == "__main__":
    unittest.main()
