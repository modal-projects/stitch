from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from stitch.protocol import VersionManifest, read_latest
from stitch.trainers.slime import publish_delta_version


class SlimeHooksTest(unittest.TestCase):
    def test_publish_delta_version_writes_manifest_latest_and_wakes_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "versions" / "weight_v000001"
            version_dir.mkdir(parents=True)
            args = Namespace(
                update_weight_delta_root=str(root),
                update_weight_delta_dir=str(root / "versions"),
                hf_checkpoint="Qwen/Qwen3-4B",
                run_id="run-1",
            )

            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume", "SLIME_DELTA_APP_NAME": "app"}):
                with mock.patch("stitch.trainers.slime.commit_volume") as commit_volume:
                    with mock.patch("stitch.trainers.slime.discover_flash_targets", return_value=["https://c"]):
                        with mock.patch("stitch.trainers.slime.wake_targets") as wake_targets:
                            refs = publish_delta_version(args, str(version_dir), ["b.safetensors", "a.safetensors"], 1, [])

            manifest = VersionManifest.read(version_dir / "manifest.json")
            self.assertEqual(refs, [])
            self.assertEqual(read_latest(root), 1)
            self.assertEqual(manifest.version, 1)
            self.assertEqual(manifest.base_version, 0)
            self.assertEqual(manifest.transition_files, ["a.safetensors", "b.safetensors"])
            self.assertEqual(manifest.artifacts[0].kind, "transition")
            self.assertEqual(manifest.metadata["trainer"], "slime")
            commit_volume.assert_called_once_with("delta-volume")
            wake_targets.assert_called_once_with(["https://c"], 1)


if __name__ == "__main__":
    unittest.main()
