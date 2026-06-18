from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from cookbook.slime_disagg import hooks
from stitch.protocol import VersionManifest, read_latest


class SlimeDisaggHooksTest(unittest.TestCase):
    def test_publish_delta_version_publishes_commits_and_wakes(self) -> None:
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

            with mock.patch.dict(
                os.environ, {"DELTA_VOLUME_NAME": "delta-volume", "SLIME_DELTA_APP_NAME": "app"}
            ):
                with mock.patch.object(hooks, "commit_volume") as commit_volume:
                    with mock.patch.object(hooks, "discover_flash_targets", return_value=["https://c"]):
                        with mock.patch.object(hooks, "wake_targets") as wake_targets:
                            refs = hooks.publish_delta_version(
                                args, str(version_dir), ["b.safetensors", "a.safetensors"], 1, []
                            )

            # Core publish wrote the manifest + advanced latest...
            self.assertEqual(refs, [])
            self.assertEqual(read_latest(root), 1)
            self.assertEqual(VersionManifest.read(version_dir / "manifest.json").version, 1)
            # ...and the Modal layer committed the Volume and woke the pool.
            commit_volume.assert_called_once_with("delta-volume")
            wake_targets.assert_called_once_with(["https://c"], 1)


if __name__ == "__main__":
    unittest.main()
