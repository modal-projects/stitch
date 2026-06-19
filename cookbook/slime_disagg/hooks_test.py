from __future__ import annotations

import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from cookbook.slime_disagg import hooks


class CommitAndWakeTest(unittest.TestCase):
    def test_advances_latest_commits_and_wakes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            version_dir = root / "weight_v000001"
            version_dir.mkdir(parents=True)
            args = Namespace(
                update_weight_disk_dir=str(root),
                rollout_modal_flash_app_name="app",
                rollout_modal_flash_server_cls_name="Server",
            )

            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume"}), mock.patch.object(
                hooks, "_distributed_rank", return_value=0
            ), mock.patch.object(hooks, "commit_volume") as commit_volume, mock.patch.object(
                hooks, "discover_flash_targets", return_value=["https://c"]
            ), mock.patch.object(hooks, "wake_targets") as wake_targets:
                hooks.commit_and_wake(args, str(version_dir), [])

            # latest advanced (raw slime-layout pointer), committed before the wake.
            self.assertEqual((root / "latest").read_text(encoding="utf-8"), "000001")
            commit_volume.assert_called_once_with("delta-volume")
            wake_targets.assert_called_once_with(["https://c"], 1)

    def test_baseline_commits_but_does_not_advance_or_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # The baseline call passes the disk-dir root, not a weight_v{N} dir.
            args = Namespace(update_weight_disk_dir=str(root))
            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume"}), mock.patch.object(
                hooks, "_distributed_rank", return_value=0
            ), mock.patch.object(hooks, "commit_volume") as commit_volume, mock.patch.object(
                hooks, "wake_targets"
            ) as wake_targets:
                hooks.commit_and_wake(args, str(root), [])

            self.assertFalse((root / "latest").exists())
            commit_volume.assert_called_once_with("delta-volume")
            wake_targets.assert_not_called()

    def test_non_rank_zero_commits_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "weight_v000002").mkdir(parents=True)
            args = Namespace(update_weight_disk_dir=str(root))
            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume"}), mock.patch.object(
                hooks, "_distributed_rank", return_value=3
            ), mock.patch.object(hooks, "commit_volume") as commit_volume, mock.patch.object(
                hooks, "wake_targets"
            ) as wake_targets:
                hooks.commit_and_wake(args, str(root / "weight_v000002"), [])

            self.assertFalse((root / "latest").exists())  # only rank 0 writes latest
            commit_volume.assert_called_once_with("delta-volume")  # every rank commits shards
            wake_targets.assert_not_called()


if __name__ == "__main__":
    unittest.main()
