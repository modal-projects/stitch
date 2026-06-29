from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

import cookbook.bulletin_hooks as bulletin_hooks
from cookbook.slime_disagg import hooks


class CommitAndWakeTest(unittest.TestCase):
    def test_advances_run_pointer_commits_and_wakes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)  # transport root (Volume mount)
            disk_dir = root / "run-a"  # update_weight_disk_dir = <root>/<run_id>
            version_dir = disk_dir / "weight_v000001"
            version_dir.mkdir(parents=True)
            args = Namespace(
                update_weight_disk_dir=str(disk_dir),
                run_id="run-a",
                rollout_modal_flash_app_name="app",
                rollout_modal_flash_server_cls_name="Server",
            )

            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume"}), mock.patch.object(
                bulletin_hooks, "distributed_rank", return_value=0
            ), mock.patch.object(bulletin_hooks, "commit_volume") as commit_volume, mock.patch.object(
                bulletin_hooks, "discover_flash_targets", return_value=["https://c"]
            ), mock.patch.object(bulletin_hooks, "wake_targets") as wake_targets:
                hooks.commit_and_wake(args, str(version_dir), [])

            # The canonical pointer lives at the transport root and is self-identifying.
            self.assertEqual((root / "latest").read_text(encoding="utf-8"), "run-a/weight_v000001")
            commit_volume.assert_called_once_with("delta-volume")
            wake_targets.assert_called_once_with(["https://c"], 1)

    def test_baseline_commits_but_does_not_advance_or_wake(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disk_dir = root / "run-a"
            disk_dir.mkdir(parents=True)
            # The baseline call passes the disk-dir root, not a weight_v{N} dir.
            args = Namespace(update_weight_disk_dir=str(disk_dir), run_id="run-a")
            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume"}), mock.patch.object(
                bulletin_hooks, "distributed_rank", return_value=0
            ), mock.patch.object(bulletin_hooks, "commit_volume") as commit_volume, mock.patch.object(
                bulletin_hooks, "wake_targets"
            ) as wake_targets:
                hooks.commit_and_wake(args, str(disk_dir), [])

            self.assertFalse((root / "latest").exists())
            commit_volume.assert_called_once_with("delta-volume")
            wake_targets.assert_not_called()

    def test_non_rank_zero_commits_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            disk_dir = root / "run-a"
            (disk_dir / "weight_v000002").mkdir(parents=True)
            args = Namespace(update_weight_disk_dir=str(disk_dir), run_id="run-a")
            with mock.patch.dict(os.environ, {"DELTA_VOLUME_NAME": "delta-volume"}), mock.patch.object(
                bulletin_hooks, "distributed_rank", return_value=3
            ), mock.patch.object(bulletin_hooks, "commit_volume") as commit_volume, mock.patch.object(
                bulletin_hooks, "wake_targets"
            ) as wake_targets:
                hooks.commit_and_wake(args, str(disk_dir / "weight_v000002"), [])

            self.assertFalse((root / "latest").exists())  # only rank 0 writes latest
            commit_volume.assert_called_once_with("delta-volume")  # every rank commits shards
            wake_targets.assert_not_called()


class GateTest(unittest.TestCase):
    def test_gate_reads_version_from_run_pointer(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                disk_dir = root / "run-a"
                disk_dir.mkdir(parents=True)
                # Canonical pointer at the transport root carries the run identity;
                # the staleness gate cares only about the version within the run.
                (root / "latest").write_text("run-a/weight_v000005", encoding="utf-8")
                args = Namespace(update_weight_disk_dir=str(disk_dir), run_id="run-a")
                # Reset the module-level cache before each test.
                cache = bulletin_hooks._latest_cache
                cache.version = 0
                cache.run_id = None
                cache._refreshed_at = -1e9
                cache._board = None
                with mock.patch.dict(os.environ, {}, clear=True):
                    version = await cache.get(args)
                self.assertEqual(version, 5)
                self.assertEqual(cache.run_id, "run-a")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
