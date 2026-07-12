from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.base_checkpoint import (
    is_hf_repo_id,
    resolve_base_checkpoint,
)


class BaseCheckpointTest(unittest.TestCase):
    def test_absolute_dir_is_used_without_downloading(self) -> None:
        calls: list[tuple] = []

        def fake_download(*args, **kwargs):
            calls.append((args, kwargs))
            return "/wrong"

        resolved = resolve_base_checkpoint(
            "/mnt/transport/customer-base", snapshot_download=fake_download
        )
        self.assertEqual(resolved, "/mnt/transport/customer-base")
        self.assertEqual(calls, [])
        self.assertFalse(is_hf_repo_id(resolved))

    def test_repo_id_resolves_from_local_cache(self) -> None:
        calls: list[tuple] = []

        def fake_download(spec, **kwargs):
            calls.append((spec, kwargs))
            return "/hf-cache/snapshot"

        resolved = resolve_base_checkpoint(
            "moonshotai/Moonlight-16B-A3B-Instruct",
            snapshot_download=fake_download,
        )
        self.assertEqual(resolved, "/hf-cache/snapshot")
        self.assertEqual(
            calls,
            [("moonshotai/Moonlight-16B-A3B-Instruct", {"local_files_only": True})],
        )


if __name__ == "__main__":
    unittest.main()
