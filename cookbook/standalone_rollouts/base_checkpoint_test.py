from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.base_checkpoint import (
    is_hf_repo_id,
    resolve_base_checkpoint,
)


class IsHfRepoIdTest(unittest.TestCase):
    def test_repo_id_is_not_an_absolute_path(self) -> None:
        self.assertTrue(is_hf_repo_id("moonshotai/Moonlight-16B-A3B-Instruct"))
        self.assertTrue(is_hf_repo_id("zai-org/GLM-4.5-Air-FP8"))

    def test_absolute_dir_is_not_a_repo_id(self) -> None:
        self.assertFalse(is_hf_repo_id("/mnt/stitch-s3-transport/base"))
        self.assertFalse(is_hf_repo_id("/prep/kimi-nvfp4"))


class ResolveBaseCheckpointTest(unittest.TestCase):
    def test_absolute_dir_is_returned_verbatim_without_touching_the_cache(self) -> None:
        calls: list[tuple] = []

        def fake_download(*args, **kwargs):
            calls.append((args, kwargs))
            return "/should/not/be/used"

        resolved = resolve_base_checkpoint(
            "/mnt/stitch-s3-transport/customer-base", snapshot_download=fake_download
        )
        self.assertEqual(resolved, "/mnt/stitch-s3-transport/customer-base")
        self.assertEqual(calls, [])

    def test_repo_id_resolves_from_the_local_cache_only(self) -> None:
        calls: list[tuple] = []

        def fake_download(spec, **kwargs):
            calls.append((spec, kwargs))
            return f"/hf-cache/{spec}/snapshots/abc123"

        resolved = resolve_base_checkpoint(
            "moonshotai/Moonlight-16B-A3B-Instruct", snapshot_download=fake_download
        )
        self.assertEqual(resolved, "/hf-cache/moonshotai/Moonlight-16B-A3B-Instruct/snapshots/abc123")
        self.assertEqual(
            calls, [("moonshotai/Moonlight-16B-A3B-Instruct", {"local_files_only": True})]
        )


if __name__ == "__main__":
    unittest.main()
