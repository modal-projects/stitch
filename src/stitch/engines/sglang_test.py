from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from stitch.engines.sglang import SGLangDiskDeltaAdapter, compose_extra_key, parse_reload_timing
from stitch.protocol import VersionManifest


class _RecordingPost:
    """Stand-in for httpx.AsyncClient that records every POST."""

    posts: list[tuple[str, dict | None]] = []
    response_json: dict = {"success": True}

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingPost":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, json=None):
        import httpx

        type(self).posts.append((url, json))
        return httpx.Response(200, json=type(self).response_json, request=httpx.Request("POST", url))


class SGLangDiskDeltaAdapterTest(unittest.TestCase):
    def test_stage_then_commit_pulls_engine_side_then_plain_reload(self) -> None:
        async def run() -> None:
            adapter = SGLangDiskDeltaAdapter(
                upstream_url="http://up/",
                local_checkpoint_dir="/local",
            )

            manifest = VersionManifest(
                version=5,
                base_version=4,
                backend="disk_delta",
                load_format="auto",
                transition_files=["model-00000-of-00001.safetensors"],
            )
            with mock.patch("httpx.AsyncClient", _RecordingPost):
                _RecordingPost.posts = []
                await adapter.stage_manifest(manifest, "/bulletin/versions/weight_v000005")
                await adapter.commit_manifest(manifest, "/bulletin/versions/weight_v000005")

            # The engine materializes the local checkpoint itself: one
            # /pull_weights against the version dir's parent (the root of
            # weight_v* dirs), then the plain disk reload — no load_format /
            # files delta payload.
            self.assertEqual(
                _RecordingPost.posts,
                [
                    (
                        "http://up/pull_weights",
                        {
                            "local_checkpoint_dir": "/local",
                            "source_dir": "/bulletin/versions",
                            "target_version": 5,
                        },
                    ),
                    (
                        "http://up/update_weights_from_disk",
                        {"model_path": "/local", "weight_version": "5", "flush_cache": False},
                    ),
                ],
            )

        asyncio.run(run())

    def test_pull_rejection_raises(self) -> None:
        async def run() -> None:
            adapter = SGLangDiskDeltaAdapter(
                upstream_url="http://up",
                local_checkpoint_dir="/local",
            )
            manifest = VersionManifest(
                version=5, base_version=4, backend="disk_delta", load_format="auto"
            )
            with mock.patch("httpx.AsyncClient", _RecordingPost):
                _RecordingPost.posts = []
                _RecordingPost.response_json = {"success": False, "message": "checksum mismatch"}
                try:
                    with self.assertRaisesRegex(RuntimeError, "rejected weight pull"):
                        await adapter.stage_manifest(manifest, "/bulletin/versions/weight_v000005")
                finally:
                    _RecordingPost.response_json = {"success": True}

        asyncio.run(run())

    def test_staged_split_reports_metrics(self) -> None:
        async def run() -> None:
            adapter = SGLangDiskDeltaAdapter(
                upstream_url="http://up",
                local_checkpoint_dir="/local",
            )
            manifest = VersionManifest(
                version=5, base_version=4, backend="disk_delta", load_format="auto"
            )

            with mock.patch("httpx.AsyncClient", _RecordingPost):
                _RecordingPost.posts = []
                stage_detail = await adapter.stage_manifest(manifest, "/bulletin/versions/weight_v000005")
                self.assertIn("engine_pull_s", stage_detail)

                _RecordingPost.response_json = {
                    "success": True,
                    "message": "Succeeded. [reload timing] iter_wait=1.50s load=4.20s postprocess=0.30s total=6.00s",
                }
                commit_detail = await adapter.commit_manifest(manifest, "/bulletin/versions/weight_v000005")
                _RecordingPost.response_json = {"success": True}

            self.assertIn("engine_reload_s", commit_detail)
            # The instrumented fork's message suffix is lifted into metrics.
            self.assertEqual(commit_detail["engine_load_s"], 4.20)
            self.assertEqual(commit_detail["engine_total_s"], 6.00)

        asyncio.run(run())


class ParseReloadTimingTest(unittest.TestCase):
    def test_parses_instrumented_message(self) -> None:
        timing = parse_reload_timing(
            "Succeeded to update model weights. [reload timing] iter_wait=12.30s load=45.60s "
            "postprocess=7.80s total=70.10s Weight version updated to 5."
        )
        self.assertEqual(
            timing,
            {
                "engine_iter_wait_s": 12.30,
                "engine_load_s": 45.60,
                "engine_postprocess_s": 7.80,
                "engine_total_s": 70.10,
            },
        )

    def test_uninstrumented_message_yields_nothing(self) -> None:
        self.assertEqual(parse_reload_timing("Succeeded to update model weights."), {})


class ExtraKeyTest(unittest.TestCase):
    def test_compose_extra_key_prefixes_version_and_run(self) -> None:
        self.assertEqual(compose_extra_key(0), "wv0;")
        self.assertEqual(compose_extra_key(7, "my-key"), "wv7;my-key")
        self.assertEqual(compose_extra_key(1, "k", run_id="run-a"), "wv1;run-a/k")


if __name__ == "__main__":
    unittest.main()
