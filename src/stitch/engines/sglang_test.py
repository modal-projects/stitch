from __future__ import annotations

import asyncio
import unittest
from unittest import mock

from stitch.engines.sglang import SGLangDiskDeltaAdapter, compose_extra_key
from stitch.protocol import VersionManifest


class _RecordingPost:
    """Stand-in for httpx.AsyncClient that records the reload POST."""

    last_url: str | None = None
    last_json: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingPost":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def post(self, url, json=None):
        import httpx

        type(self).last_url = url
        type(self).last_json = json
        return httpx.Response(200, json={"success": True}, request=httpx.Request("POST", url))


class SGLangDiskDeltaAdapterTest(unittest.TestCase):
    def test_apply_manifest_applies_host_side_then_plain_reload(self) -> None:
        async def run() -> None:
            calls: dict[str, list] = {"apply": [], "init": []}

            adapter = SGLangDiskDeltaAdapter(
                upstream_url="http://up/",
                local_checkpoint_dir="/local",
                base_checkpoint_dir="/base",
                apply_deltas=lambda local, root, version: calls["apply"].append((local, root, version)),
                init_local_checkpoint=lambda local, base: calls["init"].append((local, base)),
            )

            await adapter.prepare()
            manifest = VersionManifest(
                version=5, base_version=4, backend="disk_delta", load_format="auto"
            )
            with mock.patch("httpx.AsyncClient", _RecordingPost):
                _RecordingPost.last_json = None
                await adapter.apply_manifest(manifest, "/bulletin/versions/weight_v000005")

            # Base materialized once; delta chain applied against the version
            # dir's parent (the root of weight_v* dirs) up to version 5.
            self.assertEqual(calls["init"], [("/local", "/base")])
            self.assertEqual(calls["apply"], [("/local", "/bulletin/versions", 5)])
            # Engine reload is the plain disk path: local checkpoint, no
            # load_format / files delta payload.
            self.assertEqual(_RecordingPost.last_url, "http://up/update_weights_from_disk")
            self.assertEqual(
                _RecordingPost.last_json,
                {"model_path": "/local", "weight_version": "5", "flush_cache": False},
            )

        asyncio.run(run())


class ExtraKeyTest(unittest.TestCase):
    def test_compose_extra_key_prefixes_version_and_run(self) -> None:
        self.assertEqual(compose_extra_key(0), "wv0;")
        self.assertEqual(compose_extra_key(7, "my-key"), "wv7;my-key")
        self.assertEqual(compose_extra_key(1, "k", run_id="run-a"), "wv1;run-a/k")


if __name__ == "__main__":
    unittest.main()
