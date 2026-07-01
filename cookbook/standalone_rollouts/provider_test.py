from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cookbook.standalone_rollouts.provider import build_manager, create_app
from stitch.bulletin import FilesystemBulletinBoard
from stitch.engines.sglang import SGLangDiskDeltaAdapter
from stitch.sync import WeightSyncManager


class _FakeEngine:
    backend = "fake"

    def __init__(self) -> None:
        self.applies: list[int] = []
        self.events: list[str] = []

    async def apply_manifest(self, manifest, version_path) -> None:
        self.applies.append(manifest.version)
        self.events.append("apply")

    async def pause_generation(self) -> None:
        self.events.append("pause")

    async def continue_generation(self) -> None:
        self.events.append("continue")


def _write_slime_version(root: Path, version: int, base: int) -> None:
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


class BuildManagerTest(unittest.TestCase):
    def test_wires_slime_board_and_disk_delta_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = build_manager(
                upstream_url="http://127.0.0.1:30000/",
                transport_root=tmp,
                local_checkpoint_dir="/local",
                base_checkpoint_dir="/base",
            )

            self.assertIsInstance(manager, WeightSyncManager)
            self.assertEqual(manager.board.layout, "slime")
            self.assertEqual(str(manager.board.root), tmp)
            self.assertIsInstance(manager.engine, SGLangDiskDeltaAdapter)
            self.assertEqual(manager.engine.local_checkpoint_dir, "/local")
            self.assertEqual(manager.engine.base_checkpoint_dir, "/base")


class ProviderSyncTest(unittest.TestCase):
    def test_startup_sync_pulls_chain_from_slime_latest(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                _write_slime_version(root, 1, 0)
                _write_slime_version(root, 2, 1)
                (root / "latest").write_text("000002", encoding="utf-8")

                engine = _FakeEngine()
                manager = WeightSyncManager(
                    board=FilesystemBulletinBoard(root, layout="slime"), engine=engine
                )
                await manager.startup_sync()

                # Reconciled to the `latest` pointer: the v1..v2 tail composes
                # into a single engine apply at the target version (not one apply
                # per intermediate version) — see WeightSyncManager._sync_once.
                self.assertEqual(manager.current_version, 2)
                self.assertEqual(engine.applies, [2])

        asyncio.run(run())


class ProviderAppTest(unittest.TestCase):
    def test_create_app_serves_versioned_proxy_and_server_info(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            manager = WeightSyncManager(
                board=FilesystemBulletinBoard(tmp, layout="slime"), engine=_FakeEngine()
            )
            # poll_interval=0 disables the background reconcile task for the test.
            app = create_app(manager, upstream_url="http://127.0.0.1:9", poll_interval=0)

            with TestClient(app) as client, mock.patch("httpx.AsyncClient", _RecordingUpstream):
                _RecordingUpstream.last_json = None
                resp = client.post("/generate", json={"text": "hi"})

                self.assertEqual(resp.status_code, 200)
                # Versioned route: stamped with the composed extra_key namespace.
                self.assertEqual(_RecordingUpstream.last_json["extra_key"], "wv0;")
                self.assertEqual(resp.json()["meta_info"]["weight_version_start"], 0)
                self.assertEqual(client.get("/server_info").json()["current_version"], 0)


class _RecordingUpstream:
    last_json: dict | None = None
    last_url: str | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingUpstream":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    async def request(self, method, url, **kwargs):
        import httpx

        type(self).last_url = url
        type(self).last_json = kwargs.get("json")
        return httpx.Response(
            200,
            json={"text": "ok", "meta_info": {"finish_reason": {"type": "length"}}},
            request=httpx.Request(method, url),
        )


if __name__ == "__main__":
    unittest.main()
