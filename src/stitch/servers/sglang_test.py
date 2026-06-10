from __future__ import annotations

import tempfile
import unittest

from stitch.bulletin import FilesystemBulletinBoard
from stitch.sync import WeightSyncManager


class FakeEngine:
    backend = "fake"

    async def flush_cache(self) -> None:
        pass

    async def apply_manifest(self, manifest, version_path) -> None:
        pass


class SidecarProxyTest(unittest.TestCase):
    def _client(self, tmp: str):
        from fastapi.testclient import TestClient

        from stitch.servers.sglang import create_app

        board = FilesystemBulletinBoard(tmp)
        manager = WeightSyncManager(board=board, engine=FakeEngine())
        app = create_app(manager, upstream_url="http://127.0.0.1:9")
        return manager, TestClient(app)

    def test_engine_control_routes_are_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _manager, client = self._client(tmp)
            with client:
                for route in ("update_weights_from_disk", "flush_cache", "pause_generation", "abort_request"):
                    resp = client.post(f"/{route}", json={})
                    self.assertEqual(resp.status_code, 403, route)
                    self.assertEqual(resp.json()["error"]["type"], "RouteBlocked")

    def test_health_and_server_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _manager, client = self._client(tmp)
            with client:
                self.assertEqual(client.get("/health").json(), {"ok": True, "current_version": 0})
                info = client.get("/server_info").json()
                self.assertEqual(info["current_version"], 0)
                self.assertEqual(info["sync_state"], "IDLE")

    def test_version_policy_rejections_are_409(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, client = self._client(tmp)
            with client:
                manager.current_version = 2
                resp = client.post(
                    "/generate",
                    json={"text": "hi", "weight_version": {"exact_version": 1}},
                )
                self.assertEqual(resp.status_code, 409)
                self.assertEqual(resp.json()["error"]["type"], "WeightVersionTooOld")


if __name__ == "__main__":
    unittest.main()
