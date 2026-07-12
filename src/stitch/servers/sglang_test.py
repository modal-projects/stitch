from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest import mock

from stitch.bulletin import FilesystemBulletinBoard
from stitch.sync import WeightSyncManager


class FakeEngine:
    backend = "fake"

    def __init__(self) -> None:
        self.events: list[str] = []

    async def flush_cache(self) -> None:
        self.events.append("flush")

    async def apply_manifest(self, manifest, version_path) -> None:
        self.events.append("apply")

    async def pause_generation(self) -> None:
        self.events.append("pause")

    async def continue_generation(self) -> None:
        self.events.append("continue")


class ReconcileLoopTest(unittest.TestCase):
    def test_refresh_failure_marks_replica_unready_until_recovery(self) -> None:
        # A replica whose board refresh keeps failing cannot follow the log; it
        # must surface that in last_sync_error (unready), not just a warning log.
        import time

        from fastapi.testclient import TestClient

        from stitch.servers.sglang import create_app

        with tempfile.TemporaryDirectory() as tmp:
            broken = {"on": True}

            def refresh() -> None:
                if broken["on"]:
                    raise RuntimeError("mount hiccup")

            board = FilesystemBulletinBoard(tmp, layout="slime", refresh=refresh)
            manager = WeightSyncManager(board=board, engine=FakeEngine())
            app = create_app(
                manager, upstream_url="http://127.0.0.1:9", background_sync_interval=0.01
            )
            with TestClient(app) as client:
                deadline = time.time() + 5
                while time.time() < deadline and not str(
                    manager.last_sync_error or ""
                ).startswith("background reconcile failed"):
                    time.sleep(0.02)
                info = client.get("/server_info").json()
                self.assertIn("background reconcile failed", info["last_sync_error"] or "")

                broken["on"] = False  # recovery clears the sticky error
                deadline = time.time() + 5
                while time.time() < deadline and manager.last_sync_error:
                    time.sleep(0.02)
                self.assertIsNone(manager.last_sync_error)


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
                for route in (
                    "update_weights_from_disk",
                    "flush_cache",
                    "pause_generation",
                    "abort_request",
                ):
                    resp = client.post(f"/{route}", json={})
                    self.assertEqual(resp.status_code, 403, route)
                    self.assertEqual(resp.json()["error"]["type"], "RouteBlocked")

    def test_health_and_server_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _manager, client = self._client(tmp)
            with client:
                self.assertEqual(
                    client.get("/health").json(), {"ok": True, "current_version": 0}
                )
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


class _RecordingUpstream:
    """Stand-in for httpx.AsyncClient that records the forwarded payload.

    Fully synchronous (no suspension points) so the proxy's upstream task
    completes before its disconnect watcher can observe the test client's
    immediate ASGI disconnect.
    """

    last_json: dict | None = None
    last_url: str | None = None
    last_headers: dict | None = None

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
        type(self).last_headers = kwargs.get("headers")
        return httpx.Response(
            200,
            json={"text": "ok", "meta_info": {"finish_reason": {"type": "length"}}},
            request=httpx.Request(method, url),
        )


class _BlockingUpstream:
    """Fake upstream whose 'slow' generations block until released, recording
    every forwarded payload in order."""

    calls: list[dict] = []
    release: "asyncio.Event"

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_BlockingUpstream":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def aclose(self) -> None:
        return None

    async def request(self, method, url, **kwargs):
        import httpx

        payload = kwargs.get("json") or {}
        type(self).calls.append(payload)
        if payload.get("text") == "slow":
            await type(self).release.wait()
        return httpx.Response(
            200,
            json={"text": "ok", "meta_info": {}},
            request=httpx.Request(method, url),
        )


class SidecarInPlaceCommitTest(unittest.TestCase):
    def test_request_crossing_in_place_commit_is_stamped_start_end(self) -> None:
        """End-to-end through the sidecar app: an in-flight request crosses an
        in-place bulletin-board commit without being drained, finishes with
        (start=0, end=1) metadata, and subsequent requests land in the new
        extra_key namespace."""

        async def run() -> None:
            import httpx
            from httpx import ASGITransport

            from stitch.protocol import VersionManifest
            from stitch.servers.sglang import create_app

            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                engine = FakeEngine()
                manager = WeightSyncManager(
                    board=board, engine=engine, commit_mode="in_place"
                )
                app = create_app(manager, upstream_url="http://127.0.0.1:9")

                _BlockingUpstream.calls = []
                _BlockingUpstream.release = asyncio.Event()

                async def wait_for(predicate, timeout: float = 5.0) -> None:
                    deadline = asyncio.get_running_loop().time() + timeout
                    while not predicate():
                        if asyncio.get_running_loop().time() > deadline:
                            raise AssertionError("timed out waiting for condition")
                        await asyncio.sleep(0.01)

                driver = httpx.AsyncClient(
                    transport=ASGITransport(app=app), base_url="http://sidecar"
                )
                async with driver:
                    with mock.patch("httpx.AsyncClient", _BlockingUpstream):
                        slow = asyncio.create_task(
                            driver.post("/generate", json={"text": "slow"})
                        )
                        await wait_for(lambda: manager.active_requests == 1)
                        self.assertEqual(
                            _BlockingUpstream.calls[0]["extra_key"], "wv0;"
                        )

                        board.publish_manifest(
                            VersionManifest(
                                version=1,
                                base_version=0,
                                backend="fake",
                                load_format="noop",
                            )
                        )
                        rpc = await driver.post(
                            "/rpc_sync_from_bulletin_board", json={"target_version": 1}
                        )
                        self.assertTrue(rpc.json()["accepted"])

                        # The commit lands while the request is still in flight:
                        # in_place mode must not drain non-strict traffic.
                        await wait_for(lambda: manager.current_version == 1)
                        self.assertEqual(manager.active_requests, 1)
                        self.assertEqual(engine.events, ["pause", "apply", "continue"])

                        _BlockingUpstream.release.set()
                        meta = (await slow).json()["meta_info"]
                        self.assertEqual(meta["weight_version_start"], 0)
                        self.assertEqual(meta["weight_version_end"], 1)

                        # New admissions are stamped with the new namespace.
                        await driver.post("/generate", json={"text": "hi"})
                        self.assertEqual(
                            _BlockingUpstream.calls[-1]["extra_key"], "wv1;"
                        )

        asyncio.run(run())


class SidecarStampingTest(unittest.TestCase):
    def _client(self, tmp: str):
        from fastapi.testclient import TestClient

        from stitch.servers.sglang import create_app

        board = FilesystemBulletinBoard(tmp)
        manager = WeightSyncManager(board=board, engine=FakeEngine())
        app = create_app(manager, upstream_url="http://127.0.0.1:9")
        return manager, TestClient(app)

    def test_generate_payload_is_stamped_with_composed_extra_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _RecordingUpstream.last_json = None
            manager, client = self._client(tmp)
            with client, mock.patch("httpx.AsyncClient", _RecordingUpstream):
                resp = client.post("/generate", json={"text": "hi"})
                self.assertEqual(resp.status_code, 200)
                forwarded = _RecordingUpstream.last_json
                self.assertEqual(forwarded["extra_key"], "wv0;")
                self.assertIn("rid", forwarded)
                meta = resp.json()["meta_info"]
                self.assertEqual(meta["weight_version_start"], 0)
                self.assertEqual(meta["weight_version_end"], 0)

    def test_stamping_composes_user_extra_keys_and_lists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, client = self._client(tmp)
            manager.current_version = 3
            with client, mock.patch("httpx.AsyncClient", _RecordingUpstream):
                client.post("/generate", json={"text": "hi", "extra_key": "user-key"})
                self.assertEqual(
                    _RecordingUpstream.last_json["extra_key"], "wv3;user-key"
                )

                client.post(
                    "/generate", json={"text": ["a", "b"], "extra_key": ["k1", "k2"]}
                )
                self.assertEqual(
                    _RecordingUpstream.last_json["extra_key"], ["wv3;k1", "wv3;k2"]
                )

                client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "m",
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
                self.assertEqual(_RecordingUpstream.last_json["extra_key"], "wv3;")

    def test_unversioned_routes_are_not_stamped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager, client = self._client(tmp)
            with client, mock.patch("httpx.AsyncClient", _RecordingUpstream):
                _RecordingUpstream.last_json = None
                # /v1/completions is not in the default versioned route set, so
                # it must be neither stamped (request) nor version-annotated
                # (response). Previously the response-metadata block hardcoded
                # this path and injected version fields onto an ungated request.
                resp = client.post("/v1/completions", json={"model": "m", "prompt": "hi"})
                self.assertNotIn("extra_key", _RecordingUpstream.last_json or {})
                self.assertNotIn("weight_version_start", resp.json())
                self.assertNotIn("weight_version_end", resp.json())


class _CountingUpstream(_RecordingUpstream):
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        type(self).instances += 1


class SidecarClientPoolTest(unittest.TestCase):
    def test_upstream_client_is_pooled_across_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            from fastapi.testclient import TestClient

            from stitch.servers.sglang import create_app

            board = FilesystemBulletinBoard(tmp)
            manager = WeightSyncManager(board=board, engine=FakeEngine())
            app = create_app(manager, upstream_url="http://127.0.0.1:9")

            _CountingUpstream.instances = 0
            with TestClient(app) as client, mock.patch("httpx.AsyncClient", _CountingUpstream):
                for _ in range(3):
                    self.assertEqual(client.post("/generate", json={"text": "hi"}).status_code, 200)
            # One pooled client served all three requests, plus it was closed
            # on shutdown (no per-request construct/teardown).
            self.assertEqual(_CountingUpstream.instances, 1)


if __name__ == "__main__":
    unittest.main()
