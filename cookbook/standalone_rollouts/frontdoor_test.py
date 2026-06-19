from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.frontdoor import (
    HOT_LOAD_PATH,
    advance_latest_decision,
    create_frontdoor_app,
    pool_state_from_server_infos,
)


class AdvanceDecisionTest(unittest.TestCase):
    def test_accepts_strictly_newer_version(self) -> None:
        self.assertEqual(advance_latest_decision(4, "weight_v000005"), {"version": 5})

    def test_rejects_rewind_to_equal_or_older(self) -> None:
        for identity in ("weight_v000004", "weight_v000003"):
            decision = advance_latest_decision(4, identity)
            self.assertEqual(decision["error"]["type"], "WeightRewindRejected")
            self.assertEqual(decision["error"]["current_version"], 4)

    def test_rejects_unparseable_identity(self) -> None:
        self.assertEqual(advance_latest_decision(0, "base")["error"]["type"], "InvalidIdentity")


class PoolStateTest(unittest.TestCase):
    def test_ready_only_when_idle_and_no_error(self) -> None:
        state = pool_state_from_server_infos(
            [
                {"run_id": "a", "current_version": 5, "sync_state": "IDLE", "last_sync_error": None},
                {"run_id": "b", "current_version": 4, "sync_state": "PREFETCHING", "last_sync_error": None},
                {"run_id": "c", "current_version": 5, "sync_state": "ERROR", "last_sync_error": "boom"},
            ]
        )
        by_id = {r.replica_id: r for r in state.replicas}
        self.assertTrue(by_id["a"].readiness)
        self.assertEqual(by_id["a"].current_snapshot_identity, "weight_v000005")
        self.assertFalse(by_id["b"].readiness)
        self.assertEqual(by_id["b"].readiness_reason, "PREFETCHING")
        self.assertFalse(by_id["c"].readiness)
        self.assertEqual(by_id["c"].readiness_reason, "boom")
        # Two replicas serve v5, but only the idle one is ready for the target.
        self.assertEqual(state.ready_count(target_version=5), 1)


class FrontdoorAppTest(unittest.TestCase):
    def _client(self, *, version=5, authorize=None):
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        state = {"version": version}
        calls: dict[str, list] = {"advanced": [], "woke": [], "proxied": []}

        async def read_current_version() -> int:
            return state["version"]

        async def advance_to(v: int) -> None:
            calls["advanced"].append(v)
            state["version"] = v

        async def list_server_infos():
            return [{"run_id": "a", "current_version": state["version"], "sync_state": "IDLE"}]

        async def wake(v: int) -> None:
            calls["woke"].append(v)

        async def proxy(request, path):
            calls["proxied"].append(path)
            return JSONResponse({"proxied": path})

        app = create_frontdoor_app(
            read_current_version=read_current_version,
            advance_to=advance_to,
            list_server_infos=list_server_infos,
            proxy=proxy,
            authorize=authorize,
            wake=wake,
        )
        return TestClient(app), calls

    def test_post_advances_on_newer_identity_and_wakes(self) -> None:
        client, calls = self._client(version=4)
        resp = client.post(HOT_LOAD_PATH, json={"identity": "weight_v000005"})
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["accepted"])
        self.assertEqual(calls["advanced"], [5])
        self.assertEqual(calls["woke"], [5])

    def test_post_rejects_rewind_without_advancing(self) -> None:
        client, calls = self._client(version=5)
        resp = client.post(HOT_LOAD_PATH, json={"identity": "weight_v000005"})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"]["type"], "WeightRewindRejected")
        self.assertEqual(calls["advanced"], [])
        self.assertEqual(calls["woke"], [])

    def test_post_requires_identity(self) -> None:
        client, _ = self._client()
        self.assertEqual(client.post(HOT_LOAD_PATH, json={}).status_code, 400)

    def test_get_reports_pool_readiness(self) -> None:
        client, _ = self._client(version=5)
        body = client.get(HOT_LOAD_PATH).json()
        self.assertEqual(len(body["replicas"]), 1)
        self.assertEqual(body["replicas"][0]["current_snapshot_identity"], "weight_v000005")
        self.assertTrue(body["replicas"][0]["readiness"])

    def test_catch_all_proxies(self) -> None:
        client, calls = self._client()
        resp = client.post("/v1/chat/completions", json={"model": "m"})
        self.assertEqual(resp.json(), {"proxied": "v1/chat/completions"})
        self.assertEqual(calls["proxied"], ["v1/chat/completions"])

    def test_auth_rejection_blocks_every_route(self) -> None:
        from fastapi.responses import JSONResponse

        def deny(_headers):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        client, calls = self._client(authorize=deny)
        self.assertEqual(client.post(HOT_LOAD_PATH, json={"identity": "weight_v000009"}).status_code, 401)
        self.assertEqual(client.get(HOT_LOAD_PATH).status_code, 401)
        self.assertEqual(client.post("/v1/chat/completions", json={}).status_code, 401)
        self.assertEqual(calls["advanced"], [])
        self.assertEqual(calls["proxied"], [])


if __name__ == "__main__":
    unittest.main()
