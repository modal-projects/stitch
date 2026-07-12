from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.frontdoor import (
    HOT_LOAD_PATH,
    advance_latest_decision,
    create_frontdoor_app,
    pool_state_from_server_infos,
)


class AdvanceDecisionTest(unittest.TestCase):
    def test_accepts_strictly_newer_same_run(self) -> None:
        self.assertEqual(
            advance_latest_decision("run-a", 4, "weight_v000005", "run-a"),
            {"run_id": "run-a", "version": 5, "reset": False},
        )

    def test_rejects_rewind_same_run(self) -> None:
        for identity in ("weight_v000004", "weight_v000003"):
            decision = advance_latest_decision("run-a", 4, identity, "run-a")
            self.assertEqual(decision["error"]["type"], "WeightRewindRejected")
            self.assertEqual(decision["error"]["current_version"], 4)

    def test_new_run_resets_even_if_version_lower(self) -> None:
        # A new run restarts at v1; accepting it after the prior run reached v5 is
        # the begin-a-new-run path, not a rewind — the idempotency fix.
        self.assertEqual(
            advance_latest_decision("run-a", 5, "weight_v000001", "run-b"),
            {"run_id": "run-b", "version": 1, "reset": True},
        )

    def test_runless_layout_stays_monotonic(self) -> None:
        ok = advance_latest_decision(None, 4, "weight_v000005", None)
        self.assertEqual(ok, {"run_id": None, "version": 5, "reset": False})
        rewind = advance_latest_decision(None, 5, "weight_v000005", None)
        self.assertEqual(rewind["error"]["type"], "WeightRewindRejected")

    def test_claim_is_a_base_version_signal_for_a_fresh_run(self) -> None:
        # An explicit claim is just weight_v000000 with a fresh run id: a
        # cross-run move, so it resets the pool to base before any delta.
        self.assertEqual(
            advance_latest_decision("run-a", 5, "weight_v000000", "run-b"),
            {"run_id": "run-b", "version": 0, "reset": True},
        )

    def test_rejects_unparseable_identity(self) -> None:
        self.assertEqual(
            advance_latest_decision(None, 0, "base", None)["error"]["type"],
            "InvalidIdentity",
        )


class PoolStateTest(unittest.TestCase):
    def test_ready_only_when_idle_and_no_error(self) -> None:
        state = pool_state_from_server_infos(
            [
                {
                    "run_id": "a",
                    "current_run_id": "run-x",
                    "current_version": 5,
                    "sync_state": "IDLE",
                    "last_sync_error": None,
                },
                {
                    "run_id": "b",
                    "current_run_id": "run-x",
                    "current_version": 4,
                    "sync_state": "PREFETCHING",
                    "last_sync_error": None,
                },
                {
                    "run_id": "c",
                    "current_run_id": "run-x",
                    "current_version": 5,
                    "sync_state": "ERROR",
                    "last_sync_error": "boom",
                },
            ]
        )
        by_id = {r.replica_id: r for r in state.replicas}
        self.assertTrue(by_id["a"].readiness)
        self.assertEqual(by_id["a"].current_snapshot_identity, "run-x/weight_v000005")
        self.assertFalse(by_id["b"].readiness)
        self.assertEqual(by_id["b"].readiness_reason, "PREFETCHING")
        self.assertFalse(by_id["c"].readiness)
        self.assertEqual(by_id["c"].readiness_reason, "boom")
        self.assertEqual(
            state.ready_count(target_snapshot_identity="run-x/weight_v000005"), 1
        )

    def test_identity_carries_run_id(self) -> None:
        # A replica on run-b advertises a run-scoped identity, so it is not
        # miscounted ready for a different run's same-numbered version.
        state = pool_state_from_server_infos(
            [
                {
                    "run_id": "a",
                    "current_run_id": "run-b",
                    "current_version": 1,
                    "sync_state": "IDLE",
                }
            ]
        )
        self.assertEqual(
            state.replicas[0].current_snapshot_identity, "run-b/weight_v000001"
        )
        self.assertEqual(
            state.ready_count(target_snapshot_identity="run-b/weight_v000001"), 1
        )
        self.assertEqual(
            state.ready_count(target_snapshot_identity="run-a/weight_v000001"), 0
        )


class FrontdoorAppTest(unittest.TestCase):
    def _client(self, *, run_id="run-a", version=5, authorize=None):
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        state = {"run_id": run_id, "version": version}
        calls: dict[str, list] = {"advanced": [], "woke": [], "proxied": []}

        async def read_current_pointer():
            return (state["run_id"], state["version"])

        async def advance_to(rid, v: int) -> None:
            calls["advanced"].append((rid, v))
            state["run_id"], state["version"] = rid, v

        async def list_server_infos():
            return [
                {
                    "run_id": "a",
                    "current_run_id": state["run_id"],
                    "current_version": state["version"],
                    "sync_state": "IDLE",
                }
            ]

        async def wake(v: int) -> None:
            calls["woke"].append(v)

        async def proxy(request, path):
            calls["proxied"].append(path)
            return JSONResponse({"proxied": path})

        app = create_frontdoor_app(
            read_current_pointer=read_current_pointer,
            advance_to=advance_to,
            list_server_infos=list_server_infos,
            proxy=proxy,
            authorize=authorize or (lambda _headers: None),
            wake=wake,
        )
        return TestClient(app), calls

    def test_post_advances_on_newer_identity_and_wakes(self) -> None:
        client, calls = self._client(run_id="run-a", version=4)
        resp = client.post(
            HOT_LOAD_PATH, json={"identity": "weight_v000005", "run_id": "run-a"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["accepted"])
        self.assertEqual(
            resp.json()["current_snapshot_identity"], "run-a/weight_v000005"
        )
        self.assertEqual(calls["advanced"], [("run-a", 5)])
        self.assertEqual(calls["woke"], [5])

    def test_post_new_run_accepted_as_reset(self) -> None:
        client, calls = self._client(run_id="run-a", version=5)
        resp = client.post(
            HOT_LOAD_PATH, json={"identity": "weight_v000001", "run_id": "run-b"}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(calls["advanced"], [("run-b", 1)])
        self.assertEqual(
            resp.json()["current_snapshot_identity"], "run-b/weight_v000001"
        )

    def test_post_rejects_rewind_without_advancing(self) -> None:
        client, calls = self._client(run_id="run-a", version=5)
        resp = client.post(
            HOT_LOAD_PATH, json={"identity": "weight_v000005", "run_id": "run-a"}
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"]["type"], "WeightRewindRejected")
        self.assertEqual(calls["advanced"], [])
        self.assertEqual(calls["woke"], [])

    def test_post_requires_identity(self) -> None:
        client, _ = self._client()
        self.assertEqual(client.post(HOT_LOAD_PATH, json={}).status_code, 400)

    def test_get_reports_pool_readiness(self) -> None:
        client, _ = self._client(run_id="run-a", version=5)
        body = client.get(HOT_LOAD_PATH).json()
        self.assertEqual(len(body["replicas"]), 1)
        self.assertEqual(
            body["replicas"][0]["current_snapshot_identity"], "run-a/weight_v000005"
        )
        self.assertTrue(body["replicas"][0]["readiness"])

    def test_catch_all_proxies(self) -> None:
        client, calls = self._client()
        resp = client.post("/v1/chat/completions", json={"model": "m"})
        self.assertEqual(resp.json(), {"proxied": "v1/chat/completions"})
        self.assertEqual(calls["proxied"], ["v1/chat/completions"])

    def test_only_explicit_inference_routes_are_proxied(self) -> None:
        client, calls = self._client()
        for path in (
            "/health",
            "/server_info",
            "/v1/internal/control",
            "/v1/%2e%2e/server_info",
            "/v1/%252e%252e/server_info",
        ):
            self.assertEqual(client.get(path).status_code, 404, path)
        self.assertEqual(calls["proxied"], [])

    def test_auth_rejection_blocks_every_route(self) -> None:
        from fastapi.responses import JSONResponse

        def deny(_headers):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        client, calls = self._client(authorize=deny)
        self.assertEqual(
            client.post(
                HOT_LOAD_PATH, json={"identity": "weight_v000009", "run_id": "run-a"}
            ).status_code,
            401,
        )
        self.assertEqual(client.get(HOT_LOAD_PATH).status_code, 401)
        self.assertEqual(client.post("/v1/chat/completions", json={}).status_code, 401)
        self.assertEqual(calls["advanced"], [])
        self.assertEqual(calls["proxied"], [])


if __name__ == "__main__":
    unittest.main()
