from __future__ import annotations

import asyncio
import unittest

from cookbook.standalone_rollouts.delta_view import (
    DeltaIndexError,
    DerivedDeltaConflict,
)
from cookbook.standalone_rollouts.ledger import IdentityLedger
from cookbook.standalone_rollouts.opaque_frontdoor import (
    HOT_LOAD_PATH,
    create_opaque_frontdoor_app,
)


class FrontdoorHarness:
    def __init__(self, *, authorize=None) -> None:
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        self.events: list[tuple] = []
        self.pointer = 0
        self.fail_derive: Exception | None = None
        self.fail_save: BaseException | None = None
        self.fail_advance = False
        self.fail_wake = False
        self.infos = [{"replica_id": "r1", "current_version": 0, "sync_state": "IDLE"}]
        ledger = IdentityLedger.new("base")
        self.persisted_ledger = ledger.to_dict()

        async def save_ledger(data) -> None:
            self.events.append(("save", data))
            self.persisted_ledger = data
            if self.fail_save is not None:
                error, self.fail_save = self.fail_save, None
                raise error

        async def derive_delta(entry, *, committed: bool) -> None:
            self.events.append(("derive", entry.identity, committed))
            if self.fail_derive is not None:
                error, self.fail_derive = self.fail_derive, None
                raise error

        async def advance_to(version: int) -> None:
            self.events.append(("advance", version))
            if self.fail_advance:
                self.fail_advance = False
                raise OSError("advance failed")
            self.pointer = version

        async def list_server_infos():
            return self.infos

        async def proxy(_request, path):
            self.events.append(("proxy", path))
            return JSONResponse({"proxied": path})

        async def wake(version: int) -> None:
            self.events.append(("wake", version))
            if self.fail_wake:
                self.fail_wake = False
                raise OSError("wake failed")

        app = create_opaque_frontdoor_app(
            ledger=ledger,
            save_ledger=save_ledger,
            derive_delta=derive_delta,
            advance_to=advance_to,
            list_server_infos=list_server_infos,
            proxy=proxy,
            authorize=authorize or (lambda _headers: None),
            wake=wake,
        )
        self.client = TestClient(app, raise_server_exceptions=False)

    def post_delta(
        self,
        identity: str = "delta-a",
        previous: str = "base",
        *,
        checksum: str = "xxh3-128",
    ):
        return self.client.post(
            HOT_LOAD_PATH,
            json={
                "identity": identity,
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": previous,
                    "compression_format": "zstd",
                    "checksum_format": checksum,
                },
                "reset_prompt_cache": "new_session",
            },
        )


class FrontdoorAppTest(unittest.TestCase):
    def test_new_delta_orders_derive_save_advance_then_wake(self) -> None:
        harness = FrontdoorHarness()
        response = harness.post_delta()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["current_snapshot_identity"], "delta-a")
        self.assertEqual(response.json()["version"], 1)
        self.assertEqual(
            [event[0] for event in harness.events],
            ["derive", "save", "advance", "wake"],
        )
        self.assertEqual(harness.events[0], ("derive", "delta-a", False))

    def test_exact_head_retry_revalidates_repairs_pointer_and_wakes(self) -> None:
        harness = FrontdoorHarness()
        self.assertEqual(harness.post_delta().status_code, 200)
        harness.events.clear()
        response = harness.post_delta()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["already_current"])
        self.assertEqual(
            harness.events,
            [("derive", "delta-a", True), ("advance", 1), ("wake", 1)],
        )

    def test_pointer_failure_converges_on_exact_retry(self) -> None:
        harness = FrontdoorHarness()
        harness.fail_advance = True
        self.assertEqual(harness.post_delta().status_code, 500)
        self.assertEqual(harness.pointer, 0)
        harness.events.clear()

        self.assertEqual(harness.post_delta().status_code, 200)
        self.assertEqual(harness.pointer, 1)
        self.assertEqual(
            harness.events,
            [("derive", "delta-a", True), ("advance", 1), ("wake", 1)],
        )

    def test_ambiguous_save_failure_blocks_writes_until_recovery(self) -> None:
        harness = FrontdoorHarness()
        harness.fail_save = OSError("save outcome unknown")
        self.assertEqual(harness.post_delta().status_code, 500)
        persisted = IdentityLedger.from_dict(
            harness.persisted_ledger, expected_base_identity="base"
        )
        self.assertEqual(persisted.head.identity, "delta-a")
        harness.events.clear()

        response = harness.post_delta()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["type"], "LedgerStateUncertain")
        self.assertEqual(harness.events, [])

    def test_cancelled_save_also_blocks_writes_until_recovery(self) -> None:
        harness = FrontdoorHarness()
        harness.fail_save = asyncio.CancelledError()
        try:
            harness.post_delta()
        except BaseException:
            pass
        harness.events.clear()

        response = harness.post_delta()
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["type"], "LedgerStateUncertain")
        self.assertEqual(harness.events, [])

    def test_customer_failures_do_not_advance_and_storage_failures_remain_5xx(
        self,
    ) -> None:
        failures = (
            (FileNotFoundError("upload missing"), 409),
            (DeltaIndexError("bad index"), 400),
            (DerivedDeltaConflict("changed upload"), 409),
        )
        for failure, expected in failures:
            with self.subTest(failure=failure):
                harness = FrontdoorHarness()
                harness.fail_derive = failure
                self.assertEqual(harness.post_delta().status_code, expected)
                self.assertEqual(harness.pointer, 0)
                self.assertNotIn("advance", [event[0] for event in harness.events])

        harness = FrontdoorHarness()
        harness.fail_save = FileNotFoundError("ledger mount unavailable")
        self.assertEqual(harness.post_delta().status_code, 500)
        self.assertEqual(harness.pointer, 0)

    def test_base_forks_contradictory_retries_and_rewinds_are_rejected(self) -> None:
        harness = FrontdoorHarness()
        self.assertEqual(
            harness.client.post(HOT_LOAD_PATH, json={"identity": "base"}).status_code,
            200,
        )
        self.assertEqual(
            harness.client.post(HOT_LOAD_PATH, json={"identity": "other"}).status_code,
            409,
        )
        self.assertEqual(harness.post_delta().status_code, 200)
        self.assertEqual(
            harness.post_delta("delta-a", "base", checksum="blake3").status_code,
            409,
        )
        self.assertEqual(harness.post_delta("delta-b", "base").status_code, 409)
        self.assertEqual(harness.post_delta("delta-b", "delta-a").status_code, 200)
        self.assertEqual(harness.post_delta("delta-a", "base").status_code, 409)
        self.assertEqual(
            harness.client.post(HOT_LOAD_PATH, json={"identity": "base"}).status_code,
            409,
        )

    def test_wake_is_best_effort_and_get_uses_committed_ledger(self) -> None:
        harness = FrontdoorHarness()
        harness.fail_wake = True
        self.assertEqual(harness.post_delta().status_code, 200)
        harness.infos = [
            {"replica_id": "r1", "current_version": 1, "sync_state": "IDLE"}
        ]
        body = harness.client.get(HOT_LOAD_PATH).json()
        self.assertEqual(body["replicas"][0]["current_snapshot_identity"], "delta-a")
        self.assertTrue(body["replicas"][0]["readiness"])

    def test_closed_surface_and_auth(self) -> None:
        from fastapi.responses import JSONResponse

        harness = FrontdoorHarness()
        response = harness.client.post("/v1/chat/completions", json={})
        self.assertEqual(response.json(), {"proxied": "v1/chat/completions"})
        for path in (
            "/health",
            "/server_info",
            "/docs",
            "/openapi.json",
            "/v1/internal/control",
            "/v1/%2e%2e/server_info",
        ):
            self.assertEqual(harness.client.get(path).status_code, 404, path)

        denied = FrontdoorHarness(
            authorize=lambda _headers: JSONResponse(
                {"error": "unauthorized"}, status_code=401
            )
        )
        self.assertEqual(denied.post_delta().status_code, 401)
        self.assertEqual(denied.client.get(HOT_LOAD_PATH).status_code, 401)
        self.assertEqual(
            denied.client.post("/v1/chat/completions", json={}).status_code, 401
        )
        self.assertEqual(denied.events, [])


if __name__ == "__main__":
    unittest.main()
