from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.frontdoor import (
    HOT_LOAD_PATH,
    delta_index_metadata,
    is_customer_inference_route,
    is_valid_identity,
    pool_state_from_server_infos,
)
from cookbook.standalone_rollouts.ledger import IdentityLedger


class IdentityValidationTest(unittest.TestCase):
    def test_opaque_identities_are_accepted(self) -> None:
        for identity in ("weight_v000001", "ckpt-100", "step_500", "2026-07-11T00:00:00Z", "wéight"):
            self.assertTrue(is_valid_identity(identity), identity)

    def test_rejects_empty_slashed_control_and_overlong(self) -> None:
        self.assertFalse(is_valid_identity(""))
        self.assertFalse(is_valid_identity("a/b"))  # would escape the prefix
        self.assertFalse(is_valid_identity("a\nb"))  # control char
        self.assertFalse(is_valid_identity("x" * 513))

    def test_rejects_traversal_and_reserved_names(self) -> None:
        for identity in (".", "..", "latest", "identities.json"):
            self.assertFalse(is_valid_identity(identity), identity)


class DeltaMetadataTest(unittest.TestCase):
    def test_pads_versions_and_defaults_delta_encoding(self) -> None:
        meta = delta_index_metadata(
            2, 1, {"compression_format": "zstd", "checksum_format": "adler32"}
        )
        self.assertEqual(
            meta,
            {
                "version": "000002",
                "base_version": "000001",
                "delta_encoding": "xor",  # not in the customer API; spec §3 default
                "compression_format": "zstd",
                "checksum_format": "adler32",
            },
        )

    def test_passes_through_customer_formats(self) -> None:
        meta = delta_index_metadata(
            5, 4, {"compression_format": "zstd", "checksum_format": "xxh3-128", "delta_encoding": "overwrite"}
        )
        self.assertEqual(meta["checksum_format"], "xxh3-128")
        self.assertEqual(meta["delta_encoding"], "overwrite")


class PoolStateTest(unittest.TestCase):
    def _ledger(self) -> IdentityLedger:
        ledger = IdentityLedger()
        ledger.record("base-ckpt", previous=None)  # v0
        ledger.record("ckpt-100", previous="base-ckpt")  # v1
        return ledger

    def test_translates_version_to_customer_identity(self) -> None:
        state = pool_state_from_server_infos(
            [{"current_version": 1, "sync_state": "IDLE", "last_sync_error": None, "replica_id": "a"}],
            self._ledger(),
        )
        self.assertTrue(state.replicas[0].readiness)
        self.assertEqual(state.replicas[0].current_snapshot_identity, "ckpt-100")
        self.assertEqual(state.ready_count(target_snapshot_identity="ckpt-100"), 1)

    def test_ready_only_when_idle_and_no_error(self) -> None:
        state = pool_state_from_server_infos(
            [
                {"current_version": 1, "sync_state": "IDLE", "last_sync_error": None, "replica_id": "a"},
                {"current_version": 0, "sync_state": "PREFETCHING", "last_sync_error": None, "replica_id": "b"},
                {"current_version": 1, "sync_state": "ERROR", "last_sync_error": "boom", "replica_id": "c"},
            ],
            self._ledger(),
        )
        by_id = {r.replica_id: r for r in state.replicas}
        self.assertTrue(by_id["a"].readiness)
        self.assertFalse(by_id["b"].readiness)
        self.assertEqual(by_id["b"].readiness_reason, "PREFETCHING")
        self.assertFalse(by_id["c"].readiness)
        self.assertEqual(by_id["c"].readiness_reason, "boom")

    def test_unreachable_replica_reason(self) -> None:
        state = pool_state_from_server_infos(
            [{"sync_state": None, "last_sync_error": "unreachable"}], self._ledger()
        )
        self.assertFalse(state.replicas[0].readiness)
        self.assertEqual(state.replicas[0].readiness_reason, "unreachable")
        self.assertIsNone(state.replicas[0].current_snapshot_identity)


class RouteAllowlistTest(unittest.TestCase):
    def test_inference_routes_allowed(self) -> None:
        for path in ("generate", "v1/completions", "v1/chat/completions", "v1/models"):
            self.assertTrue(is_customer_inference_route(path), path)

    def test_control_routes_blocked(self) -> None:
        for path in ("rpc_sync_from_bulletin_board", "server_info", "update_weights_from_ipc", "start_profile"):
            self.assertFalse(is_customer_inference_route(path), path)


class FrontdoorAppTest(unittest.TestCase):
    def _client(self, *, ledger: dict | None = None, authorize=None):
        from fastapi.testclient import TestClient

        from cookbook.standalone_rollouts.frontdoor import create_frontdoor_app

        state: dict = {"ledger": dict(ledger or {}), "version": None}
        calls: dict[str, list] = {"advanced": [], "woke": [], "normalized": [], "proxied": [], "saved": []}

        async def load_ledger():
            return state["ledger"]

        async def save_ledger(data):
            state["ledger"] = data
            calls["saved"].append(data)

        async def normalize_index(identity, metadata):
            calls["normalized"].append((identity, metadata))

        async def advance_to(version):
            state["version"] = version
            calls["advanced"].append(version)

        async def list_server_infos():
            v = state["version"] if state["version"] is not None else 0
            return [{"current_version": v, "sync_state": "IDLE", "last_sync_error": None, "replica_id": "a"}]

        async def wake(version):
            calls["woke"].append(version)

        from fastapi.responses import JSONResponse

        async def proxy(request, path):
            calls["proxied"].append(path)
            return JSONResponse({"proxied": path})

        app = create_frontdoor_app(
            load_ledger=load_ledger,
            save_ledger=save_ledger,
            normalize_index=normalize_index,
            advance_to=advance_to,
            list_server_infos=list_server_infos,
            proxy=proxy,
            authorize=authorize,
            wake=wake,
        )
        return TestClient(app), calls, state

    def test_full_snapshot_signal_mints_v0_without_normalizing(self) -> None:
        client, calls, _ = self._client()
        resp = client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["version"], 0)
        self.assertEqual(body["current_snapshot_identity"], "base-ckpt")
        self.assertFalse(body["already_current"])
        self.assertEqual(calls["advanced"], [0])
        self.assertEqual(calls["normalized"], [])  # a base is not a delta

    def test_delta_signal_mints_next_version_and_normalizes_index(self) -> None:
        client, calls, _ = self._client()
        client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        resp = client.post(
            HOT_LOAD_PATH,
            json={
                "identity": "ckpt-100",
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": "base-ckpt",
                    "compression_format": "zstd",
                    "checksum_format": "adler32",
                },
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["version"], 1)
        self.assertEqual(calls["advanced"], [0, 1])
        self.assertEqual(calls["woke"], [0, 1])
        self.assertEqual(len(calls["normalized"]), 1)
        ident, meta = calls["normalized"][0]
        self.assertEqual(ident, "ckpt-100")
        self.assertEqual(meta["version"], "000001")
        self.assertEqual(meta["base_version"], "000000")

    def test_resignal_is_idempotent_no_remint_no_normalize(self) -> None:
        client, calls, _ = self._client()
        client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        first = client.post(HOT_LOAD_PATH, json={"identity": "ckpt-100", "incremental_snapshot_metadata": {"previous_snapshot_identity": "base-ckpt", "compression_format": "zstd", "checksum_format": "adler32"}})
        again = client.post(HOT_LOAD_PATH, json={"identity": "ckpt-100", "incremental_snapshot_metadata": {"previous_snapshot_identity": "base-ckpt", "compression_format": "zstd", "checksum_format": "adler32"}})
        self.assertEqual(first.json()["version"], 1)
        self.assertEqual(again.json()["version"], 1)
        self.assertTrue(again.json()["already_current"])
        # Only one normalize / one save for the mint; the retry re-advances the
        # head (idempotent) but does not re-mint or re-normalize.
        self.assertEqual(len(calls["normalized"]), 1)
        self.assertEqual(calls["advanced"], [0, 1, 1])

    def test_signal_before_upload_is_409_and_leaves_state_clean(self) -> None:
        from fastapi.testclient import TestClient

        from cookbook.standalone_rollouts.frontdoor import create_frontdoor_app

        state: dict = {"ledger": {}, "version": None}
        calls: dict[str, list] = {"advanced": [], "saved": []}

        async def load_ledger():
            return state["ledger"]

        async def save_ledger(data):
            state["ledger"] = data
            calls["saved"].append(data)

        async def normalize_index(identity, metadata):
            raise FileNotFoundError(identity)  # upload has not landed yet

        async def advance_to(version):
            calls["advanced"].append(version)

        async def list_server_infos():
            return []

        async def proxy(request, path):
            from fastapi.responses import JSONResponse

            return JSONResponse({})

        app = create_frontdoor_app(
            load_ledger=load_ledger,
            save_ledger=save_ledger,
            normalize_index=normalize_index,
            advance_to=advance_to,
            list_server_infos=list_server_infos,
            proxy=proxy,
        )
        client = TestClient(app)
        resp = client.post(
            HOT_LOAD_PATH,
            json={"identity": "ckpt-100", "incremental_snapshot_metadata": {"previous_snapshot_identity": "base", "compression_format": "zstd", "checksum_format": "adler32"}},
        )
        self.assertEqual(resp.status_code, 409)
        # No pointer move, no ledger commit -> a retry after upload converges.
        self.assertEqual(calls["advanced"], [])
        self.assertEqual(calls["saved"], [])
        self.assertEqual(state["ledger"], {})

    def test_opaque_non_weight_v_identity_is_accepted(self) -> None:
        client, _, _ = self._client()
        resp = client.post(HOT_LOAD_PATH, json={"identity": "ckpt-step-500"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["current_snapshot_identity"], "ckpt-step-500")

    def test_get_reports_readiness_in_customer_identity(self) -> None:
        client, _, _ = self._client()
        client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        client.post(HOT_LOAD_PATH, json={"identity": "ckpt-100", "incremental_snapshot_metadata": {"previous_snapshot_identity": "base-ckpt", "compression_format": "zstd", "checksum_format": "adler32"}})
        body = client.get(HOT_LOAD_PATH).json()
        self.assertEqual(len(body["replicas"]), 1)
        self.assertEqual(body["replicas"][0]["current_snapshot_identity"], "ckpt-100")
        self.assertTrue(body["replicas"][0]["readiness"])

    def test_post_requires_identity(self) -> None:
        client, _, _ = self._client()
        self.assertEqual(client.post(HOT_LOAD_PATH, json={}).status_code, 400)

    def test_post_malformed_body_is_400_not_500(self) -> None:
        client, calls, _ = self._client()
        resp = client.post(HOT_LOAD_PATH, content=b"not json", headers={"content-type": "application/json"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(calls["advanced"], [])

    def test_delta_without_previous_identity_is_400(self) -> None:
        client, calls, _ = self._client()
        resp = client.post(
            HOT_LOAD_PATH,
            json={"identity": "ckpt-100", "incremental_snapshot_metadata": {"compression_format": "zstd"}},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(calls["advanced"], [])

    def test_second_full_snapshot_is_409(self) -> None:
        client, calls, _ = self._client()
        client.post(HOT_LOAD_PATH, json={"identity": "base-a"})
        resp = client.post(HOT_LOAD_PATH, json={"identity": "base-b"})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(calls["advanced"], [0])  # pointer untouched by the reject

    def test_delta_with_unknown_parent_is_409(self) -> None:
        client, calls, _ = self._client()
        client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        resp = client.post(
            HOT_LOAD_PATH,
            json={
                "identity": "ckpt-2",
                "incremental_snapshot_metadata": {"previous_snapshot_identity": "ckpt-l"},
            },
        )
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(calls["normalized"], [])

    def test_resignalling_an_older_identity_is_a_409_rewind(self) -> None:
        client, calls, _ = self._client()
        client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        client.post(HOT_LOAD_PATH, json={"identity": "ckpt-100", "incremental_snapshot_metadata": {"previous_snapshot_identity": "base-ckpt", "compression_format": "zstd", "checksum_format": "adler32"}})
        resp = client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
        self.assertEqual(resp.status_code, 409)
        self.assertEqual(resp.json()["error"]["type"], "WeightRewindRejected")
        self.assertEqual(calls["advanced"], [0, 1])  # pointer stays at head

    def test_post_overlong_or_slashed_identity_is_400(self) -> None:
        client, calls, _ = self._client()
        for identity in ("weight_v" + "9" * 100000, "a/b/c"):
            self.assertEqual(client.post(HOT_LOAD_PATH, json={"identity": identity}).status_code, 400)
        self.assertEqual(calls["advanced"], [])

    def test_catch_all_proxies_inference_and_404s_internal(self) -> None:
        client, calls, _ = self._client()
        self.assertEqual(client.post("/v1/chat/completions", json={}).status_code, 200)
        self.assertEqual(client.post("/rpc_sync_from_bulletin_board", json={}).status_code, 404)
        self.assertEqual(calls["proxied"], ["v1/chat/completions"])

    def test_auth_rejection_blocks_every_route(self) -> None:
        from fastapi.responses import JSONResponse

        def deny(_headers):
            return JSONResponse({"error": "unauthorized"}, status_code=401)

        client, calls, _ = self._client(authorize=deny)
        self.assertEqual(client.post(HOT_LOAD_PATH, json={"identity": "x"}).status_code, 401)
        self.assertEqual(client.get(HOT_LOAD_PATH).status_code, 401)
        self.assertEqual(client.post("/v1/chat/completions", json={}).status_code, 401)
        self.assertEqual(calls["advanced"], [])
        self.assertEqual(calls["proxied"], [])


if __name__ == "__main__":
    unittest.main()
