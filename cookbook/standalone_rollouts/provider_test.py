from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from cookbook.standalone_rollouts import provider
from cookbook.standalone_rollouts.provider import (
    InMemoryStateStore,
    ModalStateStore,
    ProviderSettings,
    ProviderShim,
    create_app,
    _materialize_snapshot,
)
from stitch.protocol import WeightVersionPolicy


class ProviderShimTest(unittest.TestCase):
    def _shim(self) -> ProviderShim:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return ProviderShim(
            settings=ProviderSettings(
                upstream_url="http://127.0.0.1:30000",
                transport_root=None,
                s3_bucket="bucket",
                s3_prefix="prefix",
                snapshot_root=Path(tmp.name),
                replica_id="replica-1",
                base_snapshot_identity="base",
            ),
            store=InMemoryStateStore(),
        )

    def test_version_policy_error_uses_local_snapshot_identity(self) -> None:
        shim = self._shim()
        shim.current_identity = "weight_v000002"

        not_ready = shim._policy_error(WeightVersionPolicy(exact_version=3))
        too_old = shim._policy_error(WeightVersionPolicy(exact_version=1))
        min_ok = shim._policy_error(WeightVersionPolicy(min_required_version=2))

        self.assertEqual(not_ready["error"]["type"], "WeightVersionNotReady")
        self.assertEqual(too_old["error"]["type"], "WeightVersionTooOld")
        self.assertIsNone(min_ok)

    def test_base_identity_is_version_zero(self) -> None:
        shim = self._shim()

        self.assertIsNone(
            shim._policy_error(WeightVersionPolicy(exact_version=0))
        )
        error = shim._policy_error(WeightVersionPolicy(min_required_version=1))

        self.assertEqual(error["error"]["type"], "WeightVersionNotReady")

    def test_commit_advances_version_before_releasing_gate(self) -> None:
        """current_version must advance while the commit gate is still held, so a
        request can never be admitted observing the stale version on
        already-mutated engine weights (the P0.1 window). Mirrors
        WeightSyncManager._sync_once."""

        async def run() -> None:
            shim = self._shim()
            self.assertEqual(shim.current_version, 0)

            versions_at_gate_release: list[int] = []
            original_end_commit = shim._end_commit

            async def spy_end_commit() -> None:
                versions_at_gate_release.append(shim.current_version)
                await original_end_commit()

            shim._end_commit = spy_end_commit  # type: ignore[method-assign]

            async def noop_flush(_upstream_url: str) -> None:
                return None

            async def noop_apply(**_kwargs: object) -> None:
                return None

            def noop_materialize(_settings, _identity, _destination) -> None:
                return None

            with mock.patch.object(provider, "_flush_cache", noop_flush), mock.patch.object(
                provider, "_apply_snapshot", noop_apply
            ), mock.patch.object(provider, "_materialize_snapshot", noop_materialize):
                await shim._sync_to({"identity": "weight_v000001"})

            self.assertEqual(shim.current_version, 1)
            # The gate was released only after the version had already advanced;
            # pre-fix this recorded 0 (gate open while current_version was stale).
            self.assertEqual(versions_at_gate_release, [1])

        asyncio.run(run())

    def test_generate_uses_shared_sglang_proxy_behavior(self) -> None:
        from fastapi.testclient import TestClient

        shim = self._shim()
        with TestClient(create_app(shim)) as client:
            with mock.patch("httpx.AsyncClient", _RecordingUpstream):
                _RecordingUpstream.last_json = None
                resp = client.post(
                    "/generate",
                    json={
                        "input_ids": [1, 2],
                        "sampling_params": {"max_new_tokens": 4},
                        "weight_version": {"exact_version": 0},
                    },
                )

        self.assertEqual(resp.status_code, 200)
        forwarded = _RecordingUpstream.last_json
        self.assertIsNotNone(forwarded)
        self.assertNotIn("weight_version", forwarded)
        self.assertEqual(forwarded["extra_key"], "wv0;")
        self.assertIn("rid", forwarded)
        meta = resp.json()["meta_info"]
        self.assertEqual(meta["weight_version"], "0")
        self.assertEqual(meta["weight_version_start"], 0)
        self.assertEqual(meta["weight_version_end"], 0)

    def test_hot_load_post_accepts_json_body(self) -> None:
        from fastapi.testclient import TestClient

        shim = self._shim()
        with TestClient(create_app(shim)) as client:
            resp = client.post(
                "/hot_load/v1/models/hot_load",
                json={"identity": "weight_v000001"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["accepted"])
        self.assertEqual(
            asyncio.run(shim.store.desired())["identity"], "weight_v000001"
        )

    def test_materialize_snapshot_copies_from_transport_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            transport_root = root / "transport"
            source = transport_root / "weight_v000001"
            source.mkdir(parents=True)
            (source / "rank0000_delta.safetensors").write_bytes(b"delta")
            (source / "nested").mkdir()
            (source / "nested" / "manifest.json").write_text("{}", encoding="utf-8")
            destination = root / "snapshots" / "weight_v000001"
            settings = ProviderSettings(
                upstream_url="http://127.0.0.1:30000",
                transport_root=transport_root,
                s3_bucket=None,
                s3_prefix="",
                snapshot_root=root / "snapshots",
                replica_id="replica-1",
                base_snapshot_identity="base",
            )

            _materialize_snapshot(settings, "weight_v000001", destination)

            self.assertEqual(
                (destination / "rank0000_delta.safetensors").read_bytes(), b"delta"
            )
            self.assertEqual(
                (destination / "nested" / "manifest.json").read_text(encoding="utf-8"),
                "{}",
            )


class ModalStateStoreTest(unittest.TestCase):
    def test_uses_modal_async_methods(self) -> None:
        async def exercise() -> None:
            state_dict = _FakeModalDict(
                {
                    "desired": {"identity": "weight_v000001"},
                    "replicas/replica-1": {
                        "replica_id": "replica-1",
                        "readiness": True,
                        "current_version": 1,
                        "updated_at": 0.0,
                    },
                }
            )
            store = ModalStateStore(state_dict)

            self.assertEqual(
                await store.desired(),
                {"identity": "weight_v000001"},
            )
            await store.set_desired({"identity": "weight_v000002"})
            await store.set_replica(
                "replica-2",
                {
                    "replica_id": "replica-2",
                    "readiness": True,
                    "current_version": 2,
                    "updated_at": 0.0,
                },
            )
            pool_state = await store.pool_state(state_ttl_seconds=1000.0)

            self.assertEqual(
                state_dict.calls, ["get.aio", "put.aio", "put.aio", "items.aio"]
            )
            self.assertEqual(
                sorted(replica.replica_id for replica in pool_state.replicas),
                ["replica-1", "replica-2"],
            )

        asyncio.run(exercise())


class _RecordingUpstream:
    last_json: dict | None = None
    last_url: str | None = None
    last_headers: dict | None = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def __aenter__(self) -> "_RecordingUpstream":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

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


class _AioOnlyMethod:
    def __init__(self, name, async_fn):
        self.name = name
        self.aio = async_fn

    def __call__(self, *args, **kwargs):
        raise AssertionError(f"sync {self.name} should not be called")


class _FakeModalDict:
    def __init__(self, data):
        self.data = dict(data)
        self.calls = []
        self.get = _AioOnlyMethod("get", self._get)
        self.put = _AioOnlyMethod("put", self._put)
        self.items = _AioOnlyMethod("items", self._items)

    async def _get(self, key):
        self.calls.append("get.aio")
        return self.data.get(key)

    async def _put(self, key, value):
        self.calls.append("put.aio")
        self.data[key] = value

    async def _items(self):
        self.calls.append("items.aio")
        for item in self.data.items():
            yield item


if __name__ == "__main__":
    unittest.main()
