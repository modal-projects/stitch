"""In-process end-to-end simulator for the standalone rollouts abstraction layer.

Wires the REAL front door (ledger + normalization) to N real WeightSyncManagers
(delta-view boards + fake engines) over a tmpdir "transport", with no Modal and
no GPU. It grounds the contract the ledger/normalization/symlink-view commits
built: a customer who uploads a plain HF index (metadata = {"total_size": ...})
and signals an opaque identity — the exact shape that bricked the pool per
battle-plan F2 — now converges, because the front door normalizes the index on
POST and the sidecar resolves the identity dir through the weight_vN view.

The fake engine records applied versions without decoding real delta bytes (the
codec is the pinned slime fork's concern); everything else — opaque identity ->
version, index normalization, pointer advance, view rebuild, manifest parse,
apply gate, readiness translation — is the real code path.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import httpx

from cookbook.standalone_rollouts.delta_view import merge_index_metadata
from cookbook.standalone_rollouts.frontdoor import HOT_LOAD_PATH, create_frontdoor_app
from cookbook.standalone_rollouts.provider import _delta_view_refresh
from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import weight_identity
from stitch.sync import WeightSyncManager


class _FakeEngine:
    backend = "fake"

    def __init__(self) -> None:
        self.applies: list[int] = []

    async def flush_cache(self) -> None: ...

    async def apply_manifest(self, manifest, version_path) -> None:
        # The manifest was parsed for real by from_slime_index off the normalized
        # index reached through the weight_vN view — that is what this grounds.
        self.applies.append(manifest.version)

    async def pause_generation(self) -> None: ...

    async def continue_generation(self) -> None: ...


def _upload_plain_hf_checkpoint(transport: Path, identity: str) -> None:
    """A vanilla HF checkpoint dir: a weight_map and a bare total_size metadata,
    with NO slime version/base_version/delta_encoding block (F2's failing shape)."""
    d = transport / identity
    d.mkdir(parents=True)
    (d / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 123},
                "weight_map": {"w0": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )


class SimulatorTest(unittest.IsolatedAsyncioTestCase):
    async def _stack(self, tmp: str, *, n_replicas: int = 2):
        transport = Path(tmp) / "transport"
        transport.mkdir()

        managers: list[WeightSyncManager] = []
        engines: list[_FakeEngine] = []
        for i in range(n_replicas):
            view_dir = str(Path(tmp) / f"view{i}")
            board = FilesystemBulletinBoard(
                view_dir, layout="slime", refresh=_delta_view_refresh(view_dir, str(transport))
            )
            engine = _FakeEngine()
            managers.append(
                WeightSyncManager(board=board, engine=engine, commit_mode="in_place")
            )
            engines.append(engine)

        async def load_ledger():
            p = transport / "identities.json"
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

        async def save_ledger(data):
            (transport / "identities.json").write_text(json.dumps(data), encoding="utf-8")

        async def normalize_index(identity, metadata):
            # The real merge (FileNotFoundError -> 409, malformed -> 400).
            merge_index_metadata(transport / identity / "model.safetensors.index.json", metadata)

        async def advance_to(version):
            (transport / "latest").write_text(weight_identity(version), encoding="utf-8")

        async def list_server_infos():
            return [await m.server_info() for m in managers]

        async def wake(version):
            pass  # the sim drives reconciliation explicitly for determinism

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
            authorize=lambda headers: None,
            wake=wake,
        )
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://fd")
        return transport, managers, engines, client

    async def test_vanilla_hf_index_converges_after_frontdoor_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport, managers, engines, client = await self._stack(tmp)
            async with client:
                # Base: customer pre-uploaded a full snapshot and signals it.
                _upload_plain_hf_checkpoint(transport, "base-ckpt")
                r = await client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["version"], 0)

                # Delta: a PLAIN HF index (F2's brick shape) + opaque identity.
                _upload_plain_hf_checkpoint(transport, "ckpt-100")
                r = await client.post(
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
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()["version"], 1)

                # The front door normalized the plain index in place: it now carries
                # the slime metadata block the decoder needs (this is the F2 fix).
                normalized = json.loads(
                    (transport / "ckpt-100" / "model.safetensors.index.json").read_text()
                )["metadata"]
                self.assertEqual(normalized["version"], "000001")
                self.assertEqual(normalized["base_version"], "000000")
                self.assertEqual(normalized["delta_encoding"], "xor")
                self.assertEqual(normalized["total_size"], 123)  # original field preserved

                # Every replica converges to v1 (manifest parsed for real via the view).
                for mgr in managers:
                    await mgr.sync_to()
                for mgr, engine in zip(managers, engines):
                    self.assertEqual(mgr.current_version, 1)
                    self.assertEqual(engine.applies, [1])

                # Readiness reports the customer's opaque identity, so their
                # equality match (current_snapshot_identity == "ckpt-100") works.
                body = (await client.get(HOT_LOAD_PATH)).json()
                self.assertEqual(len(body["replicas"]), 2)
                for replica in body["replicas"]:
                    self.assertTrue(replica["readiness"])
                    self.assertEqual(replica["current_snapshot_identity"], "ckpt-100")

    async def test_signal_before_upload_is_409_and_pool_stays_put(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport, managers, engines, client = await self._stack(tmp)
            async with client:
                _upload_plain_hf_checkpoint(transport, "base-ckpt")
                await client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
                # Signal a delta whose dir has NOT been uploaded yet.
                r = await client.post(
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
                self.assertEqual(r.status_code, 409)
                # Pointer never advanced past the base; ledger has no ckpt-100.
                ledger = json.loads((transport / "identities.json").read_text())
                self.assertNotIn("ckpt-100", ledger["entries"])
                for mgr in managers:
                    await mgr.sync_to()
                    self.assertEqual(mgr.current_version, 0)

    async def test_fork_signal_is_409_and_pool_keeps_converging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport, managers, _, client = await self._stack(tmp, n_replicas=1)
            async with client:
                _upload_plain_hf_checkpoint(transport, "base-ckpt")
                await client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
                _upload_plain_hf_checkpoint(transport, "ckpt-100")
                delta = lambda ident, prev: {
                    "identity": ident,
                    "incremental_snapshot_metadata": {"previous_snapshot_identity": prev},
                }
                await client.post(HOT_LOAD_PATH, json=delta("ckpt-100", "base-ckpt"))
                for mgr in managers:
                    await mgr.sync_to()
                    self.assertEqual(mgr.current_version, 1)

                # A fork off the base is refused; nothing enters the log.
                _upload_plain_hf_checkpoint(transport, "ckpt-fork")
                r = await client.post(HOT_LOAD_PATH, json=delta("ckpt-fork", "base-ckpt"))
                self.assertEqual(r.status_code, 409)

                # The chain head still extends and the pool still converges —
                # an accepted fork would have wedged replicas at v1 forever.
                _upload_plain_hf_checkpoint(transport, "ckpt-200")
                r = await client.post(HOT_LOAD_PATH, json=delta("ckpt-200", "ckpt-100"))
                self.assertEqual(r.json()["version"], 2)
                for mgr in managers:
                    await mgr.sync_to()
                    self.assertEqual(mgr.current_version, 2)
                    self.assertIsNone(mgr.last_sync_error)

    async def test_malformed_uploaded_index_is_400_not_500(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport, _, _, client = await self._stack(tmp)
            async with client:
                _upload_plain_hf_checkpoint(transport, "base-ckpt")
                await client.post(HOT_LOAD_PATH, json={"identity": "base-ckpt"})
                # A truncated/interrupted upload left an unparseable index.
                d = transport / "ckpt-100"
                d.mkdir()
                (d / "model.safetensors.index.json").write_text('{"metadata": {', encoding="utf-8")
                r = await client.post(
                    HOT_LOAD_PATH,
                    json={
                        "identity": "ckpt-100",
                        "incremental_snapshot_metadata": {
                            "previous_snapshot_identity": "base-ckpt",
                        },
                    },
                )
                self.assertEqual(r.status_code, 400)
                # The failed signal left no ledger entry, so a fixed re-upload
                # and re-signal converges.
                ledger = json.loads((transport / "identities.json").read_text())
                self.assertNotIn("ckpt-100", ledger["entries"])


if __name__ == "__main__":
    unittest.main()
