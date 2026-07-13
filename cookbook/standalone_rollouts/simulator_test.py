from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import httpx

from cookbook.standalone_rollouts.delta_view import (
    INDEX_FILENAME,
    derive_delta_index,
)
from cookbook.standalone_rollouts.ledger import (
    IdentityLedger,
    load_ledger_data,
    save_ledger_data,
)
from cookbook.standalone_rollouts.opaque_frontdoor import (
    HOT_LOAD_PATH,
    create_opaque_frontdoor_app,
)
from cookbook.standalone_rollouts.opaque_protocol import recover_frontdoor_state
from cookbook.standalone_rollouts.provider import build_opaque_board
from stitch.bulletin import FilesystemBulletinBoard
from stitch.sync import WeightSyncManager


BASE_IDENTITY = "customer-base"
DELTA_IDENTITY = "customer-step-1"


class _RecordingEngine:
    backend = "fake"

    def __init__(self) -> None:
        self.applies: list[tuple[object, Path]] = []

    async def flush_cache(self) -> None:
        pass

    async def apply_manifest(self, manifest, version_path: str) -> None:
        self.applies.append((manifest, Path(version_path)))

    async def pause_generation(self) -> None:
        pass

    async def continue_generation(self) -> None:
        pass


def _new_replica(
    root: Path, replica_id: str
) -> tuple[WeightSyncManager, _RecordingEngine]:
    engine = _RecordingEngine()
    manager = WeightSyncManager(
        board=build_opaque_board(
            transport_root=str(root / "transport"),
            local_view_dir=str(root / f"view-{replica_id}"),
            base_snapshot_identity=BASE_IDENTITY,
        ),
        engine=engine,
        run_id=replica_id,
    )
    return manager, engine


def _create_frontdoor(transport: Path, managers: list[WeightSyncManager]):
    board = FilesystemBulletinBoard(transport, layout="slime")
    recovery = recover_frontdoor_state(
        persisted_ledger=load_ledger_data(transport),
        expected_base_identity=BASE_IDENTITY,
        pointer=board.read_latest(),
    )
    if recovery.save_ledger:
        save_ledger_data(transport, recovery.ledger.to_dict())
    if recovery.pointer_to_write is not None:
        board.write_latest(None, recovery.pointer_to_write)

    async def save_ledger(data) -> None:
        await asyncio.to_thread(save_ledger_data, transport, data)

    async def derive_delta(entry, *, committed: bool) -> None:
        await asyncio.to_thread(
            derive_delta_index,
            transport,
            entry,
            committed=committed,
        )

    async def advance_to(version: int) -> None:
        await asyncio.to_thread(board.write_latest, None, version)

    async def list_server_infos() -> list[dict]:
        return [await manager.server_info() for manager in managers]

    async def wake(version: int) -> None:
        # Production wakes enqueue this reconciliation through HTTP. Await it in
        # the simulator so the end-to-end assertion is deterministic.
        await asyncio.gather(*(manager.sync_to(version) for manager in managers))

    async def proxy(_request, _path):
        raise AssertionError("inference is outside this protocol simulator")

    app = create_opaque_frontdoor_app(
        ledger=recovery.ledger,
        save_ledger=save_ledger,
        derive_delta=derive_delta,
        advance_to=advance_to,
        list_server_infos=list_server_infos,
        proxy=proxy,
        authorize=lambda _headers: None,
        wake=wake,
    )
    return app, recovery


def _write_upload(transport: Path) -> dict[str, bytes]:
    upload = transport / DELTA_IDENTITY
    upload.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    (upload / shard_name).write_bytes(b"opaque-delta-bytes")
    (upload / INDEX_FILENAME).write_text(
        json.dumps(
            {
                "metadata": {"total_size": 18},
                "weight_map": {"model.weight": shard_name},
            }
        ),
        encoding="utf-8",
    )
    return {path.name: path.read_bytes() for path in upload.iterdir() if path.is_file()}


class OpaqueProtocolSimulatorTest(unittest.TestCase):
    def test_publish_sync_readiness_and_cold_restart(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                transport = root / "transport"
                transport.mkdir()

                replicas = [_new_replica(root, name) for name in ("r1", "r2")]
                managers = [manager for manager, _engine in replicas]
                app, initial_recovery = _create_frontdoor(transport, managers)
                self.assertTrue(initial_recovery.save_ledger)
                await asyncio.gather(*(manager.startup_sync() for manager in managers))

                raw_before = _write_upload(transport)
                payload = {
                    "identity": DELTA_IDENTITY,
                    "incremental_snapshot_metadata": {
                        "previous_snapshot_identity": BASE_IDENTITY,
                        "compression_format": "zstd",
                        "checksum_format": "xxh3-128",
                    },
                    "reset_prompt_cache": "new_session",
                }

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://frontdoor",
                ) as client:
                    response = await client.post(HOT_LOAD_PATH, json=payload)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.json()["version"], 1)
                    self.assertEqual(
                        response.json()["current_snapshot_identity"], DELTA_IDENTITY
                    )
                    pool = (await client.get(HOT_LOAD_PATH)).json()

                ledger = IdentityLedger.from_dict(
                    load_ledger_data(transport),
                    expected_base_identity=BASE_IDENTITY,
                )
                self.assertEqual(ledger.head.identity, DELTA_IDENTITY)
                board = FilesystemBulletinBoard(transport, layout="slime")
                self.assertEqual(board.read_latest(), (None, 1))
                self.assertEqual(
                    {
                        path.name: path.read_bytes()
                        for path in (transport / DELTA_IDENTITY).iterdir()
                        if path.is_file()
                    },
                    raw_before,
                )

                for manager, engine in replicas:
                    self.assertEqual(manager.current_version, 1)
                    self.assertIsNone(manager.current_run_id)
                    self.assertEqual(
                        [manifest.version for manifest, _path in engine.applies], [1]
                    )
                    manifest, version_path = engine.applies[0]
                    self.assertEqual(manifest.base_version, 0)
                    self.assertEqual(version_path.name, "weight_v000001")

                replicas_by_id = {
                    replica["replica_id"]: replica for replica in pool["replicas"]
                }
                self.assertEqual(set(replicas_by_id), {"r1", "r2"})
                for replica in replicas_by_id.values():
                    self.assertTrue(replica["readiness"])
                    self.assertEqual(replica["current_version"], 1)
                    self.assertEqual(
                        replica["current_snapshot_identity"], DELTA_IDENTITY
                    )

                # Exercise recovery of the only crash window where ledger is
                # committed but the advisory pointer remains behind.
                board.write_latest(None, 0)
                restarted = [_new_replica(root, name) for name in ("r3", "r4")]
                restarted_managers = [manager for manager, _engine in restarted]
                restarted_app, recovery = _create_frontdoor(
                    transport, restarted_managers
                )
                self.assertEqual(recovery.pointer_to_write, 1)
                self.assertEqual(board.read_latest(), (None, 1))

                await asyncio.gather(
                    *(manager.startup_sync() for manager in restarted_managers)
                )
                ledger_before_retry = (transport / "identities.json").read_bytes()

                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=restarted_app),
                    base_url="http://frontdoor",
                ) as client:
                    restarted_pool = (await client.get(HOT_LOAD_PATH)).json()
                    retry = await client.post(HOT_LOAD_PATH, json=payload)

                self.assertEqual(retry.status_code, 200)
                self.assertTrue(retry.json()["already_current"])
                self.assertEqual(
                    (transport / "identities.json").read_bytes(), ledger_before_retry
                )
                self.assertEqual(
                    {
                        path.name: path.read_bytes()
                        for path in (transport / DELTA_IDENTITY).iterdir()
                        if path.is_file()
                    },
                    raw_before,
                )
                for replica in restarted_pool["replicas"]:
                    self.assertTrue(replica["readiness"])
                    self.assertEqual(
                        replica["current_snapshot_identity"], DELTA_IDENTITY
                    )

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
