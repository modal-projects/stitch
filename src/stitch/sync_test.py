from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import SyncState, VersionManifest, WeightVersionPolicy
from stitch.sync import PolicyViolation, WeightSyncManager


async def _settle() -> None:
    """Drain currently-ready callbacks so a task that is about to park on an
    asyncio primitive reaches and parks at its await point. Deterministic
    replacement for a wall-clock ``sleep`` when asserting a task is *blocked*."""
    for _ in range(10):
        await asyncio.sleep(0)


def _write_slime_version(base: Path, version: int, prev: int) -> None:
    vdir = base / f"weight_v{version:06d}"
    vdir.mkdir(parents=True)
    (vdir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"version": f"{version:06d}", "base_version": f"{prev:06d}"},
                "weight_map": {"w": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )


class FakeEngine:
    backend = "fake"

    def __init__(self) -> None:
        self.applies: list[tuple[int, str]] = []
        self.events: list[str] = []
        self.apply_gate: asyncio.Event | None = None
        self.apply_started: asyncio.Event = asyncio.Event()

    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
        self.apply_started.set()
        if self.apply_gate is not None:
            await self.apply_gate.wait()
        self.applies.append((manifest.version, version_path))
        self.events.append("apply")

    async def reset(self) -> None:
        self.events.append("reset")

    async def pause_generation(self) -> None:
        self.events.append("pause")

    async def continue_generation(self) -> None:
        self.events.append("continue")


class SyncManagerTest(unittest.TestCase):
    def test_startup_sync_composes_tail_into_one_apply(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                board.publish_manifest(VersionManifest(version=2, base_version=1, backend="fake", load_format="noop"))
                engine = FakeEngine()
                manager = WeightSyncManager(board=board, engine=engine)

                await manager.startup_sync()

                self.assertEqual(manager.current_version, 2)
                self.assertEqual(manager.sync_state, SyncState.IDLE)
                self.assertEqual([v for v, _ in engine.applies], [2])
                self.assertEqual(engine.events, ["pause", "apply", "continue"])

        asyncio.run(run())

    def test_run_change_rematerializes_and_resets_under_gate(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                board = FilesystemBulletinBoard(root, layout="slime")
                _write_slime_version(root / "run-a", 1, 0)
                _write_slime_version(root / "run-a", 2, 1)
                board.write_latest("run-a", 2)
                engine = FakeEngine()
                manager = WeightSyncManager(board=board, engine=engine)

                await manager.startup_sync()
                self.assertEqual(manager.current_run_id, "run-a")
                self.assertEqual(manager.current_version, 2)
                self.assertEqual([v for v, _ in engine.applies], [2])

                _write_slime_version(root / "run-b", 1, 0)
                board.write_latest("run-b", 1)
                await manager.sync_to()

                self.assertEqual(manager.current_run_id, "run-b")
                self.assertEqual(manager.current_version, 1)
                self.assertIn("run-b", engine.applies[-1][1])
                i = engine.events.index("reset")
                self.assertEqual(engine.events[i - 1], "pause")
                self.assertEqual(engine.events[i + 1], "continue")

        asyncio.run(run())

    def test_stitch_layout_never_switches_run(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                board.publish_manifest(VersionManifest(version=2, base_version=1, backend="fake", load_format="noop"))
                engine = FakeEngine()
                manager = WeightSyncManager(board=board, engine=engine)

                await manager.startup_sync()

                self.assertIsNone(manager.current_run_id)
                self.assertEqual(manager.current_version, 2)
                self.assertNotIn("reset", engine.events)

        asyncio.run(run())

    def test_exact_and_min_policy_errors_are_retryable(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                manager = WeightSyncManager(board=board, engine=FakeEngine())

                ok, current, error = await manager.validate_policy(WeightVersionPolicy(exact_version=1))
                self.assertFalse(ok)
                self.assertEqual(current, 0)
                self.assertEqual(error["error"]["type"], "WeightVersionNotReady")

                await manager.sync_to(1)
                ok, current, error = await manager.validate_policy(WeightVersionPolicy(min_required_version=1))
                self.assertTrue(ok)
                self.assertEqual(current, 1)
                self.assertIsNone(error)

                ok, _current, error = await manager.validate_policy(WeightVersionPolicy(exact_version=0))
                self.assertFalse(ok)
                self.assertEqual(error["error"]["type"], "WeightVersionTooOld")

        asyncio.run(run())

    def test_request_context_pins_and_reports_serving_version(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                manager = WeightSyncManager(board=board, engine=FakeEngine())
                await manager.sync_to(1)

                async with manager.request_context(WeightVersionPolicy(exact_version=1)) as version:
                    self.assertEqual(version, 1)
                info = await manager.server_info()
                self.assertEqual(info["current_version"], 1)

                with self.assertRaises(PolicyViolation) as cm:
                    async with manager.request_context(WeightVersionPolicy(exact_version=0)):
                        pass
                self.assertEqual(cm.exception.error["error"]["type"], "WeightVersionTooOld")

        asyncio.run(run())

    def test_request_context_during_commit_is_not_gated(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                engine = FakeEngine()
                engine.apply_gate = asyncio.Event()
                manager = WeightSyncManager(board=board, engine=engine)

                sync = asyncio.create_task(manager.sync_to(1))
                await engine.apply_started.wait()
                async with manager.request_context(WeightVersionPolicy(exact_version=0)) as version:
                    self.assertEqual(version, 0)
                engine.apply_gate.set()
                await sync
                self.assertEqual(manager.current_version, 1)

        asyncio.run(run())

    def test_in_place_commit_pauses_applies_continues_without_flush(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))

                versions_at_continue: list[int] = []

                class InPlaceEngine(FakeEngine):
                    async def continue_generation(self) -> None:
                        versions_at_continue.append(manager.current_version)
                        await super().continue_generation()

                engine = InPlaceEngine()
                manager = WeightSyncManager(board=board, engine=engine)
                await manager.sync_to(1)

                self.assertEqual(manager.current_version, 1)
                self.assertEqual(engine.events, ["pause", "apply", "continue"])
                self.assertEqual(versions_at_continue, [1])

        asyncio.run(run())

    def test_in_place_commit_continues_engine_on_apply_failure(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))

                class FailingEngine(FakeEngine):
                    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
                        raise RuntimeError("apply blew up")

                engine = FailingEngine()
                manager = WeightSyncManager(board=board, engine=engine)
                await manager.sync_to(1)

                self.assertEqual(manager.sync_state, SyncState.ERROR)
                self.assertEqual(manager.current_version, 0)
                self.assertEqual(engine.events, ["pause", "continue"])
                async with manager.request_context() as version:
                    self.assertEqual(version, 0)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
