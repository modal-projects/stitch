from __future__ import annotations

import asyncio
import tempfile
import unittest

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import SyncState, VersionManifest, WeightVersionPolicy
from stitch.sync import PolicyViolation, RolloutAdmissionGate, WeightSyncManager


class FakeEngine:
    backend = "fake"

    def __init__(self) -> None:
        self.flushes = 0
        self.applies: list[tuple[int, str]] = []
        self.events: list[str] = []
        self.apply_gate: asyncio.Event | None = None
        self.apply_started: asyncio.Event = asyncio.Event()

    async def flush_cache(self) -> None:
        self.flushes += 1
        self.events.append("flush")

    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
        self.apply_started.set()
        if self.apply_gate is not None:
            await self.apply_gate.wait()
        self.applies.append((manifest.version, version_path))
        self.events.append("apply")

    async def pause_generation(self) -> None:
        self.events.append("pause")

    async def continue_generation(self) -> None:
        self.events.append("continue")


class SyncManagerTest(unittest.TestCase):
    def test_startup_sync_applies_transition_chain(self) -> None:
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
                self.assertEqual([v for v, _ in engine.applies], [1, 2])

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
                    self.assertEqual(manager.active_requests, 1)
                    info = await manager.server_info()
                    self.assertEqual(info["inflight_exact_versions"], {"1": 1})
                self.assertEqual(manager.active_requests, 0)
                info = await manager.server_info()
                self.assertEqual(info["inflight_exact_versions"], {})

                with self.assertRaises(PolicyViolation) as cm:
                    async with manager.request_context(WeightVersionPolicy(exact_version=0)):
                        pass
                self.assertEqual(cm.exception.error["error"]["type"], "WeightVersionTooOld")

        asyncio.run(run())

    def test_commit_window_gates_admissions(self) -> None:
        """A request arriving while the engine apply is in flight must not be
        validated against the stale pre-commit version (it would otherwise be
        served on the new weights while reporting the old version)."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                engine = FakeEngine()
                engine.apply_gate = asyncio.Event()
                manager = WeightSyncManager(board=board, engine=engine)

                sync = asyncio.create_task(manager.sync_to(1))
                await engine.apply_started.wait()
                # Engine apply for v1 is now in flight; current_version still 0.
                self.assertEqual(manager.current_version, 0)

                async def admit_exact_zero() -> str:
                    try:
                        async with manager.request_context(WeightVersionPolicy(exact_version=0)):
                            return "served"
                    except PolicyViolation as exc:
                        return exc.error["error"]["type"]

                admit = asyncio.create_task(admit_exact_zero())
                await asyncio.sleep(0.05)
                # The admission must be gated, not validated against version 0.
                self.assertFalse(admit.done())

                engine.apply_gate.set()
                await sync
                # After the commit lands, exact_version=0 is no longer
                # satisfiable and must be rejected, not silently served on v1.
                self.assertEqual(await admit, "WeightVersionTooOld")
                self.assertEqual(manager.current_version, 1)

                async with manager.request_context(WeightVersionPolicy(min_required_version=1)) as version:
                    self.assertEqual(version, 1)

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
                manager = WeightSyncManager(board=board, engine=engine, commit_mode="in_place")
                await manager.sync_to(1)

                self.assertEqual(manager.current_version, 1)
                self.assertEqual(engine.events, ["pause", "apply", "continue"])
                self.assertEqual(engine.flushes, 0)
                # New admissions must see the new namespace before the engine
                # resumes, so the bump happens before continue_generation.
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
                manager = WeightSyncManager(board=board, engine=engine, commit_mode="in_place")
                await manager.sync_to(1)

                self.assertEqual(manager.sync_state, SyncState.ERROR)
                self.assertEqual(manager.current_version, 0)
                # The engine must not be left paused after a failed apply.
                self.assertEqual(engine.events, ["pause", "continue"])
                async with manager.request_context() as version:
                    self.assertEqual(version, 0)

        asyncio.run(run())

    def test_in_place_commit_gates_exact_but_admits_nonstrict(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))
                engine = FakeEngine()
                engine.apply_gate = asyncio.Event()
                manager = WeightSyncManager(board=board, engine=engine, commit_mode="in_place")

                # An exact-version request in flight blocks the commit point.
                exact_ctx = manager.request_context(WeightVersionPolicy(exact_version=0))
                await exact_ctx.__aenter__()
                sync = asyncio.create_task(manager.sync_to(1))
                await asyncio.sleep(0.05)
                self.assertFalse(engine.apply_started.is_set())

                # Releasing the exact request lets the commit proceed.
                await exact_ctx.__aexit__(None, None, None)
                await engine.apply_started.wait()

                # Mid-commit: non-strict requests are admitted (and stamped
                # with the pre-commit version); exact requests are gated.
                async with manager.request_context() as version:
                    self.assertEqual(version, 0)

                async def admit_exact() -> str:
                    try:
                        async with manager.request_context(WeightVersionPolicy(exact_version=0)):
                            return "served"
                    except PolicyViolation as exc:
                        return exc.error["error"]["type"]

                gated = asyncio.create_task(admit_exact())
                await asyncio.sleep(0.05)
                self.assertFalse(gated.done())

                engine.apply_gate.set()
                await sync
                self.assertEqual(manager.current_version, 1)
                self.assertEqual(await gated, "WeightVersionTooOld")

        asyncio.run(run())

    def test_commit_gate_clears_on_apply_failure(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = FilesystemBulletinBoard(tmp)
                board.publish_manifest(VersionManifest(version=1, base_version=0, backend="fake", load_format="noop"))

                class FailingEngine(FakeEngine):
                    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
                        raise RuntimeError("apply blew up")

                manager = WeightSyncManager(board=board, engine=FailingEngine())
                await manager.sync_to(1)
                self.assertEqual(manager.sync_state, SyncState.ERROR)

                # The gate must not be left set after a failed commit.
                async with manager.request_context() as version:
                    self.assertEqual(version, 0)

        asyncio.run(run())


class AdmissionGateCommitDriverTest(unittest.TestCase):
    """The shared commit_version driver both WeightSyncManager and the hot-load
    ProviderShim commit through."""

    def test_in_place_commit_advances_version_before_resume(self) -> None:
        async def run() -> None:
            gate = RolloutAdmissionGate(commit_mode="in_place")
            events: list[str] = []

            async def pause() -> None:
                events.append("pause")

            async def apply() -> None:
                events.append("apply")

            async def resume() -> None:
                events.append("resume")

            await gate.commit_version(
                apply=apply,
                on_applied=lambda: events.append("applied"),
                pause=pause,
                resume=resume,
            )

            # Pause, apply, advance the served version, then resume — and the
            # gate is reopened afterward.
            self.assertEqual(events, ["pause", "apply", "applied", "resume"])
            self.assertFalse(gate._committing)

        asyncio.run(run())

    def test_quiesce_commit_skips_pause_and_unwinds_on_failure(self) -> None:
        async def run() -> None:
            gate = RolloutAdmissionGate(commit_mode="quiesce")
            events: list[str] = []

            async def pause() -> None:
                events.append("pause")

            async def apply() -> None:
                events.append("apply")
                raise RuntimeError("boom")

            async def resume() -> None:
                events.append("resume")

            with self.assertRaises(RuntimeError):
                await gate.commit_version(
                    apply=apply,
                    on_applied=lambda: events.append("applied"),
                    pause=pause,
                    resume=resume,
                )

            # quiesce never pauses; a failed apply advances nothing and the gate
            # is cleared so admissions resume.
            self.assertEqual(events, ["apply"])
            self.assertFalse(gate._committing)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
