"""Pool-claim invariants, exercised on the minimal in-memory harness.

These are the guarantees the explicit claim/advance design is meant to provide,
phrased against the same primitives the real cookbooks use (board.claim /
board.advance + a WeightSyncManager pool). Each test reads as one invariant.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from cookbook.local_disagg.harness import (
    LocalReplica,
    LocalTrainer,
    make_pool,
    open_board,
    reconcile_pool,
)
from stitch.protocol import PointerRewind


class PoolClaimInvariantTest(unittest.TestCase):
    def test_claim_resets_pool_to_base(self) -> None:
        """A claim drops every replica to base (v0) under the new run's id —
        the explicit 'empty' starting state, written before any delta."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = open_board(tmp)
                trainer = LocalTrainer(board)
                pool = make_pool(board, size=3)

                move = trainer.claim()
                self.assertTrue(move.reset)
                self.assertEqual(move.version, 0)
                await reconcile_pool(pool)

                self.assertTrue(all(r.served_version == 0 for r in pool))
                self.assertTrue(all(r.served_run_id == trainer.run_id for r in pool))

        asyncio.run(run())

    def test_publish_advances_pool_monotonically(self) -> None:
        """Within a run the pool converges to each published version in order."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = open_board(tmp)
                trainer = LocalTrainer(board)
                pool = make_pool(board, size=2)
                trainer.claim()

                for expected in (1, 2, 3):
                    self.assertEqual(trainer.publish().version, expected)
                    await reconcile_pool(pool)
                    self.assertTrue(all(r.served_version == expected for r in pool))

                # The chain was replayed delta-by-delta, never skipped.
                self.assertEqual(pool[0].engine.applied, [1, 2, 3])

        asyncio.run(run())

    def test_new_run_resets_pool_even_from_higher_version(self) -> None:
        """A fresh run forks at base: the pool re-materializes base and replays
        the new chain, even though the finished run reached a higher version."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = open_board(tmp)
                pool = make_pool(board, size=2)

                old = LocalTrainer(board)
                old.claim()
                for _ in range(5):
                    old.publish()
                await reconcile_pool(pool)
                self.assertTrue(all(r.served_version == 5 for r in pool))

                new = LocalTrainer(board)
                self.assertNotEqual(new.run_id, old.run_id)
                new.claim()
                new.publish()  # the new run is only at v1
                await reconcile_pool(pool)

                self.assertTrue(all(r.served_version == 1 for r in pool))
                self.assertTrue(all(r.served_run_id == new.run_id for r in pool))
                # The drop from v5 to the new run's base went through an engine reset.
                self.assertTrue(all(r.engine.resets >= 1 for r in pool))

        asyncio.run(run())

    def test_restart_with_reused_run_id_is_rejected_as_rewind(self) -> None:
        """The restart hazard, made impossible: re-claiming a run already at the
        pointer is a rewind, not a silent stale-pointer reuse. A restart must
        mint a fresh run_id (a new epoch), which claims cleanly."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = open_board(tmp)
                crashed = LocalTrainer(board, run_id="run-fixed")
                crashed.claim()
                for _ in range(3):
                    crashed.publish()

                # Crash-restart that reused the same run_id would rewind latest
                # (v3 -> v0) onto stale weights — rejected.
                restarted_same = LocalTrainer(board, run_id="run-fixed")
                with self.assertRaises(PointerRewind):
                    restarted_same.claim()

                # The correct restart is a new epoch: a fresh run_id claims clean.
                pool = make_pool(board, size=2)
                restarted_fresh = LocalTrainer(board)
                restarted_fresh.claim()
                await reconcile_pool(pool)
                self.assertTrue(all(r.served_version == 0 for r in pool))
                self.assertTrue(all(r.served_run_id == restarted_fresh.run_id for r in pool))

        asyncio.run(run())

    def test_non_monotonic_publish_within_run_is_rejected(self) -> None:
        """Within a run the pointer only moves forward; re-advancing to an
        already-published version is a rewind."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = open_board(tmp)
                trainer = LocalTrainer(board)
                trainer.claim()
                trainer.publish()
                trainer.publish()  # at v2

                with self.assertRaises(PointerRewind):
                    board.advance(trainer.run_id, 2)
                with self.assertRaises(PointerRewind):
                    board.advance(trainer.run_id, 1)

        asyncio.run(run())

    def test_late_joining_replica_reconciles_to_current_run(self) -> None:
        """A replica that joins after the claim (a scaled-up / cold container)
        converges to the *current* run's chain, never a finished run's pointer."""

        async def run() -> None:
            with tempfile.TemporaryDirectory() as tmp:
                board = open_board(tmp)

                old = LocalTrainer(board)
                old.claim()
                for _ in range(4):
                    old.publish()

                new = LocalTrainer(board)
                new.claim()
                new.publish()
                new.publish()  # current run at v2

                # Replica created only now, with no prior state.
                latecomer = LocalReplica(board)
                await latecomer.reconcile()

                self.assertEqual(latecomer.served_version, 2)
                self.assertEqual(latecomer.served_run_id, new.run_id)
                self.assertEqual(latecomer.engine.applied, [1, 2])

        asyncio.run(run())

    def test_claim_requires_run_id(self) -> None:
        """A claim must name its run (the per-launch epoch token); an empty run
        id is a launch misconfiguration, not a usable claim."""
        with tempfile.TemporaryDirectory() as tmp:
            board = open_board(tmp)
            with self.assertRaises(ValueError):
                board.claim("")


if __name__ == "__main__":
    unittest.main()
