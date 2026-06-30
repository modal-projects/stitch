"""A minimal in-memory disaggregated-rollout harness.

Three pieces mirror the real cookbooks with none of the infrastructure:

- :class:`MemoryEngine` — an in-memory rollout server. It "serves" exactly one
  weight version; ``apply_manifest`` steps it forward one delta and ``reset``
  drops it back to base (v0). Stands in for the SGLang adapter.
- :class:`LocalReplica` — one ``WeightSyncManager`` driving one ``MemoryEngine``
  against a shared bulletin board. The pool is a list of these; each is a pure
  reconciler that converges to ``latest`` on ``reconcile()``.
- :class:`LocalTrainer` — the single writer for one run. It ``claim``s the pool
  (resets every replica to base) at launch, then ``publish``es monotonic delta
  versions. One trainer ↔ one run ↔ one pool epoch.

The trainer writes the *same* slime-layout chain the real disk-delta publisher
writes (``<root>/<run_id>/weight_v{N}/model.safetensors.index.json``), so the
replicas reconcile through the production ``WeightSyncManager`` path — only the
engine and the transport (a local dir instead of a Modal Volume / S3) are fakes.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from stitch.bulletin import FilesystemBulletinBoard
from stitch.protocol import BASE_VERSION, PointerMove, VersionManifest


class MemoryEngine:
    """In-memory rollout engine: tracks the single weight version it serves.

    base is v0; each ``apply_manifest`` advances exactly one delta and ``reset``
    re-materializes base. ``applied`` / ``resets`` are recorded so tests can
    assert the reconcile path (replay the chain, reset on a run switch).
    """

    backend = "memory"

    def __init__(self) -> None:
        self.version = 0
        self.applied: list[int] = []
        self.resets = 0

    async def prepare(self) -> None:
        pass

    async def flush_cache(self) -> None:
        pass

    async def apply_manifest(self, manifest: VersionManifest, version_path: str) -> None:
        self.version = manifest.version
        self.applied.append(manifest.version)

    async def reset(self) -> None:
        self.version = BASE_VERSION
        self.resets += 1

    async def pause_generation(self) -> None:
        pass

    async def continue_generation(self) -> None:
        pass


class LocalReplica:
    """One rollout-pool replica: a ``WeightSyncManager`` + its ``MemoryEngine``.

    Constructed lazily (the sync manager needs a running event loop), so the
    pool can be sized before any reconcile. ``reconcile`` converges the replica
    to the board's current ``(run_id, version)`` — following a run switch
    (reset → replay) exactly like the production sidecar.
    """

    def __init__(self, board: FilesystemBulletinBoard) -> None:
        from stitch.sync import WeightSyncManager

        self.engine = MemoryEngine()
        self.manager = WeightSyncManager(board=board, engine=self.engine, commit_mode="in_place")

    async def reconcile(self) -> None:
        await self.manager.sync_to()

    @property
    def served_version(self) -> int:
        return self.engine.version

    @property
    def served_run_id(self) -> str | None:
        return self.manager.current_run_id


class LocalTrainer:
    """The single writer for one run: claim the pool, then publish deltas.

    ``run_id`` defaults to a fresh per-launch token (the epoch/fence id that
    makes restart a clean new-run reset, never a colliding rewind). ``claim``
    writes the empty pointer; ``publish`` writes the next version dir and
    advances the pointer. Both go through the board's guarded writers, so a
    reused run_id or a non-monotonic publish raises rather than serving stale
    weights.
    """

    def __init__(self, board: FilesystemBulletinBoard, run_id: str | None = None) -> None:
        self.board = board
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.version = BASE_VERSION

    def claim(self) -> PointerMove:
        move = self.board.claim(self.run_id)
        self.version = BASE_VERSION
        return move

    def publish(self) -> PointerMove:
        """Materialize the next delta version and advance ``latest`` to it."""
        nxt = self.version + 1
        _write_version_dir(self.board.version_dir(self.run_id, nxt), version=nxt, base=self.version)
        move = self.board.advance(self.run_id, nxt)
        self.version = nxt
        return move


def make_pool(board: FilesystemBulletinBoard, size: int) -> list[LocalReplica]:
    return [LocalReplica(board) for _ in range(size)]


async def reconcile_pool(pool: list[LocalReplica]) -> None:
    for replica in pool:
        await replica.reconcile()


def open_board(root: str | Path) -> FilesystemBulletinBoard:
    """The slime-layout board both trainer and pool share (run-scoped chains)."""
    return FilesystemBulletinBoard(str(root), layout="slime")


def _write_version_dir(version_dir: Path, *, version: int, base: int) -> None:
    """Write the slime disk-delta version dir the real publisher would write:
    a canonical HF index whose metadata carries the version lineage."""
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"version": f"{version:06d}", "base_version": f"{base:06d}"},
                "weight_map": {"w": "model-00001-of-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
