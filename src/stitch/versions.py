"""stitch domain vocabulary: version types, the request constraint, replica/pool
readiness, and the pure pointer rules.

No I/O, no Modal, no engine, no framework. Every other module depends on this,
and this depends on nothing else in stitch. The Store/Engine/Pool ports live
with their instances (``stores/base.py``, ``engines/base.py``, ``pools/base.py``).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

_WEIGHT_PREFIX = "weight_v"


@dataclass(frozen=True)
class VersionRef:
    """A published policy version: a run epoch (``run_id``) and a monotonic number.

    Its string form — ``<run_id>/weight_v000007``, or bare ``weight_v000007``
    run-less — is self-identifying, so one run's chain can't be mistaken for
    another's; external specs call this the checkpoint identity.
    """

    run_id: str | None
    version: int

    @property
    def identity(self) -> str:
        name = f"{_WEIGHT_PREFIX}{self.version:06d}"
        return f"{self.run_id}/{name}" if self.run_id else name

    @classmethod
    def parse(cls, text: str) -> "VersionRef":
        text = (text or "").strip()
        if not text:
            return cls(None, 0)  # no valid pointer -> treated as not-ready, not a misparse
        run_id, _, tail = text.rpartition("/")
        digits = tail[len(_WEIGHT_PREFIX):] if tail.startswith(_WEIGHT_PREFIX) else tail
        return cls(run_id or None, int(digits) if digits.isdigit() else 0)


class VersionKind(str, Enum):
    FULL = "full"    # an anchor: the version's files are the weights
    DELTA = "delta"  # a diff against base_version (same run), chaining back to an anchor


@dataclass(frozen=True)
class VersionManifest:
    """One published version, derived from its directory's HF index (never stored
    separately). ``kind`` alone decides how a replica applies it: FULL seeds from
    ``files``; DELTA decodes (``delta_encoding`` / ``compression`` / ``checksum``) against
    ``base_version``, which chains back to the nearest FULL anchor.
    """

    ref: VersionRef
    kind: VersionKind
    files: list[str]
    base_version: int | None = None   # required iff DELTA; always the same run as ref
    delta_encoding: str | None = None
    compression: str | None = None
    checksum: str | None = None
    base_model: str | None = None
    created_at: float = 0.0

    @classmethod
    def from_hf_index(
        cls,
        version_dir: str | Path,
        *,
        run_id: str | None = None,
        base_model: str | None = None,
    ) -> "VersionManifest":
        # The dir's model.safetensors.index.json carries the lineage + codec under
        # `metadata` and the files as `weight_map` values. A `diff` key => DELTA.
        index = json.loads((Path(version_dir) / "model.safetensors.index.json").read_text())
        meta = index.get("metadata") or {}
        weight_map = index.get("weight_map") or {}
        diff = meta.get("diff") or None
        base = meta.get("base_version")
        return cls(
            ref=VersionRef(run_id, int(meta["version"])),
            kind=VersionKind.DELTA if diff else VersionKind.FULL,
            files=sorted({str(f) for f in weight_map.values()}),
            base_version=int(base) if diff and base is not None else None,
            delta_encoding=diff,
            compression=meta.get("compression") or None,
            checksum=meta.get("checksum") or None,
            base_model=base_model,
            created_at=float(meta.get("created_at", time.time())),
        )


@dataclass(frozen=True)
class VersionConstraint:
    """What a rollout request requires of the serving version. ``exact_version``
    pins one version; ``min_version`` is a floor (a bounded-lag request sets it to
    ``latest - lag``). Both None means no constraint.
    """

    min_version: int | None = None
    exact_version: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "VersionConstraint":
        raw = (payload or {}).get("weight_version")
        raw = raw if isinstance(raw, dict) else {}
        mn, ex = raw.get("min_version"), raw.get("exact_version")
        return cls(int(mn) if mn is not None else None, int(ex) if ex is not None else None)

    def to_payload(self) -> dict[str, int | None]:
        return {"min_version": self.min_version, "exact_version": self.exact_version}

    def satisfied_by(self, applied: int | None) -> bool:
        if applied is None:
            return self.min_version is None and self.exact_version is None
        if self.exact_version is not None:
            return applied == self.exact_version
        return self.min_version is None or applied >= self.min_version


class SyncState(str, Enum):
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    PREFETCHING = "PREFETCHING"
    PREPARING = "PREPARING"
    COMMITTING = "COMMITTING"
    ERROR = "ERROR"


_SYNC_STATES = {s.value for s in SyncState}


@dataclass(frozen=True)
class ReplicaState:
    """One replica's readiness report — the body of its ``server_info``."""

    ready: bool
    applied: VersionRef | None = None
    sync_state: SyncState | None = None
    reason: str | None = None  # why not ready, surfaced to the readiness poll

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReplicaState":
        applied, state = data.get("applied"), data.get("sync_state")
        return cls(
            ready=bool(data.get("ready", False)),
            applied=VersionRef.parse(applied) if applied else None,
            sync_state=SyncState(state) if state in _SYNC_STATES else None,
            reason=data.get("reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"ready": self.ready}
        if self.applied is not None:
            data["applied"] = self.applied.identity
        if self.sync_state is not None:
            data["sync_state"] = self.sync_state.value
        if self.reason is not None:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True)
class PoolState:
    """Aggregate readiness across a pool's replicas; drives the readiness poll."""

    replicas: list[ReplicaState]

    def ready_fraction(self, target: VersionRef) -> float:
        if not self.replicas:
            return 0.0
        return sum(r.ready and r.applied == target for r in self.replicas) / len(self.replicas)

    def is_ready(self, target: VersionRef, *, threshold: float = 1.0) -> bool:
        return bool(self.replicas) and self.ready_fraction(target) >= threshold


class PointerRewind(Exception):
    """A move that would rewind ``latest`` within the same run. The single writer
    advances monotonically per run; crossing to a *different* run forks at base
    and is not a rewind."""

    def __init__(self, current: VersionRef, proposed: VersionRef) -> None:
        super().__init__(
            f"latest is {current.identity!r}; refusing to rewind to {proposed.identity!r}"
        )
        self.current, self.proposed = current, proposed


@dataclass(frozen=True)
class PointerMove:
    ref: VersionRef
    reset: bool  # crossed to a new run -> re-materialize base and restart the version space


def decide_pointer_move(current: VersionRef | None, proposed: VersionRef) -> PointerMove:
    """A different run forks at base (a reset, even at a lower number); within the
    same run the move must be strictly newer, else :class:`PointerRewind`."""
    current_run = current.run_id if current is not None else None
    if proposed.run_id != current_run:
        return PointerMove(proposed, reset=True)
    if current is not None and proposed.version <= current.version:
        raise PointerRewind(current, proposed)
    return PointerMove(proposed, reset=False)
