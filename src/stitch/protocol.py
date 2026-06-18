"""Core wire protocol helpers for disaggregated rollout weight sync."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


PROTOCOL_VERSION = 1
LATEST_FILE = "latest.json"


class SyncState(str, Enum):
    IDLE = "IDLE"
    QUEUED = "QUEUED"
    PREFETCHING = "PREFETCHING"
    PREPARING = "PREPARING"
    COMMITTING = "COMMITTING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class RolloutReplicaState:
    """Readiness report for one rollout server replica.

    Providers may identify weights by an integer stitch version, an external
    snapshot identity, or both. Readiness is separate from identity matching:
    a healthy replica on an old snapshot is observable, but it is not ready for
    requests that require the new target.
    """

    readiness: bool
    current_version: int | None = None
    current_snapshot_identity: str | None = None
    replica_id: str | None = None
    sync_state: str | None = None
    readiness_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutReplicaState":
        known = {
            "readiness",
            "current_version",
            "current_snapshot_identity",
            "replica_id",
            "sync_state",
            "readiness_reason",
            "metadata",
        }
        metadata = {k: v for k, v in data.items() if k not in known}
        metadata.update(dict(data.get("metadata") or {}))
        return cls(
            readiness=_bool(data.get("readiness", False)),
            current_version=_optional_int(data.get("current_version")),
            current_snapshot_identity=_optional_str(data.get("current_snapshot_identity")),
            replica_id=_optional_str(data.get("replica_id")),
            sync_state=_optional_str(data.get("sync_state")),
            readiness_reason=_optional_str(data.get("readiness_reason")),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"readiness": self.readiness}
        if self.current_version is not None:
            data["current_version"] = self.current_version
        if self.current_snapshot_identity is not None:
            data["current_snapshot_identity"] = self.current_snapshot_identity
        if self.replica_id is not None:
            data["replica_id"] = self.replica_id
        if self.sync_state is not None:
            data["sync_state"] = self.sync_state
        if self.readiness_reason is not None:
            data["readiness_reason"] = self.readiness_reason
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def matches_target(
        self,
        *,
        target_version: int | None = None,
        target_snapshot_identity: str | None = None,
    ) -> bool:
        if not self.readiness:
            return False
        if target_version is not None and self.current_version != int(target_version):
            return False
        if target_snapshot_identity is not None and self.current_snapshot_identity != str(target_snapshot_identity):
            return False
        return True


@dataclass(frozen=True)
class RolloutPoolState:
    """Readiness report for a rollout server pool."""

    replicas: list[RolloutReplicaState] = field(default_factory=list)
    protocol_version: int = PROTOCOL_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RolloutPoolState":
        known = {"protocol_version", "replicas", "metadata"}
        metadata = {k: v for k, v in data.items() if k not in known}
        metadata.update(dict(data.get("metadata") or {}))
        return cls(
            protocol_version=int(data.get("protocol_version", PROTOCOL_VERSION)),
            replicas=[RolloutReplicaState.from_dict(x) for x in data.get("replicas", [])],
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "protocol_version": int(self.protocol_version),
            "replicas": [replica.to_dict() for replica in self.replicas],
        }
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def ready_count(
        self,
        *,
        target_version: int | None = None,
        target_snapshot_identity: str | None = None,
    ) -> int:
        return sum(
            replica.matches_target(
                target_version=target_version,
                target_snapshot_identity=target_snapshot_identity,
            )
            for replica in self.replicas
        )

    def readiness_fraction(
        self,
        *,
        target_version: int | None = None,
        target_snapshot_identity: str | None = None,
    ) -> float:
        if not self.replicas:
            return 0.0
        return self.ready_count(
            target_version=target_version,
            target_snapshot_identity=target_snapshot_identity,
        ) / len(self.replicas)

    def is_ready(
        self,
        *,
        threshold: float = 1.0,
        target_version: int | None = None,
        target_snapshot_identity: str | None = None,
    ) -> bool:
        return bool(self.replicas) and self.readiness_fraction(
            target_version=target_version,
            target_snapshot_identity=target_snapshot_identity,
        ) >= float(threshold)


@dataclass(frozen=True)
class WeightVersionPolicy:
    """Request-level policy for acceptable rollout server weights."""

    min_required_version: int | None = None
    exact_version: int | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "WeightVersionPolicy":
        raw = (payload or {}).get("weight_version") or {}
        if not isinstance(raw, dict):
            raw = {}
        return cls(
            min_required_version=_optional_int(raw.get("min_required_version")),
            exact_version=_optional_int(raw.get("exact_version")),
        )

    def to_payload(self) -> dict[str, int | None]:
        return {
            "min_required_version": self.min_required_version,
            "exact_version": self.exact_version,
        }


@dataclass(frozen=True)
class Artifact:
    """An immutable artifact referenced by a version manifest."""

    kind: str
    path: str
    size_bytes: int | None = None
    checksum: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Artifact":
        return cls(
            kind=str(data["kind"]),
            path=str(data["path"]),
            size_bytes=_optional_int(data.get("size_bytes")),
            checksum=None if data.get("checksum") is None else str(data["checksum"]),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
        }
        if self.size_bytes is not None:
            data["size_bytes"] = self.size_bytes
        if self.checksum is not None:
            data["checksum"] = self.checksum
        if self.metadata:
            data["metadata"] = self.metadata
        return data


@dataclass(frozen=True)
class VersionManifest:
    """Published description of one trainer-to-rollout weight version."""

    version: int
    base_version: int
    backend: str
    load_format: str
    transition_files: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    protocol_version: int = PROTOCOL_VERSION
    artifacts: list[Artifact] = field(default_factory=list)
    run_id: str | None = None
    base_model: str | None = None
    recovery: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def read(cls, path: str | Path) -> "VersionManifest":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        protocol_version = int(data.get("protocol_version", PROTOCOL_VERSION))
        if protocol_version != PROTOCOL_VERSION:
            raise ValueError(
                f"manifest at {path} declares protocol_version {protocol_version}, "
                f"but this build supports {PROTOCOL_VERSION}; refusing to read it"
            )
        artifacts = [Artifact.from_dict(x) for x in data.get("artifacts", [])]
        transition_files = [str(x) for x in data.get("transition_files", [])]
        if not transition_files:
            transition_files = [a.path for a in artifacts if a.kind == "transition"]
        return cls(
            version=int(data["version"]),
            base_version=int(data["base_version"]),
            backend=str(data.get("backend", "sparse_delta")),
            load_format=str(data.get("load_format", "delta")),
            transition_files=transition_files,
            created_at=float(data.get("created_at", 0.0)),
            protocol_version=protocol_version,
            artifacts=artifacts,
            run_id=None if data.get("run_id") is None else str(data["run_id"]),
            base_model=None if data.get("base_model") is None else str(data["base_model"]),
            recovery=data.get("recovery"),
            metadata=dict(data.get("metadata") or {}),
        )

    def write(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        artifacts = self.artifacts or [Artifact(kind="transition", path=p) for p in self.transition_files]
        data: dict[str, Any] = {
            "protocol_version": int(self.protocol_version),
            "version": int(self.version),
            "base_version": int(self.base_version),
            "backend": self.backend,
            "load_format": self.load_format,
            "transition_files": list(self.transition_files),
            "artifacts": [a.to_dict() for a in artifacts],
            "created_at": float(self.created_at),
        }
        if self.run_id is not None:
            data["run_id"] = self.run_id
        if self.base_model is not None:
            data["base_model"] = self.base_model
        if self.recovery is not None:
            data["recovery"] = self.recovery
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def transition_artifact_paths(self) -> list[str]:
        if self.transition_files:
            return list(self.transition_files)
        return [a.path for a in self.artifacts if a.kind == "transition"]


def weight_identity(version: int) -> str:
    """Canonical snapshot-identity string for an integer weight version."""
    return f"weight_v{int(version):06d}"


def parse_weight_identity(identity: str) -> int | None:
    """Inverse of :func:`weight_identity`; None if not ``weight_v<digits>``."""
    prefix = "weight_v"
    if not identity.startswith(prefix):
        return None
    digits = identity[len(prefix):]
    return int(digits) if digits.isdigit() else None


def version_dir(root: str | Path, version: int) -> Path:
    return Path(root) / "versions" / weight_identity(version)


def latest_path(root: str | Path) -> Path:
    return Path(root) / LATEST_FILE


def read_latest(root: str | Path) -> int:
    path = latest_path(root)
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("version", 0))


def write_latest(root: str | Path, version: int) -> None:
    atomic_write_json(
        latest_path(root),
        {
            "protocol_version": PROTOCOL_VERSION,
            "version": int(version),
            "updated_at": time.time(),
        },
    )


def version_not_ready_error(current: int, target: int) -> dict[str, Any]:
    return {
        "error": {
            "type": "WeightVersionNotReady",
            "message": f"server is at version {current}, target {target} is not ready",
            "current_version": int(current),
            "target_version": int(target),
        }
    }


def version_too_old_error(current: int, target: int) -> dict[str, Any]:
    return {
        "error": {
            "type": "WeightVersionTooOld",
            "message": f"server is at version {current}, cannot roll back to {target}",
            "current_version": int(current),
            "target_version": int(target),
        }
    }


def evaluate_version_policy(
    current_version: int, policy: WeightVersionPolicy
) -> dict[str, Any] | None:
    """Shared exact/min admission check. Returns a typed error dict or None.

    Callers decide how to react to a not-ready error (pull toward the target vs
    reject): the bulletin-board manager queues a sync, the hot-load shim rejects.
    """
    if policy.exact_version is not None:
        target = int(policy.exact_version)
        if current_version < target:
            return version_not_ready_error(current_version, target)
        if current_version > target:
            return version_too_old_error(current_version, target)
        return None
    if policy.min_required_version is not None and current_version < int(
        policy.min_required_version
    ):
        return version_not_ready_error(current_version, int(policy.min_required_version))
    return None


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)
