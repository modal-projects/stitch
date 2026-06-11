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
            protocol_version=int(data.get("protocol_version", PROTOCOL_VERSION)),
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


EXTRA_KEY_DELIMITER = ";"


def compose_extra_key(version: int, user_extra_key: str | None = None) -> str:
    """Compose a weight-version-namespaced engine ``extra_key``.

    The version segment sits at a fixed position (the prefix) and is
    delimiter-terminated, so it parses unambiguously regardless of the user
    key's content. sglang appends ``lora_id`` to ``extra_key`` with no
    delimiter, so the version must never be parsed from the right.
    Examples: ``wv7;`` (no user key), ``wv7;my-key``.
    """
    return f"wv{int(version)}{EXTRA_KEY_DELIMITER}{user_extra_key or ''}"


def parse_extra_key_version(extra_key: str) -> int | None:
    """Inverse of :func:`compose_extra_key`. None for non-composed keys."""
    if not extra_key.startswith("wv"):
        return None
    head, delim, _rest = extra_key.partition(EXTRA_KEY_DELIMITER)
    if not delim or not head[2:].isdigit():
        return None
    return int(head[2:])


def version_dir(root: str | Path, version: int) -> Path:
    return Path(root) / "versions" / f"weight_v{int(version):06d}"


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
