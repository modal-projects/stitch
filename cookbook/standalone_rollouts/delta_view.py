"""Provider-owned decoder indexes and an immutable local version view."""

from __future__ import annotations

import errno
import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cookbook.standalone_rollouts.ledger import (
    IdentityLedger,
    LedgerEntry,
    is_valid_identity,
)
from stitch.protocol import atomic_write_text, weight_identity


INDEX_FILENAME = "model.safetensors.index.json"
DERIVED_INDEX_ROOT = Path(".stitch") / "deltas"


class DeltaIndexError(ValueError):
    """An uploaded or derived delta index is malformed."""


class DerivedDeltaConflict(DeltaIndexError):
    """A committed derived index differs from the current upload."""


@dataclass(frozen=True)
class DerivedDelta:
    index_path: Path
    shard_names: tuple[str, ...]


def derive_delta_index(
    transport_root: str | Path,
    entry: LedgerEntry,
    *,
    committed: bool,
) -> DerivedDelta:
    """Validate a raw upload and publish its deterministic decoder index.

    ``committed=False`` may replace an orphan left before a ledger save.
    ``committed=True`` accepts the exact existing artifact or recreates it when
    missing, but never changes one that belongs to a committed ledger entry.
    """
    _validate_delta_entry(entry)
    root = Path(transport_root)
    upload_dir = root / entry.identity
    _, weight_map, shard_names = _load_index(
        upload_dir / INDEX_FILENAME, shard_dir=upload_dir
    )
    derived_data = {
        "metadata": _expected_metadata(entry),
        "weight_map": weight_map,
    }
    contents = _canonical_json(derived_data)
    index_path = root / DERIVED_INDEX_ROOT / entry.identity / INDEX_FILENAME
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if committed:
        try:
            existing = index_path.read_bytes()
        except FileNotFoundError:
            pass
        else:
            if existing != contents.encode("utf-8"):
                raise DerivedDeltaConflict(
                    f"committed derived index for {entry.identity!r} does not match the upload"
                )
            return DerivedDelta(index_path=index_path, shard_names=shard_names)

    atomic_write_text(index_path, contents)
    return DerivedDelta(index_path=index_path, shard_names=shard_names)


class LocalDeltaView:
    """Materialize immutable slime ``weight_vN`` dirs from opaque uploads."""

    def __init__(self, root: str | Path, transport_root: str | Path) -> None:
        self.root = Path(root)
        self.transport_root = Path(transport_root)
        self._lock = threading.Lock()

    def rebuild(self, ledger: IdentityLedger) -> None:
        """Install every missing delta version; completed dirs never change."""
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            self._ensure_latest_link()
            for entry in ledger.delta_entries:
                self._install(entry)

    def _ensure_latest_link(self) -> None:
        link = self.root / "latest"
        target = (self.transport_root / "latest").absolute()
        if os.path.lexists(link):
            if link.is_symlink() and Path(os.readlink(link)) == target:
                return
            raise DeltaIndexError(
                f"local view path {link} is not the transport pointer link"
            )
        link.symlink_to(target)

    def _install(self, entry: LedgerEntry) -> None:
        _validate_delta_entry(entry)
        version_name = weight_identity(entry.version)
        destination = self.root / version_name
        if destination.is_symlink():
            raise DeltaIndexError(
                f"local view path {destination} must not be a symlink"
            )
        if destination.is_dir():
            return
        if os.path.lexists(destination):
            raise DeltaIndexError(f"local view path {destination} is not a directory")

        derived_path = (
            self.transport_root / DERIVED_INDEX_ROOT / entry.identity / INDEX_FILENAME
        )
        _, _, shard_names = _load_index(
            derived_path,
            shard_dir=self.transport_root / entry.identity,
            expected_metadata=_expected_metadata(entry),
            exact_top_level=True,
        )
        temporary = self.root / f".{version_name}.tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        try:
            shutil.copyfile(derived_path, temporary / INDEX_FILENAME)
            upload_dir = (self.transport_root / entry.identity).absolute()
            for shard_name in shard_names:
                (temporary / shard_name).symlink_to(upload_dir / shard_name)
            try:
                os.rename(temporary, destination)
            except OSError as exc:
                if (
                    exc.errno not in {errno.EEXIST, errno.ENOTEMPTY}
                    or not destination.is_dir()
                ):
                    raise
        finally:
            shutil.rmtree(temporary, ignore_errors=True)


def _expected_metadata(entry: LedgerEntry) -> dict[str, str]:
    if entry.is_base or entry.formats is None:
        raise ValueError("a derived delta index requires a non-base ledger entry")
    return {
        "version": f"{entry.version:06d}",
        "base_version": f"{entry.version - 1:06d}",
        **entry.formats.to_dict(),
    }


def _validate_delta_entry(entry: LedgerEntry) -> None:
    if (
        type(entry.version) is not int
        or entry.version <= 0
        or not is_valid_identity(entry.identity)
        or entry.formats is None
    ):
        raise DeltaIndexError("a derived index requires a validated delta ledger entry")


def _load_index(
    path: Path,
    *,
    shard_dir: Path,
    expected_metadata: dict[str, str] | None = None,
    exact_top_level: bool = False,
) -> tuple[dict[str, Any], dict[str, str], tuple[str, ...]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DeltaIndexError(f"{path} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise DeltaIndexError(f"{path} must contain a JSON object")
    if exact_top_level and set(data) != {"metadata", "weight_map"}:
        raise DeltaIndexError(f"{path} must contain only metadata and weight_map")
    if "metadata" not in data or not isinstance(data["metadata"], dict):
        raise DeltaIndexError(f"{path}.metadata must be an object")
    if expected_metadata is not None and data["metadata"] != expected_metadata:
        raise DeltaIndexError(f"{path}.metadata does not match the identity ledger")
    if "weight_map" not in data or not isinstance(data["weight_map"], dict):
        raise DeltaIndexError(f"{path}.weight_map must be an object")

    weight_map: dict[str, str] = {}
    for tensor_name, shard_name in data["weight_map"].items():
        if not _is_valid_text(tensor_name):
            raise DeltaIndexError(f"{path}.weight_map has an invalid tensor name")
        if not _is_safe_shard_name(shard_name):
            raise DeltaIndexError(
                f"{path}.weight_map[{tensor_name!r}] has unsafe shard {shard_name!r}"
            )
        weight_map[tensor_name] = shard_name

    shard_names = tuple(sorted(set(weight_map.values())))
    for shard_name in shard_names:
        shard_path = shard_dir / shard_name
        if shard_path.is_symlink():
            raise DeltaIndexError(
                f"referenced shard {shard_path} must not be a symlink"
            )
        try:
            with shard_path.open("rb"):
                pass
        except FileNotFoundError:
            raise FileNotFoundError(
                f"referenced shard {shard_path} is missing"
            ) from None
        except IsADirectoryError:
            raise DeltaIndexError(
                f"referenced shard {shard_path} is not a regular file"
            ) from None
    return data, weight_map, shard_names


def _is_valid_text(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return not any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in value
    )


def _is_safe_shard_name(value: object) -> bool:
    if not _is_valid_text(value):
        return False
    assert isinstance(value, str)
    return (
        len(value.encode("utf-8")) <= 255
        and "/" not in value
        and "\\" not in value
        and value not in {".", "..", INDEX_FILENAME}
        and value.endswith(".safetensors")
    )


def _canonical_json(data: dict[str, Any]) -> str:
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
