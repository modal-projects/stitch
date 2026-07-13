"""Strict mapping from opaque customer identities to stitch versions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.protocol import atomic_write_text


LEDGER_FILENAME = "identities.json"
LEDGER_SCHEMA_VERSION = 1
MAX_IDENTITY_BYTES = 255
RESERVED_IDENTITIES = frozenset({".", "..", ".stitch", "latest", LEDGER_FILENAME})


class LedgerError(ValueError):
    """Base class for invalid identity-ledger operations."""


class LedgerCorruption(LedgerError):
    """Persisted ledger data violates the schema or chain invariants."""


class LedgerConflict(LedgerError):
    """A new signal conflicts with the deployment's existing chain."""


class LedgerRewind(LedgerConflict):
    """A valid historical identity was signalled after the chain advanced."""


def is_valid_identity(identity: object) -> bool:
    """Return whether an opaque identity is safe as one filesystem component."""
    if not isinstance(identity, str) or identity in RESERVED_IDENTITIES:
        return False
    try:
        identity_bytes = identity.encode("utf-8")
    except UnicodeEncodeError:
        return False
    if not identity_bytes or len(identity_bytes) > MAX_IDENTITY_BYTES:
        return False
    if "/" in identity or "\\" in identity:
        return False
    return not any(
        ord(character) < 0x20 or ord(character) == 0x7F for character in identity
    )


@dataclass(frozen=True)
class DeltaFormats:
    """Decoder-relevant formats that must stay stable across POST retries."""

    delta_encoding: str
    compression_format: str
    checksum_format: str

    def __post_init__(self) -> None:
        supported = {
            "delta_encoding": {"xor"},
            "compression_format": {"zstd"},
            "checksum_format": {"adler32", "xxh3-128", "blake3"},
        }
        for name, allowed in supported.items():
            value = getattr(self, name)
            if not isinstance(value, str) or value not in allowed:
                raise ValueError(f"unsupported {name} {value!r}")

    @classmethod
    def defaults(cls) -> DeltaFormats:
        return cls(
            delta_encoding="xor",
            compression_format="zstd",
            checksum_format="adler32",
        )

    @classmethod
    def from_dict(cls, data: object) -> DeltaFormats:
        expected = {"delta_encoding", "compression_format", "checksum_format"}
        _require_exact_keys(data, expected, "formats")
        assert isinstance(data, dict)
        return cls(**{name: data[name] for name in expected})

    def to_dict(self) -> dict[str, str]:
        return {
            "delta_encoding": self.delta_encoding,
            "compression_format": self.compression_format,
            "checksum_format": self.checksum_format,
        }


@dataclass(frozen=True)
class LedgerEntry:
    version: int
    identity: str
    previous_snapshot_identity: str | None = None
    formats: DeltaFormats | None = None

    @property
    def is_base(self) -> bool:
        return self.version == 0


@dataclass(frozen=True)
class RecordResult:
    entry: LedgerEntry
    is_new: bool


class IdentityLedger:
    """One configured base followed by a contiguous append-only delta chain."""

    def __init__(self, entries: list[LedgerEntry]) -> None:
        self._entries = list(entries)
        self._by_identity = {entry.identity: entry for entry in entries}

    @classmethod
    def new(cls, base_identity: str) -> IdentityLedger:
        if not is_valid_identity(base_identity):
            raise ValueError("base identity must be one safe non-empty path component")
        return cls([LedgerEntry(version=0, identity=base_identity)])

    @classmethod
    def from_dict(cls, data: object, *, expected_base_identity: str) -> IdentityLedger:
        """Parse persisted state without coercion and validate every invariant."""
        try:
            _require_exact_keys(
                data, {"schema_version", "base_identity", "deltas"}, "ledger"
            )
            assert isinstance(data, dict)
            schema_version = data["schema_version"]
            if (
                type(schema_version) is not int
                or schema_version != LEDGER_SCHEMA_VERSION
            ):
                raise ValueError(f"unsupported schema_version {schema_version!r}")
            base_identity = data["base_identity"]
            if not is_valid_identity(base_identity):
                raise ValueError("base_identity is invalid")
            if base_identity != expected_base_identity:
                raise ValueError(
                    f"persisted base {base_identity!r} does not match configured base "
                    f"{expected_base_identity!r}"
                )
            raw_deltas = data["deltas"]
            if not isinstance(raw_deltas, list):
                raise ValueError("deltas must be a list")

            entries = [LedgerEntry(version=0, identity=base_identity)]
            identities = {base_identity}
            for version, raw_delta in enumerate(raw_deltas, start=1):
                label = f"deltas[{version - 1}]"
                _require_exact_keys(raw_delta, {"identity", "formats"}, label)
                assert isinstance(raw_delta, dict)
                identity = raw_delta["identity"]
                if not is_valid_identity(identity):
                    raise ValueError(f"{label}.identity is invalid")
                if identity in identities:
                    raise ValueError(f"identity {identity!r} appears more than once")
                entries.append(
                    LedgerEntry(
                        version=version,
                        identity=identity,
                        previous_snapshot_identity=entries[-1].identity,
                        formats=DeltaFormats.from_dict(raw_delta["formats"]),
                    )
                )
                identities.add(identity)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, LedgerCorruption):
                raise
            raise LedgerCorruption(f"invalid {LEDGER_FILENAME}: {exc}") from exc
        return cls(entries)

    @property
    def base_identity(self) -> str:
        return self._entries[0].identity

    @property
    def head(self) -> LedgerEntry:
        return self._entries[-1]

    @property
    def head_version(self) -> int:
        return self.head.version

    @property
    def delta_entries(self) -> tuple[LedgerEntry, ...]:
        return tuple(self._entries[1:])

    def version_for(self, identity: str) -> int | None:
        entry = self._by_identity.get(identity)
        return entry.version if entry is not None else None

    def identity_for(self, version: int) -> str | None:
        if type(version) is not int or version < 0 or version >= len(self._entries):
            return None
        return self._entries[version].identity

    def confirm_base(self, identity: str) -> RecordResult:
        """Accept the configured base only while it remains the current head."""
        if identity != self.base_identity:
            raise LedgerConflict(
                f"this deployment serves base {self.base_identity!r}, not {identity!r}"
            )
        if self.head_version != 0:
            raise LedgerRewind(
                f"base {identity!r} is older than current identity {self.head.identity!r}"
            )
        return RecordResult(self._entries[0], is_new=False)

    def append_delta(
        self,
        identity: str,
        previous_snapshot_identity: str,
        formats: DeltaFormats,
    ) -> RecordResult:
        """Append one delta or accept an exact retry of the current head."""
        if not is_valid_identity(identity) or not is_valid_identity(
            previous_snapshot_identity
        ):
            raise LedgerConflict(
                "identity and previous identity must be safe path components"
            )
        if not isinstance(formats, DeltaFormats):
            raise LedgerConflict("formats must be validated DeltaFormats")

        existing = self._by_identity.get(identity)
        if existing is not None:
            if existing.is_base:
                raise LedgerConflict("the configured base cannot also be a delta")
            if (
                existing.previous_snapshot_identity != previous_snapshot_identity
                or existing.formats != formats
            ):
                raise LedgerConflict(
                    f"retry for identity {identity!r} does not match its recorded lineage and formats"
                )
            if existing.version != self.head_version:
                raise LedgerRewind(
                    f"identity {identity!r} is older than current identity {self.head.identity!r}"
                )
            return RecordResult(existing, is_new=False)

        if previous_snapshot_identity != self.head.identity:
            raise LedgerConflict(
                f"previous_snapshot_identity {previous_snapshot_identity!r} must equal "
                f"current identity {self.head.identity!r}"
            )
        entry = LedgerEntry(
            version=self.head_version + 1,
            identity=identity,
            previous_snapshot_identity=previous_snapshot_identity,
            formats=formats,
        )
        self._entries.append(entry)
        self._by_identity[identity] = entry
        return RecordResult(entry, is_new=True)

    def to_dict(self) -> dict[str, Any]:
        deltas: list[dict[str, Any]] = []
        for entry in self.delta_entries:
            assert entry.formats is not None
            deltas.append(
                {"identity": entry.identity, "formats": entry.formats.to_dict()}
            )
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "base_identity": self.base_identity,
            "deltas": deltas,
        }


def load_ledger_data(transport_root: str | Path) -> object | None:
    """Read persisted JSON directly, or return ``None`` when it is absent."""
    path = Path(transport_root) / LEDGER_FILENAME
    try:
        contents = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    try:
        return json.loads(contents)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise LedgerCorruption(f"invalid {LEDGER_FILENAME}: {exc}") from exc


def save_ledger_data(transport_root: str | Path, data: dict[str, Any]) -> None:
    """Persist ledger JSON through the Mountpoint-safe control-file writer."""
    contents = json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    atomic_write_text(Path(transport_root) / LEDGER_FILENAME, contents + "\n")


def _require_exact_keys(data: object, expected: set[str], label: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be an object")
    actual = set(data)
    if actual != expected:
        raise ValueError(
            f"{label} fields must be {sorted(expected)!r}, got {sorted(actual)!r}"
        )
