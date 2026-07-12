"""Identity ledger: opaque customer checkpoint identities -> stitch versions.

The customer's hot-load API names checkpoints by an opaque ``<identity>`` and
gives lineage only via ``incremental_snapshot_metadata.previous_snapshot_identity``.
stitch's core is an integer-versioned monotonic log. The front door bridges the
two: it is the single writer (under its advance lock) of this ledger, minting a
monotonic version per newly-signalled identity and recording the parent identity
so a version dir's index can carry ``base_version``.

Because versions only ever increase, two identities never collide on a number
and ``run_id`` is unnecessary. Re-signalling a known identity returns its
existing version (idempotent — a retried POST is not a rewind). Every delta
must extend the chain head: the apply path replays a contiguous chain, so a
delta against an older ancestor (a fork/resume) is rejected at signal time —
once minted, a non-contiguous version would sit in the log forever and block
every replica from serving anything published after it.

The ledger itself is pure: :meth:`to_dict` / :meth:`from_dict` round-trip the
state. Persistence is one JSON file on the transport (``LEDGER_FILENAME``,
written by the front door, read back via :func:`load_ledger_dict`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stitch.protocol import BASE_VERSION


LEDGER_FILENAME = "identities.json"


def load_ledger_dict(transport_root: str | Path) -> dict[str, Any]:
    """The persisted ledger dict from the transport, or ``{}`` before the
    front door's first save."""
    try:
        path = Path(transport_root) / LEDGER_FILENAME
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


class LedgerError(ValueError):
    """A signal whose lineage cannot be recorded (second full snapshot, unknown
    parent, or fork from a non-head checkpoint) or a persisted ledger that
    violates an invariant. The front door reports it as a 409."""


@dataclass(frozen=True)
class LedgerEntry:
    """One signalled checkpoint: the version minted for it and its parent
    identity (``None`` for a full snapshot / base, which has no delta parent)."""

    version: int
    previous: str | None


class IdentityLedger:
    def __init__(self, entries: dict[str, LedgerEntry] | None = None) -> None:
        self._by_identity: dict[str, LedgerEntry] = dict(entries or {})
        self._by_version: dict[int, str] = {}
        for identity, entry in self._by_identity.items():
            other = self._by_version.get(entry.version)
            if other is not None:
                # A persisted ledger that violates the one-identity-per-version
                # invariant must fail loudly, not silently collapse the reverse
                # map onto whichever entry deserialized last.
                raise LedgerError(
                    f"ledger maps both {other!r} and {identity!r} to version {entry.version}"
                )
            self._by_version[entry.version] = identity

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentityLedger":
        entries = {
            str(identity): LedgerEntry(
                version=int(record["version"]),
                previous=(None if record.get("previous") is None else str(record["previous"])),
            )
            for identity, record in (data.get("entries") or {}).items()
        }
        return cls(entries)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": {
                identity: {"version": entry.version, "previous": entry.previous}
                for identity, entry in self._by_identity.items()
            }
        }

    @property
    def head_version(self) -> int | None:
        """The highest minted version (the chain head), or None when empty. The
        front door re-advances the pointer to the head even on an idempotent
        re-signal, so a POST whose save landed but whose pointer write failed
        converges on retry."""
        return max(self._by_version) if self._by_version else None

    def version_for(self, identity: str) -> int | None:
        entry = self._by_identity.get(identity)
        return entry.version if entry is not None else None

    def identity_for(self, version: int) -> str | None:
        """Reverse lookup, for translating readiness back to the customer's
        identity string and for resolving a version's on-disk upload dir."""
        return self._by_version.get(int(version))

    def items_by_version(self) -> list[tuple[int, str]]:
        """``(version, identity)`` pairs in version order, for building the
        sidecar's ``weight_vN`` -> identity-dir symlink view."""
        return sorted(self._by_version.items())

    def base_version_for(self, identity: str) -> int:
        """The version a checkpoint's delta builds on: its parent's version, or
        ``BASE_VERSION`` for a full snapshot / the first delta's unsignalled
        (booted) parent — the one case :meth:`record` admits an unknown parent."""
        entry = self._by_identity[identity]
        if entry.previous is None:
            return BASE_VERSION
        parent = self._by_identity.get(entry.previous)
        return parent.version if parent is not None else BASE_VERSION

    def record(self, identity: str, previous: str | None) -> tuple[LedgerEntry, bool]:
        """Mint a version for ``identity`` (or return its existing entry).

        Returns ``(entry, is_new)``. A known identity is idempotent: its entry is
        returned unchanged and ``is_new`` is False, so a retried signal never
        moves the pointer. A new identity is minted the next monotonic version
        (``BASE_VERSION`` for the first ever, else max+1) and records ``previous``.

        Raises :class:`LedgerError` on lineage the pool could not serve: a
        second full snapshot (the base slot is single-occupancy), a parent this
        ledger has never seen (typo'd or lost signals would otherwise be applied
        against the wrong base weights), or a known parent that is not the chain
        head (fork/resume — unsupported, and a non-contiguous version would
        permanently block the replay of everything after it). The one sanctioned
        unknown parent is a first delta on an empty ledger, whose base booted
        from BASE_CHECKPOINT and was never signalled.
        """
        existing = self._by_identity.get(identity)
        if existing is not None:
            return existing, False
        if previous is None:
            claimed = self._by_version.get(BASE_VERSION)
            if claimed is not None:
                raise LedgerError(
                    f"base already signalled as {claimed!r}; cannot record a second full snapshot"
                )
            version = BASE_VERSION
        else:
            parent = self._by_identity.get(previous)
            if parent is None and self._by_identity:
                raise LedgerError(
                    f"previous_snapshot_identity {previous!r} was never signalled; "
                    "signal the parent checkpoint first"
                )
            if parent is not None and parent.version != max(self._by_version):
                raise LedgerError(
                    f"previous_snapshot_identity {previous!r} is not the latest checkpoint; "
                    "forking from an older checkpoint is not supported"
                )
            # A delta always mints a real version >= 1; v0 is reserved for the
            # base, so a delta signalled before any base (its parent booted, not
            # signalled) still becomes v1, not a phantom base that never applies.
            deltas = [v for v in self._by_version if v > BASE_VERSION]
            version = (max(deltas) + 1) if deltas else BASE_VERSION + 1
        entry = LedgerEntry(version=version, previous=previous)
        self._by_identity[identity] = entry
        self._by_version[version] = identity
        return entry, True
