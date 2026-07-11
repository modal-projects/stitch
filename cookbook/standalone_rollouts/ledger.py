"""Identity ledger: opaque customer checkpoint identities -> stitch versions.

The customer's hot-load API names checkpoints by an opaque ``<identity>`` and
gives lineage only via ``incremental_snapshot_metadata.previous_snapshot_identity``.
stitch's core is an integer-versioned monotonic log. The front door bridges the
two: it is the single writer (under its advance lock) of this ledger, minting a
monotonic version per newly-signalled identity and recording the parent identity
so a version dir's index can carry ``base_version``.

Because versions only ever increase, two identities never collide on a number
and ``run_id`` is unnecessary. Re-signalling a known identity returns its
existing version (idempotent — a retried POST is not a rewind). A delta whose
parent is the immediately-preceding identity forms the contiguous chain the
apply path replays; a delta against an older ancestor (training resume against
the base) is recorded faithfully but is not yet replayable (the deferred
fork-from-version case), so its non-contiguous base surfaces at apply time
rather than being silently reordered here.

Pure and persistence-free: :meth:`to_dict` / :meth:`from_dict` round-trip the
state, and the front door writes it to the transport with a rename-free write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stitch.protocol import BASE_VERSION


@dataclass(frozen=True)
class LedgerEntry:
    """One signalled checkpoint: the version minted for it and its parent
    identity (``None`` for a full snapshot / base, which has no delta parent)."""

    version: int
    previous: str | None


class IdentityLedger:
    def __init__(self, entries: dict[str, LedgerEntry] | None = None) -> None:
        self._by_identity: dict[str, LedgerEntry] = dict(entries or {})
        self._by_version: dict[int, str] = {
            entry.version: identity for identity, entry in self._by_identity.items()
        }

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

    def base_version_for(self, identity: str) -> int:
        """The version a checkpoint's delta builds on: its parent's version, or
        ``BASE_VERSION`` when it is a full snapshot or the parent is unknown."""
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
        """
        existing = self._by_identity.get(identity)
        if existing is not None:
            return existing, False
        version = BASE_VERSION if not self._by_version else max(self._by_version) + 1
        entry = LedgerEntry(version=version, previous=previous)
        self._by_identity[identity] = entry
        self._by_version[version] = identity
        return entry, True
