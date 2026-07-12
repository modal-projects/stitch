"""Pure rules for the fixed-base opaque-identity hot-load contract."""

from dataclasses import dataclass
from typing import Any

from cookbook.standalone_rollouts.ledger import (
    DeltaFormats,
    IdentityLedger,
    LedgerCorruption,
    is_valid_identity,
)
from stitch.protocol import RolloutPoolState, RolloutReplicaState


class HotLoadRequestError(ValueError):
    """The JSON body does not match the fixed customer contract."""


@dataclass(frozen=True)
class HotLoadSignal:
    identity: str
    previous_snapshot_identity: str | None
    formats: DeltaFormats | None

    @property
    def is_delta(self) -> bool:
        return self.previous_snapshot_identity is not None


@dataclass(frozen=True)
class FrontdoorRecovery:
    ledger: IdentityLedger
    save_ledger: bool
    pointer_to_write: int | None


def parse_hot_load_payload(payload: object) -> HotLoadSignal:
    """Parse the customer body strictly; no string or number coercion."""
    _require_fields(
        payload,
        required={"identity"},
        optional={"incremental_snapshot_metadata", "reset_prompt_cache"},
        label="body",
    )
    assert isinstance(payload, dict)
    identity = payload["identity"]
    if not is_valid_identity(identity):
        raise HotLoadRequestError(
            "body.identity must be one non-empty filesystem-safe string"
        )

    if (
        "reset_prompt_cache" in payload
        and payload["reset_prompt_cache"] != "new_session"
    ):
        raise HotLoadRequestError(
            "body.reset_prompt_cache, when present, must be 'new_session'"
        )

    if "incremental_snapshot_metadata" not in payload:
        return HotLoadSignal(identity, previous_snapshot_identity=None, formats=None)

    incremental = payload["incremental_snapshot_metadata"]
    _require_fields(
        incremental,
        required={"previous_snapshot_identity"},
        optional={"compression_format", "checksum_format"},
        label="body.incremental_snapshot_metadata",
    )
    assert isinstance(incremental, dict)
    previous = incremental["previous_snapshot_identity"]
    if not is_valid_identity(previous):
        raise HotLoadRequestError(
            "incremental_snapshot_metadata.previous_snapshot_identity must be "
            "one non-empty filesystem-safe string"
        )
    values = {
        "delta_encoding": "xor",
        "compression_format": incremental.get("compression_format", "zstd"),
        "checksum_format": incremental.get("checksum_format", "adler32"),
    }
    if not all(isinstance(value, str) for value in values.values()):
        raise HotLoadRequestError("delta format fields must be strings")
    try:
        formats = DeltaFormats(**values)
    except ValueError as exc:
        raise HotLoadRequestError(str(exc)) from exc
    return HotLoadSignal(identity, previous_snapshot_identity=previous, formats=formats)


def recover_frontdoor_state(
    *,
    persisted_ledger: object | None,
    expected_base_identity: str,
    pointer: tuple[str | None, int],
) -> FrontdoorRecovery:
    """Validate durable state and describe the minimal startup repair.

    The ledger is the commit point. A pointer behind it is repaired to the head;
    a pointer ahead of it cannot be explained by the transaction ordering and is
    corruption. A missing ledger is seeded only when the pool is still at v0.
    """
    run_id, pointer_version = pointer
    if run_id is not None:
        raise LedgerCorruption("standalone rollout pointers must not be run-scoped")
    if type(pointer_version) is not int or pointer_version < 0:
        raise LedgerCorruption(f"invalid latest pointer version {pointer_version!r}")

    if persisted_ledger is None:
        if pointer_version != 0:
            raise LedgerCorruption(
                "identity ledger is missing while latest points above the configured base"
            )
        return FrontdoorRecovery(
            ledger=IdentityLedger.new(expected_base_identity),
            save_ledger=True,
            pointer_to_write=0,
        )

    ledger = IdentityLedger.from_dict(
        persisted_ledger, expected_base_identity=expected_base_identity
    )
    if pointer_version > ledger.head_version:
        raise LedgerCorruption(
            f"latest points to v{pointer_version}, ahead of ledger head v{ledger.head_version}"
        )
    pointer_to_write = (
        ledger.head_version if pointer_version < ledger.head_version else None
    )
    return FrontdoorRecovery(
        ledger=ledger,
        save_ledger=False,
        pointer_to_write=pointer_to_write,
    )


def pool_state_from_server_infos(
    infos: list[dict[str, Any]], ledger: IdentityLedger
) -> RolloutPoolState:
    """Translate replica integer versions back to opaque customer identities."""
    replicas: list[RolloutReplicaState] = []
    for info in infos:
        raw_version = info.get("current_version")
        version = raw_version if type(raw_version) is int and raw_version >= 0 else None
        identity = ledger.identity_for(version) if version is not None else None
        sync_state = info.get("sync_state")
        last_error = info.get("last_sync_error")
        current_run_id = info.get("current_run_id")

        reason: str | None = None
        if last_error:
            reason = str(last_error)
        elif sync_state != "IDLE":
            reason = str(sync_state or "unreachable")
        elif current_run_id is not None:
            reason = "replica is serving an unexpected run-scoped chain"
        elif identity is None:
            reason = "replica version is absent from the identity ledger"

        replicas.append(
            RolloutReplicaState(
                readiness=reason is None,
                current_version=version,
                current_snapshot_identity=identity,
                replica_id=info.get("replica_id") or info.get("run_id"),
                sync_state=sync_state,
                readiness_reason=reason,
            )
        )
    return RolloutPoolState(replicas=replicas)


def _require_fields(
    data: object,
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    if not isinstance(data, dict):
        raise HotLoadRequestError(f"{label} must be a JSON object")
    fields = set(data)
    missing = required - fields
    unknown = fields - required - optional
    if missing:
        raise HotLoadRequestError(f"{label} is missing fields {sorted(missing)!r}")
    if unknown:
        raise HotLoadRequestError(f"{label} has unknown fields {sorted(unknown)!r}")
