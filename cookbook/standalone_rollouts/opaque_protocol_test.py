from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.ledger import (
    DeltaFormats,
    IdentityLedger,
    LedgerCorruption,
)
from cookbook.standalone_rollouts.opaque_protocol import (
    HotLoadRequestError,
    parse_hot_load_payload,
    pool_state_from_server_infos,
    recover_frontdoor_state,
)


FORMATS = DeltaFormats(
    delta_encoding="xor",
    compression_format="zstd",
    checksum_format="xxh3-128",
)


class ParseHotLoadPayloadTest(unittest.TestCase):
    def test_parses_base_assertion_and_delta_defaults_without_coercion(self) -> None:
        base = parse_hot_load_payload({"identity": "base"})
        self.assertEqual(base.identity, "base")
        self.assertIsNone(base.previous_snapshot_identity)
        self.assertIsNone(base.formats)

        delta = parse_hot_load_payload(
            {
                "identity": "opaque-delta",
                "incremental_snapshot_metadata": {"previous_snapshot_identity": "base"},
                "reset_prompt_cache": "new_session",
            }
        )
        self.assertEqual(delta.previous_snapshot_identity, "base")
        self.assertEqual(delta.formats, DeltaFormats.defaults())

    def test_accepts_only_the_fixed_wire_schema(self) -> None:
        invalid_payloads = (
            None,
            [],
            {},
            {"identity": 123},
            {"identity": "../escape"},
            {"identity": "base", "run_id": "old-run"},
            {"identity": "delta", "incremental_snapshot_metadata": None},
            {"identity": "delta", "incremental_snapshot_metadata": []},
            {"identity": "delta", "incremental_snapshot_metadata": {}},
            {
                "identity": "delta",
                "incremental_snapshot_metadata": {"previous_snapshot_identity": 12},
            },
            {
                "identity": "delta",
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": "../base"
                },
            },
            {
                "identity": "delta",
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": "base",
                    "compression_format": None,
                },
            },
            {
                "identity": "delta",
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": "base",
                    "compression_format": "gzip",
                },
            },
            {
                "identity": "delta",
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": "base",
                    "delta_encoding": "xor",
                },
            },
            {
                "identity": "delta",
                "incremental_snapshot_metadata": {
                    "previous_snapshot_identity": "base",
                    "unexpected": True,
                },
            },
            {"identity": "base", "reset_prompt_cache": None},
            {"identity": "base", "reset_prompt_cache": "reuse"},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(HotLoadRequestError):
                    parse_hot_load_payload(payload)


class RecoveryTest(unittest.TestCase):
    def test_missing_ledger_seeds_only_at_base_pointer(self) -> None:
        recovery = recover_frontdoor_state(
            persisted_ledger=None,
            expected_base_identity="base",
            pointer=(None, 0),
        )
        self.assertEqual(
            recovery.ledger.to_dict(), IdentityLedger.new("base").to_dict()
        )
        self.assertTrue(recovery.save_ledger)
        self.assertEqual(recovery.pointer_to_write, 0)

        with self.assertRaises(LedgerCorruption):
            recover_frontdoor_state(
                persisted_ledger=None,
                expected_base_identity="base",
                pointer=(None, 1),
            )

    def test_existing_ledger_repairs_only_a_pointer_behind_its_head(self) -> None:
        ledger = IdentityLedger.new("base")
        ledger.append_delta("delta-a", "base", FORMATS)
        ledger.append_delta("delta-b", "delta-a", FORMATS)

        recovery = recover_frontdoor_state(
            persisted_ledger=ledger.to_dict(),
            expected_base_identity="base",
            pointer=(None, 1),
        )
        self.assertFalse(recovery.save_ledger)
        self.assertEqual(recovery.pointer_to_write, 2)

        current = recover_frontdoor_state(
            persisted_ledger=ledger.to_dict(),
            expected_base_identity="base",
            pointer=(None, 2),
        )
        self.assertIsNone(current.pointer_to_write)

    def test_run_scoped_or_ahead_pointer_is_corruption(self) -> None:
        ledger = IdentityLedger.new("base")
        for pointer in (("old-run", 0), (None, 1), (None, True), (None, -1)):
            with self.subTest(pointer=pointer):
                with self.assertRaises(LedgerCorruption):
                    recover_frontdoor_state(
                        persisted_ledger=ledger.to_dict(),
                        expected_base_identity="base",
                        pointer=pointer,
                    )


class PoolStateTest(unittest.TestCase):
    def test_ready_requires_idle_runless_and_a_known_ledger_identity(self) -> None:
        ledger = IdentityLedger.new("base")
        ledger.append_delta("delta-a", "base", FORMATS)
        state = pool_state_from_server_infos(
            [
                {"replica_id": "base", "current_version": 0, "sync_state": "IDLE"},
                {"replica_id": "delta", "current_version": 1, "sync_state": "IDLE"},
                {"replica_id": "bool", "current_version": True, "sync_state": "IDLE"},
                {"replica_id": "unknown", "current_version": 9, "sync_state": "IDLE"},
                {
                    "replica_id": "scoped",
                    "current_run_id": "old-run",
                    "current_version": 1,
                    "sync_state": "IDLE",
                },
                {
                    "replica_id": "error",
                    "current_version": 1,
                    "sync_state": "ERROR",
                    "last_sync_error": "boom",
                },
            ],
            ledger,
        )
        replicas = {replica.replica_id: replica for replica in state.replicas}
        self.assertEqual(replicas["base"].current_snapshot_identity, "base")
        self.assertTrue(replicas["base"].readiness)
        self.assertEqual(replicas["delta"].current_snapshot_identity, "delta-a")
        self.assertTrue(replicas["delta"].readiness)
        for replica_id in ("bool", "unknown", "scoped", "error"):
            self.assertFalse(replicas[replica_id].readiness, replica_id)


if __name__ == "__main__":
    unittest.main()
