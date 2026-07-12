from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.ledger import IdentityLedger, LedgerEntry, LedgerError


class RecordTest(unittest.TestCase):
    def test_first_identity_is_base_version(self) -> None:
        ledger = IdentityLedger()
        entry, is_new = ledger.record("ckpt-base", previous=None)
        self.assertEqual((entry.version, entry.previous, is_new), (0, None, True))

    def test_contiguous_deltas_mint_sequential_versions(self) -> None:
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        b, _ = ledger.record("ckpt-100", previous="ckpt-base")
        c, _ = ledger.record("ckpt-200", previous="ckpt-100")
        self.assertEqual((b.version, c.version), (1, 2))
        # base_version tracks lineage -> a contiguous chain (base = version - 1).
        self.assertEqual(ledger.base_version_for("ckpt-100"), 0)
        self.assertEqual(ledger.base_version_for("ckpt-200"), 1)

    def test_resignal_is_idempotent_and_does_not_mint(self) -> None:
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        first, is_new_1 = ledger.record("ckpt-100", previous="ckpt-base")
        again, is_new_2 = ledger.record("ckpt-100", previous="ckpt-base")
        self.assertTrue(is_new_1)
        self.assertFalse(is_new_2)
        self.assertEqual(first, again)
        # No phantom version was minted for the duplicate.
        self.assertIsNone(ledger.identity_for(2))

    def test_fork_from_non_head_parent_is_rejected(self) -> None:
        # A minted non-contiguous version would permanently block the linear
        # replay of everything after it, so a fork is refused at signal time.
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        ledger.record("ckpt-100", previous="ckpt-base")
        ledger.record("ckpt-200", previous="ckpt-100")
        with self.assertRaises(LedgerError):
            ledger.record("ckpt-resume", previous="ckpt-base")
        # The head is untouched and still extendable.
        entry, _ = ledger.record("ckpt-300", previous="ckpt-200")
        self.assertEqual(entry.version, 3)

    def test_delta_before_any_base_mints_v1_not_v0(self) -> None:
        # A delta whose parent was never signalled (the base booted from
        # BASE_CHECKPOINT) still mints v1 and treats its base as v0, so it is
        # applied as a delta rather than mistaken for the base and skipped.
        ledger = IdentityLedger()
        entry, _ = ledger.record("ckpt-orphan", previous="never-seen")
        self.assertEqual(entry.version, 1)
        self.assertEqual(ledger.base_version_for("ckpt-orphan"), 0)

    def test_second_full_snapshot_is_rejected_not_a_v0_takeover(self) -> None:
        # v0 is single-occupancy: silently repointing it would strand pollers
        # waiting on the first base and rewire the sidecar's weight_v000000 link.
        ledger = IdentityLedger()
        ledger.record("base-a", previous=None)
        with self.assertRaises(LedgerError):
            ledger.record("base-b", previous=None)
        # The first base re-signals idempotently.
        entry, is_new = ledger.record("base-a", previous=None)
        self.assertEqual((entry.version, is_new), (0, False))

    def test_unknown_parent_on_nonempty_ledger_is_rejected(self) -> None:
        # A typo'd parent must not be coerced to base_version=0 — the delta
        # would be applied against the wrong weights and serve garbage.
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        ledger.record("ckpt-1", previous="ckpt-base")
        with self.assertRaises(LedgerError):
            ledger.record("ckpt-2", previous="ckpt-l")  # typo of "ckpt-1"

    def test_self_parent_first_delta_is_rejected(self) -> None:
        # A self-parent slips past the unknown-parent check on an empty ledger
        # and would mint base_version == version, an index no replica can ever
        # apply — a permanently wedged pool. Reject it, empty ledger or not.
        ledger = IdentityLedger()
        with self.assertRaises(LedgerError):
            ledger.record("ckpt-1", previous="ckpt-1")
        # And after a base is signalled.
        ledger2 = IdentityLedger()
        ledger2.record("ckpt-base", previous=None)
        with self.assertRaises(LedgerError):
            ledger2.record("ckpt-2", previous="ckpt-2")


class LookupTest(unittest.TestCase):
    def test_version_and_identity_round_trip(self) -> None:
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        ledger.record("ckpt-100", previous="ckpt-base")
        self.assertEqual(ledger.version_for("ckpt-100"), 1)
        self.assertEqual(ledger.identity_for(1), "ckpt-100")
        self.assertIsNone(ledger.version_for("missing"))
        self.assertIsNone(ledger.identity_for(99))


class SerializationTest(unittest.TestCase):
    def test_to_from_dict_round_trip(self) -> None:
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        ledger.record("ckpt-100", previous="ckpt-base")
        restored = IdentityLedger.from_dict(ledger.to_dict())
        self.assertEqual(restored.version_for("ckpt-100"), 1)
        self.assertEqual(restored.identity_for(0), "ckpt-base")
        self.assertEqual(restored.base_version_for("ckpt-100"), 0)
        # Minting continues from the restored high-water mark.
        entry, is_new = restored.record("ckpt-200", previous="ckpt-100")
        self.assertEqual((entry.version, is_new), (2, True))

    def test_from_empty_dict(self) -> None:
        ledger = IdentityLedger.from_dict({})
        self.assertIsNone(ledger.version_for("anything"))
        entry, _ = ledger.record("ckpt-base", previous=None)
        self.assertEqual(entry.version, 0)

    def test_duplicate_versions_in_persisted_ledger_are_rejected(self) -> None:
        # A corrupt identities.json must not silently collapse the reverse map.
        with self.assertRaises(LedgerError):
            IdentityLedger.from_dict(
                {
                    "entries": {
                        "identity-a": {"version": 1, "previous": "base"},
                        "identity-b": {"version": 1, "previous": "base"},
                    }
                }
            )

    def test_entries_expose_previous(self) -> None:
        ledger = IdentityLedger({"a": LedgerEntry(version=0, previous=None)})
        self.assertEqual(ledger.to_dict()["entries"]["a"], {"version": 0, "previous": None})


if __name__ == "__main__":
    unittest.main()
