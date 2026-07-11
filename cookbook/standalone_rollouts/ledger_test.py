from __future__ import annotations

import unittest

from cookbook.standalone_rollouts.ledger import IdentityLedger, LedgerEntry


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

    def test_resume_against_base_records_true_non_contiguous_lineage(self) -> None:
        # A delta whose parent is the original base while the chain is at v2:
        # version is still monotonic (v3), but base_version records the real
        # parent (0), so the non-contiguity is visible to the apply path.
        ledger = IdentityLedger()
        ledger.record("ckpt-base", previous=None)
        ledger.record("ckpt-100", previous="ckpt-base")
        ledger.record("ckpt-200", previous="ckpt-100")
        resume, _ = ledger.record("ckpt-resume", previous="ckpt-base")
        self.assertEqual(resume.version, 3)
        self.assertEqual(ledger.base_version_for("ckpt-resume"), 0)

    def test_unknown_parent_falls_back_to_base_version(self) -> None:
        ledger = IdentityLedger()
        entry, _ = ledger.record("ckpt-orphan", previous="never-seen")
        self.assertEqual(entry.version, 0)
        self.assertEqual(ledger.base_version_for("ckpt-orphan"), 0)


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

    def test_entries_expose_previous(self) -> None:
        ledger = IdentityLedger({"a": LedgerEntry(version=0, previous=None)})
        self.assertEqual(ledger.to_dict()["entries"]["a"], {"version": 0, "previous": None})


if __name__ == "__main__":
    unittest.main()
